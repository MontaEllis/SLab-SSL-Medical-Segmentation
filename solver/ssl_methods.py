from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from losses import build_supervised_loss_terms, compute_kl_loss, entropy_loss, softmax_mse_loss
from models.model_utils import get_discriminator, get_model, initialize_weights
from solver.optim import build_optimizer
from solver.scheduler import (
    adjust_multiple_poly_learning_rates,
    adjust_poly_learning_rate,
    get_current_consistency_weight,
    sigmoid_rampup,
)
from utils.ema import update_ema_variables


class SSLMethodRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method_name = args.method_group.method_name
        self.num_classes = args.model.out_channels
        self.batch_size = args.dataset.batch
        self.labeled_bs = args.dataset.labeled_bs
        self.max_iterations = args.experiment.max_iterations
        self.train_transform_name = getattr(args.method_group, "train_transform", "random")

        self.ce_loss, self.dice_loss = build_supervised_loss_terms(args)
        self.models = {}
        self.optimizers = {}
        self.best_scores = {}
        self._poly_optimizer_names = []

        self._build_components()

    def _build_components(self):
        self.models["model"] = get_model(self.args).to(self.device)
        if getattr(self.args.method_group, "primary_init", None):
            initialize_weights(self.models["model"], self.args.method_group.primary_init)

        self.optimizers["model"] = build_optimizer(self.args, self.models["model"].parameters())
        self._poly_optimizer_names.append("model")

        if self.method_name in {
            "cross_pseudo_supervision",
            "regularized_dropout",
            "cross_teaching_between_cnn_transformer",
        }:
            secondary_name = self.args.model.secondary_model_name
            if secondary_name is None:
                secondary_name = getattr(self.args.method_group, "secondary_model_name", None)
            if secondary_name is None:
                secondary_name = self.args.model.model_name
            self.models["model2"] = get_model(self.args, model_name=secondary_name).to(self.device)
            if getattr(self.args.method_group, "secondary_init", None):
                initialize_weights(self.models["model2"], self.args.method_group.secondary_init)
            self.optimizers["model2"] = build_optimizer(self.args, self.models["model2"].parameters())
            self._poly_optimizer_names.append("model2")

        if self.method_name in {
            "mean_teacher",
            "interpolation_consistency_training",
            "uncertainty_aware_mean_teacher",
            "fixmatch_standard_augs",
        }:
            self.models["ema"] = deepcopy(self.models["model"]).to(self.device)
            for parameter in self.models["ema"].parameters():
                parameter.detach_()

        if self.method_name == "adversarial_network":
            self.models["discriminator"] = get_discriminator(self.args).to(self.device)
            dan_lr = getattr(self.args.method_group, "discriminator_lr", 1e-4)
            self.optimizers["discriminator"] = torch.optim.Adam(
                self.models["discriminator"].parameters(),
                lr=dan_lr,
                betas=(0.9, 0.99),
            )

        self._validate_method_constraints()

    def _validate_method_constraints(self):
        method_2d_only = {
            "cross_consistency_training",
            "deep_co_training",
            "fixmatch_standard_augs",
            "cross_teaching_between_cnn_transformer",
        }
        if self.args.model.dimension != "2d" and self.method_name in method_2d_only:
            raise ValueError(f"{self.method_name} is only implemented for 2D models")

    def on_epoch_start(self, trainset):
        return

    def train_mode(self):
        for name, model in self.models.items():
            if name == "discriminator":
                continue
            model.train()
        if "discriminator" in self.models:
            self.models["discriminator"].train()

    def eval_mode(self):
        for model in self.models.values():
            model.eval()

    def get_eval_models(self):
        if self.method_name in {
            "cross_pseudo_supervision",
            "regularized_dropout",
            "cross_teaching_between_cnn_transformer",
        }:
            return {
                "model": self.models["model"],
                "model2": self.models["model2"],
            }
        return {"model": self.models["model"]}

    def primary_model(self):
        return self.models["model"]

    def state_dict(self):
        return {
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "method_name": self.method_name,
        }

    def load_state_dict(self, checkpoint):
        for name, state in checkpoint.get("models", {}).items():
            if name in self.models:
                self.models[name].load_state_dict(state, strict=False)
        for name, state in checkpoint.get("optimizers", {}).items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)

    @staticmethod
    def _get_batch_tensor(batch, key):
        value = batch[key]
        if isinstance(value, dict):
            return value["data"]
        return value

    def _split_standard_batch(self, batch):
        image = self._get_batch_tensor(batch, "source").to(self.device)
        label = self._get_batch_tensor(batch, "label").to(self.device)
        return image, label

    def _main_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _auxiliary_outputs(self, output, num_outputs):
        if isinstance(output, (tuple, list)):
            outputs = list(output)
            if len(outputs) >= num_outputs:
                return outputs[:num_outputs]
            main_output = outputs[0]
        else:
            main_output = output
            outputs = [main_output]

        if main_output.dim() < 4:
            raise ValueError("Segmentation logits are expected to have spatial dimensions")

        drop_rates = [0.1, 0.2, 0.3, 0.4]
        for index in range(len(outputs), num_outputs):
            dropout_p = drop_rates[min(index - 1, len(drop_rates) - 1)]
            auxiliary = F.dropout(main_output, p=dropout_p, training=True)
            if auxiliary.shape == main_output.shape and torch.allclose(auxiliary, main_output):
                noise = torch.randn_like(main_output) * (0.05 * index)
                auxiliary = main_output + noise
            outputs.append(auxiliary)
        return outputs[:num_outputs]

    def _supervised_loss(self, logits, labels):
        loss_ce = self.ce_loss(logits, labels.long())
        loss_dice = self.dice_loss(torch.softmax(logits, dim=1), labels.unsqueeze(1))
        total = (
            self.args.loss_group.supervised_ce_weight * loss_ce
            + self.args.loss_group.supervised_dice_weight * loss_dice
        )
        return loss_ce, loss_dice, total

    def _consistency_weight(self, iteration):
        return get_current_consistency_weight(self.args, iteration)

    def _step_poly_lr(self, iteration):
        if self.args.schedulers.scheduler_type != "poly_lr":
            return self.args.optimizer.lr
        next_iteration = min(iteration + 1, self.max_iterations)
        optimizers = [self.optimizers[name] for name in self._poly_optimizer_names]
        if len(optimizers) == 1:
            return adjust_poly_learning_rate(
                optimizers[0],
                self.args.optimizer.lr,
                next_iteration,
                self.max_iterations,
                power=self.args.schedulers.power,
            )
        return adjust_multiple_poly_learning_rates(
            optimizers,
            self.args.optimizer.lr,
            next_iteration,
            self.max_iterations,
            power=self.args.schedulers.power,
        )

    def _teacher_update(self, iteration):
        if "ema" not in self.models:
            return
        update_ema_variables(
            self.models["model"],
            self.models["ema"],
            self.args.ema_group.ema_decay,
            iteration,
        )

    def _normalize_probabilities(self, tensor):
        min_value = tensor.min(1, keepdim=True)[0]
        max_value = tensor.max(1, keepdim=True)[0]
        normalized = tensor - min_value
        return normalized / torch.clamp(max_value, min=1e-6)

    def _complementary_loss(self, weak_probabilities, strong_probabilities):
        flat = torch.reshape(
            strong_probabilities,
            (
                strong_probabilities.shape[0],
                self.num_classes,
                int(np.prod(strong_probabilities.shape[2:])),
            ),
        )
        entropy = torch.distributions.Categorical(probs=flat).entropy()
        entropy = entropy / np.log(int(np.prod(strong_probabilities.shape[2:])))
        adaptive_weight = torch.mean(1 - entropy)
        complementary_labels = torch.argmin(weak_probabilities.detach(), dim=1)
        complementary_loss = adaptive_weight * self.ce_loss(1 - strong_probabilities, complementary_labels)
        return complementary_loss, adaptive_weight

    def train_step(self, batch, iteration):
        if self.method_name == "fully_supervised":
            return self._train_fully_supervised(batch, iteration)
        if self.method_name == "mean_teacher":
            return self._train_mean_teacher(batch, iteration)
        if self.method_name == "entropy_minimization":
            return self._train_entropy_minimization(batch, iteration)
        if self.method_name == "adversarial_network":
            return self._train_adversarial(batch, iteration)
        if self.method_name == "interpolation_consistency_training":
            return self._train_ict(batch, iteration)
        if self.method_name == "cross_pseudo_supervision":
            return self._train_cps(batch, iteration)
        if self.method_name == "cross_consistency_training":
            return self._train_cct(batch, iteration)
        if self.method_name == "uncertainty_aware_mean_teacher":
            return self._train_uamt(batch, iteration)
        if self.method_name == "uncertainty_rectified_pyramid_consistency":
            return self._train_urpc(batch, iteration)
        if self.method_name == "regularized_dropout":
            return self._train_regularized_dropout(batch, iteration)
        if self.method_name == "deep_co_training":
            return self._train_deep_co_training(batch, iteration)
        if self.method_name == "fixmatch_standard_augs":
            return self._train_fixmatch_standard(batch, iteration)
        if self.method_name == "cross_teaching_between_cnn_transformer":
            return self._train_cross_teaching(batch, iteration)
        raise ValueError(f"Unsupported method: {self.method_name}")

    def _train_fully_supervised(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        logits = self._main_output(self.models["model"](image))
        loss_ce, loss_dice, loss = self._supervised_loss(logits, label)

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "metric_logits": logits.detach(),
            "metric_labels": label.unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_mean_teacher(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        unlabeled_image = image[self.labeled_bs :]

        noise = torch.clamp(torch.randn_like(unlabeled_image) * 0.1, -0.2, 0.2)
        teacher_inputs = unlabeled_image + noise

        logits = self._main_output(self.models["model"](image))
        probabilities = torch.softmax(logits, dim=1)
        with torch.no_grad():
            teacher_logits = self._main_output(self.models["ema"](teacher_inputs))
            teacher_probabilities = torch.softmax(teacher_logits, dim=1)

        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)
        if unlabeled_image.shape[0] == 0 or iteration < getattr(self.args.method_group, "consistency_start", 0):
            consistency_loss = torch.tensor(0.0, device=self.device)
        else:
            consistency_loss = torch.mean((probabilities[self.labeled_bs :] - teacher_probabilities) ** 2)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self._teacher_update(iteration)
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_entropy_minimization(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        logits = self._main_output(self.models["model"](image))
        probabilities = torch.softmax(logits, dim=1)

        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)
        consistency_loss = entropy_loss(probabilities, self.num_classes)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_adversarial(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        unlabeled_count = image.shape[0] - self.labeled_bs
        fool_target = torch.ones(unlabeled_count, dtype=torch.long, device=self.device)
        dan_target = torch.zeros(image.shape[0], dtype=torch.long, device=self.device)
        dan_target[: self.labeled_bs] = 1

        self.models["model"].train()
        self.models["discriminator"].eval()

        logits = self._main_output(self.models["model"](image))
        probabilities = torch.softmax(logits, dim=1)
        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)
        if unlabeled_count > 0:
            dan_outputs = self.models["discriminator"](probabilities[self.labeled_bs :], image[self.labeled_bs :])
            consistency_loss = F.cross_entropy(dan_outputs, fool_target)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()

        self.models["model"].eval()
        self.models["discriminator"].train()
        with torch.no_grad():
            logits = self._main_output(self.models["model"](image))
            probabilities = torch.softmax(logits, dim=1)
        dan_outputs = self.models["discriminator"](probabilities, image)
        dan_loss = F.cross_entropy(dan_outputs, dan_target)
        self.optimizers["discriminator"].zero_grad()
        dan_loss.backward()
        self.optimizers["discriminator"].step()

        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "discriminator_loss": dan_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_ict(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        unlabeled_image = image[self.labeled_bs :]
        labeled_image = image[: self.labeled_bs]
        half = unlabeled_image.shape[0] // 2

        if half == 0:
            return self._train_mean_teacher(batch, iteration)

        mix_shape = [half] + [1] * (unlabeled_image.dim() - 1)
        mix_factors = np.random.beta(self.args.method_group.ict_alpha, self.args.method_group.ict_alpha, size=mix_shape)
        mix_factors = torch.tensor(mix_factors, dtype=torch.float32, device=self.device)
        unlabeled_0 = unlabeled_image[:half]
        unlabeled_1 = unlabeled_image[half : half * 2]
        mixed_unlabeled = unlabeled_0 * (1.0 - mix_factors) + unlabeled_1 * mix_factors

        model_inputs = torch.cat([labeled_image, mixed_unlabeled], dim=0)
        logits = self._main_output(self.models["model"](model_inputs))
        probabilities = torch.softmax(logits, dim=1)
        with torch.no_grad():
            teacher_prob_0 = torch.softmax(self._main_output(self.models["ema"](unlabeled_0)), dim=1)
            teacher_prob_1 = torch.softmax(self._main_output(self.models["ema"](unlabeled_1)), dim=1)
            mixed_teacher = teacher_prob_0 * (1.0 - mix_factors) + teacher_prob_1 * mix_factors

        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)
        consistency_loss = torch.mean((probabilities[self.labeled_bs :] - mixed_teacher) ** 2)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self._teacher_update(iteration)
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_cps(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        logits_1 = self._main_output(self.models["model"](image))
        logits_2 = self._main_output(self.models["model2"](image))
        probabilities_1 = torch.softmax(logits_1, dim=1)
        probabilities_2 = torch.softmax(logits_2, dim=1)

        _, _, supervised_1 = self._supervised_loss(logits_1[: self.labeled_bs], label[: self.labeled_bs])
        _, _, supervised_2 = self._supervised_loss(logits_2[: self.labeled_bs], label[: self.labeled_bs])
        pseudo_1 = torch.argmax(probabilities_1[self.labeled_bs :].detach(), dim=1)
        pseudo_2 = torch.argmax(probabilities_2[self.labeled_bs :].detach(), dim=1)
        consistency_weight = self._consistency_weight(iteration)
        pseudo_loss_1 = self.ce_loss(logits_1[self.labeled_bs :], pseudo_2) if pseudo_2.numel() > 0 else torch.tensor(0.0, device=self.device)
        pseudo_loss_2 = self.ce_loss(logits_2[self.labeled_bs :], pseudo_1) if pseudo_1.numel() > 0 else torch.tensor(0.0, device=self.device)

        model1_loss = supervised_1 + consistency_weight * pseudo_loss_1
        model2_loss = supervised_2 + consistency_weight * pseudo_loss_2
        loss = model1_loss + model2_loss

        self.optimizers["model"].zero_grad()
        self.optimizers["model2"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self.optimizers["model2"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "model1_loss": model1_loss.detach(),
            "model2_loss": model2_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits_1[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_cct(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        outputs = self._auxiliary_outputs(self.models["model"](image), 4)
        probabilities = [torch.softmax(output, dim=1) for output in outputs]
        supervised_terms = []
        for output, probability in zip(outputs, probabilities):
            loss_ce, loss_dice, total = self._supervised_loss(output[: self.labeled_bs], label[: self.labeled_bs])
            supervised_terms.extend([loss_ce, loss_dice])
        supervised_loss = sum(supervised_terms) / len(supervised_terms)

        consistency_weight = self._consistency_weight(iteration)
        consistency_losses = []
        for auxiliary_prob in probabilities[1:]:
            if auxiliary_prob[self.labeled_bs :].numel() == 0:
                consistency_losses.append(torch.tensor(0.0, device=self.device))
            else:
                consistency_losses.append(torch.mean((probabilities[0][self.labeled_bs :] - auxiliary_prob[self.labeled_bs :]) ** 2))
        consistency_loss = sum(consistency_losses) / len(consistency_losses)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "supervised_loss": supervised_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": outputs[0][: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_uamt(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        unlabeled_image = image[self.labeled_bs :]

        noise = torch.clamp(torch.randn_like(unlabeled_image) * 0.1, -0.2, 0.2)
        teacher_inputs = unlabeled_image + noise
        logits = self._main_output(self.models["model"](image))
        probabilities = torch.softmax(logits, dim=1)
        with torch.no_grad():
            teacher_logits = self._main_output(self.models["ema"](teacher_inputs))

        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)

        if unlabeled_image.shape[0] == 0:
            consistency_loss = torch.tensor(0.0, device=self.device)
        else:
            uncertainty_passes = getattr(self.args.method_group, "uncertainty_passes", 8)
            repeat_image = unlabeled_image.repeat([2] + [1] * (unlabeled_image.dim() - 1))
            stride = repeat_image.shape[0] // 2
            prediction_bank = torch.zeros(
                [stride * uncertainty_passes, self.num_classes] + list(unlabeled_image.shape[2:]),
                device=self.device,
            )
            for idx in range(uncertainty_passes // 2):
                ema_inputs = repeat_image + torch.clamp(torch.randn_like(repeat_image) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    prediction_bank[2 * stride * idx : 2 * stride * (idx + 1)] = self._main_output(
                        self.models["ema"](ema_inputs)
                    )
            prediction_bank = F.softmax(prediction_bank, dim=1)
            prediction_bank = prediction_bank.reshape(
                uncertainty_passes,
                stride,
                self.num_classes,
                *unlabeled_image.shape[2:],
            )
            prediction_bank = torch.mean(prediction_bank, dim=0)
            uncertainty = -1.0 * torch.sum(prediction_bank * torch.log(prediction_bank + 1e-6), dim=1, keepdim=True)

            consistency_distance = softmax_mse_loss(logits[self.labeled_bs :], teacher_logits)
            threshold = (
                getattr(self.args.method_group, "uncertainty_threshold_base", 0.75)
                + getattr(self.args.method_group, "uncertainty_threshold_scale", 0.25)
                * sigmoid_rampup(iteration, self.max_iterations)
            ) * np.log(max(self.num_classes, 2))
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_distance) / (
                self.num_classes * torch.sum(mask) + 1e-16
            )

        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self._teacher_update(iteration)
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_urpc(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        outputs = self._auxiliary_outputs(self.models["model"](image), 4)
        probabilities = [torch.softmax(output, dim=1) for output in outputs]
        supervised_terms = []
        for output, probability in zip(outputs, probabilities):
            loss_ce, loss_dice, _ = self._supervised_loss(output[: self.labeled_bs], label[: self.labeled_bs])
            supervised_terms.extend([loss_ce, loss_dice])
        supervised_loss = sum(supervised_terms) / len(supervised_terms)

        preds = sum(probabilities) / len(probabilities)
        kl_distance = torch.nn.KLDivLoss(reduction="none")
        consistency_weight = self._consistency_weight(iteration)
        consistency_terms = []
        for probability in probabilities:
            if probability[self.labeled_bs :].numel() == 0:
                consistency_terms.append(torch.tensor(0.0, device=self.device))
                continue
            variance = torch.sum(
                kl_distance(torch.log(probability[self.labeled_bs :] + 1e-6), preds[self.labeled_bs :]),
                dim=1,
                keepdim=True,
            )
            exp_variance = torch.exp(-variance)
            consistency_distance = (preds[self.labeled_bs :] - probability[self.labeled_bs :]) ** 2
            consistency_terms.append(
                torch.mean(consistency_distance * exp_variance) / (torch.mean(exp_variance) + 1e-8)
                + torch.mean(variance)
            )
        consistency_loss = sum(consistency_terms) / len(consistency_terms)
        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "supervised_loss": supervised_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": outputs[0][: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_regularized_dropout(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        logits_1 = self._main_output(self.models["model"](image))
        logits_2 = self._main_output(self.models["model2"](image))
        probabilities_1 = torch.softmax(logits_1, dim=1)
        probabilities_2 = torch.softmax(logits_2, dim=1)

        _, _, supervised_1 = self._supervised_loss(logits_1[: self.labeled_bs], label[: self.labeled_bs])
        _, _, supervised_2 = self._supervised_loss(logits_2[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)
        if logits_1[self.labeled_bs :].numel() == 0:
            r_drop_loss = torch.tensor(0.0, device=self.device)
        else:
            r_drop_loss = compute_kl_loss(logits_1[self.labeled_bs :], logits_2[self.labeled_bs :])
        loss = supervised_1 + supervised_2 + consistency_weight * r_drop_loss

        self.optimizers["model"].zero_grad()
        self.optimizers["model2"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self.optimizers["model2"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "model1_loss": supervised_1.detach(),
            "model2_loss": supervised_2.detach(),
            "r_drop_loss": r_drop_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits_1[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_deep_co_training(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        unlabeled_image = image[self.labeled_bs :]

        logits = self._main_output(self.models["model"](image))
        probabilities = torch.softmax(logits, dim=1)
        loss_ce, loss_dice, supervised_loss = self._supervised_loss(logits[: self.labeled_bs], label[: self.labeled_bs])
        consistency_weight = self._consistency_weight(iteration)

        if unlabeled_image.shape[0] == 0:
            consistency_loss = torch.tensor(0.0, device=self.device)
        else:
            rotation_k = np.random.randint(0, 4)
            rotated_unlabeled = torch.rot90(unlabeled_image, rotation_k, dims=[-2, -1])
            rotated_logits = self._main_output(self.models["model"](rotated_unlabeled))
            rotated_probabilities = torch.softmax(rotated_logits, dim=1)
            rotated_student = torch.rot90(probabilities[self.labeled_bs :], rotation_k, dims=[-2, -1])
            consistency_loss = 0.5 * (
                torch.mean((rotated_probabilities.detach() - rotated_student) ** 2)
                + torch.mean((rotated_probabilities - rotated_student.detach()) ** 2)
            )

        loss = supervised_loss + consistency_weight * consistency_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": consistency_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_fixmatch_standard(self, batch, iteration):
        weak_batch = self._get_batch_tensor(batch, "source_weak").to(self.device)
        strong_batch = self._get_batch_tensor(batch, "source_strong").to(self.device)
        label_batch = self._get_batch_tensor(batch, "label_aug").to(self.device)

        logits_weak = self._main_output(self.models["model"](weak_batch))
        logits_strong = self._main_output(self.models["model"](strong_batch))
        probabilities_weak = torch.softmax(logits_weak, dim=1)
        probabilities_strong = torch.softmax(logits_strong, dim=1)

        pseudo_mask = (self._normalize_probabilities(probabilities_weak) > self.args.method_group.confidence_threshold).float()
        masked_weak = probabilities_weak * pseudo_mask
        pseudo_outputs = torch.argmax(masked_weak[self.labeled_bs :].detach(), dim=1)
        consistency_weight = self._consistency_weight(iteration)

        loss_ce = self.ce_loss(logits_weak[: self.labeled_bs], label_batch[: self.labeled_bs].long())
        loss_dice = self.dice_loss(probabilities_weak[: self.labeled_bs], label_batch[: self.labeled_bs].unsqueeze(1))
        supervised_loss = loss_ce + loss_dice

        comp_loss, adaptive_weight = self._complementary_loss(probabilities_weak, probabilities_strong)
        if pseudo_outputs.numel() > 0:
            unsupervised_loss = (
                self.ce_loss(logits_strong[self.labeled_bs :], pseudo_outputs)
                + self.dice_loss(probabilities_strong[self.labeled_bs :], pseudo_outputs.unsqueeze(1))
                + adaptive_weight * comp_loss
            )
        else:
            unsupervised_loss = torch.tensor(0.0, device=self.device)

        loss = supervised_loss + consistency_weight * unsupervised_loss

        self.optimizers["model"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self._teacher_update(iteration)
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "consistency_loss": unsupervised_loss.detach(),
            "adaptive_weight": adaptive_weight.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits_weak[: self.labeled_bs].detach(),
            "metric_labels": label_batch[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }

    def _train_cross_teaching(self, batch, iteration):
        image, label = self._split_standard_batch(batch)
        logits_1 = self._main_output(self.models["model"](image))
        logits_2 = self._main_output(self.models["model2"](image))
        probabilities_1 = torch.softmax(logits_1, dim=1)
        probabilities_2 = torch.softmax(logits_2, dim=1)

        _, _, supervised_1 = self._supervised_loss(logits_1[: self.labeled_bs], label[: self.labeled_bs])
        _, _, supervised_2 = self._supervised_loss(logits_2[: self.labeled_bs], label[: self.labeled_bs])
        pseudo_1 = torch.argmax(probabilities_1[self.labeled_bs :].detach(), dim=1)
        pseudo_2 = torch.argmax(probabilities_2[self.labeled_bs :].detach(), dim=1)
        consistency_weight = self._consistency_weight(iteration)

        if pseudo_1.numel() > 0:
            pseudo_supervision_1 = self.dice_loss(probabilities_1[self.labeled_bs :], pseudo_2.unsqueeze(1))
            pseudo_supervision_2 = self.dice_loss(probabilities_2[self.labeled_bs :], pseudo_1.unsqueeze(1))
        else:
            pseudo_supervision_1 = torch.tensor(0.0, device=self.device)
            pseudo_supervision_2 = torch.tensor(0.0, device=self.device)

        model1_loss = supervised_1 + consistency_weight * pseudo_supervision_1
        model2_loss = supervised_2 + consistency_weight * pseudo_supervision_2
        loss = model1_loss + model2_loss

        self.optimizers["model"].zero_grad()
        self.optimizers["model2"].zero_grad()
        loss.backward()
        self.optimizers["model"].step()
        self.optimizers["model2"].step()
        lr = self._step_poly_lr(iteration)

        return {
            "loss": loss.detach(),
            "model1_loss": model1_loss.detach(),
            "model2_loss": model2_loss.detach(),
            "consistency_weight": torch.tensor(consistency_weight, device=self.device),
            "metric_logits": logits_1[: self.labeled_bs].detach(),
            "metric_labels": label[: self.labeled_bs].unsqueeze(1).detach(),
            "lr": lr,
        }
