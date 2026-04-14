import datetime
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from config.options import TrainOptions
from dataset.utils import build_dataset
from solver.evaluation import evaluate_model
from solver.ssl_methods import SSLMethodRunner
from utils.configuration import save_configure
from utils.metrics import calculate_metrics
from utils.plot_metrics import MetricsPlotter


def build_train_loader(args, trainset):
    def worker_init_fn(worker_id):
        random.seed(args.experiment.seed + worker_id)

    if args.method_group.method_name == "fully_supervised":
        dataset = trainset.queue_dataset
        if hasattr(trainset, "get_labeled_and_unlabeled_indices"):
            labeled_indices, _ = trainset.get_labeled_and_unlabeled_indices()
            dataset = Subset(trainset.queue_dataset, labeled_indices)
        return DataLoader(
            dataset,
            batch_size=args.dataset.batch,
            shuffle=True,
            num_workers=args.dataset.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    labeled_indices, unlabeled_indices = trainset.get_labeled_and_unlabeled_indices()
    if len(unlabeled_indices) == 0:
        raise ValueError("Semi-supervised training requires at least one unlabeled sample")

    batch_sampler = trainset.batch_sampler_cls(
        labeled_indices,
        unlabeled_indices,
        args.dataset.batch,
        args.dataset.batch - args.dataset.labeled_bs,
    )
    return DataLoader(
        trainset.queue_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.dataset.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def _get_batch_tensor(batch, key):
    value = batch[key]
    if isinstance(value, dict):
        return value["data"]
    return value


def get_visual_batch(batch):
    if "source_weak" in batch:
        return _get_batch_tensor(batch, "source_weak"), _get_batch_tensor(batch, "label_aug")
    return _get_batch_tensor(batch, "source"), _get_batch_tensor(batch, "label")


def log_visuals(writer, batch, logits, step):
    with torch.no_grad():
        images, labels = get_visual_batch(batch)
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        logits = logits.detach().cpu()

        if logits.shape[0] == 0:
            return

        if logits.dim() == 4:
            writer.add_image("train/Image", images[0, 0:1], step)
            prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
            writer.add_image("train/Prediction", prediction[0] * 50, step)
            writer.add_image("train/GroundTruth", labels[0].unsqueeze(0) * 50, step)
            return

        center_index = images.shape[-1] // 2
        writer.add_image("train/Image", images[0, 0:1, :, :, center_index], step)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
        writer.add_image("train/Prediction", prediction[0, 0:1, :, :, center_index] * 50, step)
        writer.add_image("train/GroundTruth", labels[0, :, :, center_index].unsqueeze(0) * 50, step)


def save_checkpoint(path, runner, iteration, metric_value=None, metric_name="mean_dice", primary_key="model"):
    state = runner.state_dict()
    state["iteration"] = iteration
    state["model"] = runner.models[primary_key].state_dict()
    if metric_value is not None:
        state[metric_name] = metric_value
    torch.save(state, path)


def main():
    args = TrainOptions().parse()

    random.seed(args.experiment.seed)
    np.random.seed(args.experiment.seed)
    torch.manual_seed(args.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.experiment.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    current_time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    args.experiment.log_path = os.path.join("logs", f"{args.experiment.expname}_{current_time_str}")
    os.makedirs(args.experiment.log_path, exist_ok=True)
    os.makedirs(os.path.join(args.experiment.log_path, "checkpoint"), exist_ok=True)
    save_configure(args)

    runner = SSLMethodRunner(args)
    trainset = build_dataset(args, mode="train", transform_name=runner.train_transform_name)
    trainloader = build_train_loader(args, trainset)

    writer = SummaryWriter(args.experiment.log_path)
    plotter = MetricsPlotter(save_dir=os.path.join(args.experiment.log_path, "plots"), window_size=50)

    start_iteration = 0
    if args.experiment.ckpt:
        checkpoint = torch.load(args.experiment.ckpt, map_location=runner.device)
        runner.load_state_dict(checkpoint)
        start_iteration = checkpoint.get("iteration", 0)

    iterations_per_epoch = max(len(trainloader), 1)
    if args.experiment.num_epochs > 0:
        max_epochs = args.experiment.num_epochs
    else:
        max_epochs = math.ceil(args.experiment.max_iterations / iterations_per_epoch) + 1
    start_epoch = start_iteration // iterations_per_epoch
    iteration = start_iteration

    for epoch in range(start_epoch, max_epochs):
        runner.on_epoch_start(trainset)
        runner.train_mode()

        for batch in trainloader:
            if iteration >= args.experiment.max_iterations:
                break

            stats = runner.train_step(batch, iteration)
            metrics = calculate_metrics(
                stats["metric_logits"],
                stats["metric_labels"],
                num_classes=args.model.out_channels,
                threshold=0.5,
                include_distance_metrics=args.experiment.include_distance_metrics,
            )

            writer.add_scalar("Train/Loss/total", float(stats["loss"].item()), iteration + 1)
            writer.add_scalar("Train/Metrics/dice", metrics["dice"], iteration + 1)
            writer.add_scalar("Train/Metrics/iou", metrics["iou"], iteration + 1)
            writer.add_scalar("Train/Metrics/precision", metrics["precision"], iteration + 1)
            writer.add_scalar("Train/Metrics/recall", metrics["recall"], iteration + 1)
            writer.add_scalar("Train/Metrics/specificity", metrics["specificity"], iteration + 1)
            writer.add_scalar("Train/Metrics/f1", metrics["f1"], iteration + 1)
            writer.add_scalar("Train/Metrics/accuracy", metrics["accuracy"], iteration + 1)
            writer.add_scalar("Train/lr", float(stats["lr"]), iteration + 1)

            metric_dict = {
                "total_loss": float(stats["loss"].item()),
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }

            for key, value in stats.items():
                if key in {"loss", "metric_logits", "metric_labels", "lr"}:
                    continue
                if torch.is_tensor(value):
                    writer.add_scalar(f"Train/{key}", float(value.item()), iteration + 1)
                    metric_dict[key] = float(value.item())

            plotter.update(iteration + 1, **metric_dict)
            if (iteration + 1) % 20 == 0:
                plotter.plot(show_ma=True)
                plotter.save_csv()

            if (iteration + 1) % args.experiment.log_every == 0:
                print(
                    f"Iter {iteration + 1:06d}/{args.experiment.max_iterations:06d} | "
                    f"loss {float(stats['loss'].item()):.5f} | "
                    f"dice {metrics['dice']:.4f} | "
                    f"iou {metrics['iou']:.4f} | "
                    f"f1 {metrics['f1']:.4f}"
                )

            if (iteration + 1) % args.experiment.vis_every == 0:
                log_visuals(writer, batch, stats["metric_logits"], iteration + 1)

            if (iteration + 1) % args.experiment.val_every == 0:
                eval_models = runner.get_eval_models()
                for model_key, model in eval_models.items():
                    eval_result = evaluate_model(args, model)
                    writer.add_scalar(f"Val/{model_key}/mean_dice", eval_result["mean_dice"], iteration + 1)
                    writer.add_scalar(f"Val/{model_key}/mean_hd95", eval_result["mean_hd95"], iteration + 1)
                    best_so_far = runner.best_scores.get(model_key, 0.0)
                    if eval_result["mean_dice"] > best_so_far:
                        runner.best_scores[model_key] = eval_result["mean_dice"]
                        save_checkpoint(
                            os.path.join(args.experiment.log_path, "checkpoint", f"{model_key}_best.pt"),
                            runner,
                            iteration + 1,
                            metric_value=eval_result["mean_dice"],
                            primary_key=model_key,
                        )
                    print(
                        f"Validation {model_key} | iter {iteration + 1:06d} | "
                        f"dice {eval_result['mean_dice']:.4f} | hd95 {eval_result['mean_hd95']:.4f}"
                    )
                runner.train_mode()

            if (iteration + 1) % args.experiment.save_every == 0:
                save_checkpoint(
                    os.path.join(args.experiment.log_path, "checkpoint", f"iter_{iteration + 1:06d}.pt"),
                    runner,
                    iteration + 1,
                )

            iteration += 1

        if iteration >= args.experiment.max_iterations:
            break

    save_checkpoint(
        os.path.join(args.experiment.log_path, "checkpoint", "last.pt"),
        runner,
        iteration,
    )
    writer.close()


if __name__ == "__main__":
    main()
