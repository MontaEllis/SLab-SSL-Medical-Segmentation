import argparse
import os

import yaml
from munch import Munch, munchify


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def merge_munch(source: Munch, target: Munch) -> Munch:
    for key, value in source.items():
        if key not in target:
            target[key] = value
            continue
        if isinstance(value, Munch) and isinstance(target[key], Munch):
            merge_munch(value, target[key])
    return target


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SSL Medical Image Segmentation")
        self.initialized = False
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def initialize(self):
        dataset = self.parser.add_argument_group("dataset")
        dataset.add_argument("--dataset_name", type=str, default="acdc", help="dataset name")
        dataset.add_argument("--dataset_path", type=str, default="../data/ACDC", help="path to the dataset")
        dataset.add_argument("--batch", type=int, default=24, help="batch size")
        dataset.add_argument("--labeled_bs", type=int, default=12, help="labeled samples per batch")
        dataset.add_argument("--labeled_num", type=int, default=7, help="number of labeled patients/cases")
        dataset.add_argument("--total_labeled_num", type=int, default=250, help="total available train cases")
        dataset.add_argument("--num_workers", type=int, default=4, help="num_workers")
        dataset.add_argument("--patch_size", nargs="+", type=int, default=[256, 256], help="patch size")
        dataset.add_argument("--image_size", type=int, default=256, help="2D image size alias")
        dataset.add_argument("--stride_xy", type=int, default=64, help="3D validation stride in xy")
        dataset.add_argument("--stride_z", type=int, default=64, help="3D validation stride in z")

        experiment = self.parser.add_argument_group("experiment")
        experiment.add_argument("--expname", type=str, default="debug", help="experiment name")
        experiment.add_argument("--ckpt", type=str, default=None, help="checkpoint to resume")
        experiment.add_argument("--conf", type=str, default=None, help="path to the config yaml")
        experiment.add_argument("--seed", type=int, default=1337, help="seed")
        experiment.add_argument("--num_epochs", type=int, default=0, help="optional explicit epoch limit")
        experiment.add_argument("--max_iterations", type=int, default=30000, help="maximum training iterations")
        experiment.add_argument("--val_every", type=int, default=200, help="validation interval in iterations")
        experiment.add_argument("--save_every", type=int, default=3000, help="checkpoint interval in iterations")
        experiment.add_argument("--vis_every", type=int, default=50, help="visualization interval in iterations")
        experiment.add_argument("--log_every", type=int, default=10, help="log interval in iterations")
        experiment.add_argument("--include_distance_metrics", type=str2bool, default=False, help="whether to compute hd/asd during train logging")

        optimizer = self.parser.add_argument_group("optimizer")
        optimizer.add_argument("--optimizer_type", type=str, default="sgd", help="optimizer type")
        optimizer.add_argument("--lr", type=float, default=0.01, help="learning rate")

        schedulers = self.parser.add_argument_group("schedulers")
        schedulers.add_argument("--scheduler_type", type=str, default="poly_lr", help="scheduler type")
        schedulers.add_argument("--warmup", type=str2bool, default=False, help="warmup enabled")

        model = self.parser.add_argument_group("model")
        model.add_argument("--model_name", type=str, default="unet_2d", help="primary model name")
        model.add_argument("--secondary_model_name", type=str, default=None, help="secondary model name for dual-model methods")
        model.add_argument("--dimension", type=str, default="2d", help="model dimension")
        model.add_argument("--in_channels", type=int, default=1, help="in_channels")
        model.add_argument("--out_channels", type=int, default=4, help="out_channels")
        model.add_argument("--model_cfg", type=str, default=None, help="optional transformer config path")
        model.add_argument("--use_pretrained", type=str2bool, default=False, help="whether to load pretrained transformer weights")
        model.add_argument("--pretrained_ckpt", type=str, default=None, help="optional pretrained checkpoint path")

        loss = self.parser.add_argument_group("loss_group")
        loss.add_argument(
            "--loss_type",
            type=str,
            default="ssl_multiclass",
            help="SSL supervised loss preset: ssl_multiclass or ssl_binary",
        )

        method = self.parser.add_argument_group("method_group")
        method.add_argument("--method_name", type=str, default="mean_teacher", help="semi-supervised method name")

        ema = self.parser.add_argument_group("ema_group")
        ema.add_argument("--ema", type=str2bool, default=False, help="ema enabled")
        ema.add_argument("--ema_decay", type=float, default=0.99, help="ema decay")

        self.initialized = True

    def _load_yaml_into_opt(self, relative_path):
        config_path = os.path.join(self.project_root, relative_path)
        with open(config_path, "r") as handle:
            loaded = yaml.safe_load(handle) or {}
        return munchify(loaded)

    def parse(self, argv=None):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()

        args = self.parser.parse_args(args=argv)

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                self.opt[title][action.dest] = getattr(args, action.dest)

            if title == "optimizer":
                optim_cfg = self._load_yaml_into_opt(f"config/optimizer/{self.opt[title].optimizer_type}.yaml")
                self.opt = merge_munch(optim_cfg, self.opt)
            elif title == "schedulers":
                sched_cfg = self._load_yaml_into_opt(f"config/scheduler/{self.opt[title].scheduler_type}.yaml")
                self.opt = merge_munch(sched_cfg, self.opt)
            elif title == "model":
                dim_folder = "two_d" if self.opt[title].dimension == "2d" else "three_d"
                arch_cfg = self._load_yaml_into_opt(
                    f"config/arch/{dim_folder}/{self.opt[title].model_name}.yaml"
                )
                self.opt = merge_munch(arch_cfg, self.opt)
            elif title == "loss_group":
                loss_cfg = self._load_yaml_into_opt(f"config/losses/{self.opt[title].loss_type}.yaml")
                self.opt = merge_munch(loss_cfg, self.opt)
            elif title == "method_group":
                method_cfg = self._load_yaml_into_opt(f"config/methods/{self.opt[title].method_name}.yaml")
                self.opt = merge_munch(method_cfg, self.opt)

        if self.opt.experiment.conf is not None:
            with open(self.opt.experiment.conf, "r") as handle:
                exp_cfg = yaml.safe_load(handle) or {}
            self.opt = merge_munch(munchify(exp_cfg), self.opt)

        return self.opt

    def parse_from_yaml(self, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r") as handle:
            config = yaml.safe_load(handle) or {}
        self.opt = munchify(config)
        return self.opt
