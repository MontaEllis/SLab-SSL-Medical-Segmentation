import itertools
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


def _pack_sample(image, label):
    return {
        "source": {"data": image},
        "label": {"data": label},
    }


class BraTS2019VolumeDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode
        self.base_dir = Path(args.dataset_path)
        self.image_list = self._load_case_list()
        self.transforms = self._build_transforms()

    def _load_case_list(self):
        list_path = self.base_dir / ("train.txt" if self.mode == "train" else "val.txt")
        with open(list_path, "r") as handle:
            image_list = [item.strip().split(",")[0] for item in handle.readlines()]
        print("total {} samples".format(len(image_list)))
        return image_list

    def _build_transforms(self):
        if self.mode == "train":
            return transforms.Compose(
                [
                    RandomRotFlip(),
                    RandomCrop(self.args.patch_size),
                    ToTensor(),
                ]
            )
        return ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        with h5py.File(self.base_dir / "data" / f"{image_name}.h5", "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:]

        sample = self.transforms({"image": image, "label": label.astype(np.uint8)})
        sample["idx"] = idx
        return sample


class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if (
            label.shape[0] <= self.output_size[0]
            or label.shape[1] <= self.output_size[1]
            or label.shape[2] <= self.output_size[2]
        ):
            pad_w = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            pad_h = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pad_d = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)], mode="constant", constant_values=0)
            label = np.pad(label, [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)], mode="constant", constant_values=0)

        width, height, depth = image.shape
        start_w = np.random.randint(0, width - self.output_size[0])
        start_h = np.random.randint(0, height - self.output_size[1])
        start_d = np.random.randint(0, depth - self.output_size[2])

        return {
            "image": image[
                start_w : start_w + self.output_size[0],
                start_h : start_h + self.output_size[1],
                start_d : start_d + self.output_size[2],
            ],
            "label": label[
                start_w : start_w + self.output_size[0],
                start_h : start_h + self.output_size[1],
                start_d : start_d + self.output_size[2],
            ],
        }


class RandomRotFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {"image": image, "label": label}


class ToTensor:
    def __call__(self, sample):
        image = sample["image"]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return _pack_sample(
            torch.from_numpy(image),
            torch.from_numpy(sample["label"]).long(),
        )


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for primary_batch, secondary_batch in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


class BraTS2019Dataset:
    def __init__(self, args, mode="train", transform_name=None):
        self.args = args
        self.mode = mode
        self.transform_name = transform_name
        self.batch_sampler_cls = TwoStreamBatchSampler
        self.dataset = BraTS2019VolumeDataset(args, mode=mode)
        self.queue_dataset = self.dataset

    def get_labeled_and_unlabeled_indices(self):
        total_cases = len(self.queue_dataset)
        labeled_indices = list(range(0, self.args.labeled_num))
        unlabeled_indices = list(range(self.args.labeled_num, total_cases))
        return labeled_indices, unlabeled_indices
