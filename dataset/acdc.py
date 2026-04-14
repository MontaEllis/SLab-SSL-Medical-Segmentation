import itertools
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


def _pack_supervised_sample(image, label):
    return {
        "source": {"data": image},
        "label": {"data": label},
    }


def _pack_augmented_sample(image, image_weak, image_strong, label):
    return {
        "source": {"data": image},
        "source_weak": {"data": image_weak},
        "source_strong": {"data": image_strong},
        "label": {"data": label},
        "label_aug": {"data": label},
    }


class ACDCDataset(Dataset):
    def __init__(self, args, mode="train", transform_name=None):
        self.args = args
        self.mode = mode
        self.transform_name = transform_name or "random"
        self.base_dir = Path(args.dataset_path)
        self.sample_list = self._load_sample_list()
        self.transforms = self._build_transforms()

    def _load_sample_list(self):
        if self.mode == "train":
            list_path = self.base_dir / "train_slices.list"
        else:
            list_path = self.base_dir / "val.list"

        with open(list_path, "r") as handle:
            sample_list = [item.strip() for item in handle.readlines()]
        print("total {} samples".format(len(sample_list)))
        return sample_list

    def _build_transforms(self):
        if self.mode != "train":
            return None
        if self.transform_name == "weak_strong":
            return WeakStrongAugment(self.args.patch_size)
        return RandomGenerator(self.args.patch_size)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        if self.mode == "train":
            file_path = self.base_dir / "data" / "slices" / f"{case_name}.h5"
        else:
            file_path = self.base_dir / "data" / f"{case_name}.h5"

        with h5py.File(file_path, "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:]

        if self.transforms is not None:
            sample = self.transforms({"image": image, "label": label})
        else:
            sample = _pack_supervised_sample(
                torch.from_numpy(image.astype(np.float32)),
                torch.from_numpy(label.astype(np.uint8)),
            )

        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        image = transforms.ToTensor()(image)
    jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return jitter(image)


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() < 0.5:
            image, label = random_rot_flip(image, label)
        else:
            image, label = random_rotate(image, label)

        height, width = image.shape
        image = zoom(image, (self.output_size[0] / height, self.output_size[1] / width), order=0)
        label = zoom(label, (self.output_size[0] / height, self.output_size[1] / width), order=0)

        return _pack_supervised_sample(
            torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(label.astype(np.uint8)),
        )


class WeakStrongAugment:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        image_weak, label = random_rot_flip(image, label)
        image_strong = color_jitter(image_weak).type(torch.FloatTensor)

        return _pack_augmented_sample(
            torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0),
            image_strong,
            torch.from_numpy(label.astype(np.uint8)),
        )

    def resize(self, image):
        height, width = image.shape
        return zoom(image, (self.output_size[0] / height, self.output_size[1] / width), order=0)


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


class ACDC:
    def __init__(self, args, mode="train", transform_name=None):
        self.args = args
        self.mode = mode
        self.transform_name = transform_name or "random"
        self.batch_sampler_cls = TwoStreamBatchSampler
        self.dataset = ACDCDataset(args, mode=mode, transform_name=self.transform_name)
        self.queue_dataset = self.dataset

    def get_labeled_and_unlabeled_indices(self):
        total_slices = len(self.queue_dataset)
        labeled_slices = self.patients_to_slices(self.args.dataset_path, self.args.labeled_num)
        labeled_indices = list(range(0, labeled_slices))
        unlabeled_indices = list(range(labeled_slices, total_slices))
        return labeled_indices, unlabeled_indices

    @staticmethod
    def patients_to_slices(dataset_path, patients_num):
        if "ACDC" in dataset_path or "acdc" in dataset_path:
            ref_dict = {
                "3": 68,
                "7": 136,
                "14": 256,
                "21": 396,
                "28": 512,
                "35": 664,
                "140": 1312,
            }
        elif "Prostate" in dataset_path or "prostate" in dataset_path:
            ref_dict = {
                "2": 27,
                "4": 53,
                "8": 120,
                "12": 179,
                "16": 256,
                "21": 312,
                "42": 623,
            }
        else:
            raise ValueError(f"Unsupported ACDC-style dataset path: {dataset_path}")

        key = str(patients_num)
        if key not in ref_dict:
            raise ValueError(f"Unsupported labeled_num={patients_num} for ACDC-style split")
        return ref_dict[key]
