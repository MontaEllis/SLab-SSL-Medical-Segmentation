import math

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.utils import build_dataset


def _get_batch_tensor(batch, key):
    value = batch[key]
    if isinstance(value, dict):
        return value["data"]
    return value


def _main_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _binary_metric(prediction, label):
    prediction[prediction > 0] = 1
    label[label > 0] = 1
    if prediction.sum() > 0 and label.sum() > 0:
        return metric.binary.dc(prediction, label), metric.binary.hd95(prediction, label)
    return 0.0, 0.0


def _test_single_volume(image, label, model, classes, patch_size):
    device = next(model.parameters()).device
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for index in range(image.shape[0]):
        slice_image = image[index, :, :]
        height, width = slice_image.shape
        resized = zoom(slice_image, (patch_size[0] / height, patch_size[1] / width), order=0)
        input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = _main_output(model(input_tensor))
            output = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            output = output.cpu().detach().numpy()

        prediction[index] = zoom(output, (height / patch_size[0], width / patch_size[1]), order=0)

    metric_list = []
    for class_index in range(1, classes):
        metric_list.append(_binary_metric(prediction == class_index, label == class_index))
    return metric_list


def _test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes):
    device = next(model.parameters()).device
    width, height, depth = image.shape
    add_pad = False

    width_pad = max(patch_size[0] - width, 0)
    height_pad = max(patch_size[1] - height, 0)
    depth_pad = max(patch_size[2] - depth, 0)
    if width_pad > 0 or height_pad > 0 or depth_pad > 0:
        add_pad = True

    left_w, right_w = width_pad // 2, width_pad - width_pad // 2
    left_h, right_h = height_pad // 2, height_pad - height_pad // 2
    left_d, right_d = depth_pad // 2, depth_pad - depth_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(left_w, right_w), (left_h, right_h), (left_d, right_d)],
            mode="constant",
            constant_values=0,
        )

    padded_w, padded_h, padded_d = image.shape
    sx = math.ceil((padded_w - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((padded_h - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((padded_d - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    count_map = np.zeros(image.shape, dtype=np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, padded_w - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, padded_h - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, padded_d - patch_size[2])
                patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0).astype(np.float32)
                patch = torch.from_numpy(patch).to(device)

                with torch.no_grad():
                    output = _main_output(model(patch))
                    output = torch.softmax(output, dim=1)
                output = output.cpu().numpy()[0]

                score_map[:, xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] += output
                count_map[xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] += 1

    score_map = score_map / np.expand_dims(count_map, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[left_w : left_w + width, left_h : left_h + height, left_d : left_d + depth]
    return label_map


def evaluate_model(args, model):
    model.eval()
    valset = build_dataset(args, mode="val")
    valloader = DataLoader(valset.queue_dataset, batch_size=1, shuffle=False, num_workers=1)

    if args.dataset.dataset_name == "acdc":
        metric_list = 0.0
        for sampled_batch in valloader:
            image = _get_batch_tensor(sampled_batch, "source")
            label = _get_batch_tensor(sampled_batch, "label")
            metric_list += np.array(
                _test_single_volume(
                    image,
                    label,
                    model,
                    classes=args.model.out_channels,
                    patch_size=args.dataset.patch_size,
                )
            )
        metric_list = metric_list / len(valset.queue_dataset)
        return {
            "mean_dice": float(np.mean(metric_list, axis=0)[0]),
            "mean_hd95": float(np.mean(metric_list, axis=0)[1]),
            "per_class": metric_list.tolist(),
        }

    total_metric = np.zeros((args.model.out_channels - 1, 2))
    for sampled_batch in tqdm(valloader, desc="Validation"):
        image = _get_batch_tensor(sampled_batch, "source")
        label = _get_batch_tensor(sampled_batch, "label")

        if image.dim() == 5:
            image = image[0, 0]
        else:
            image = image[0]
        if label.dim() == 5:
            label = label[0, 0]
        else:
            label = label[0]

        prediction = _test_single_case(
            model,
            image.cpu().numpy(),
            stride_xy=args.dataset.stride_xy,
            stride_z=args.dataset.stride_z,
            patch_size=tuple(args.dataset.patch_size),
            num_classes=args.model.out_channels,
        )
        label = label.cpu().numpy()
        for class_index in range(1, args.model.out_channels):
            total_metric[class_index - 1, :] += np.array(
                _binary_metric(prediction == class_index, label == class_index)
            )

    avg_metric = total_metric / len(valset.queue_dataset)
    return {
        "mean_dice": float(avg_metric[:, 0].mean()),
        "mean_hd95": float(avg_metric[:, 1].mean()),
        "per_class": avg_metric.tolist(),
    }
