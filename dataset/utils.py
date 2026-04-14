def build_dataset(args, mode, transform_name=None):
    if args.dataset.dataset_name == "acdc":
        from .acdc import ACDC

        return ACDC(args.dataset, mode=mode, transform_name=transform_name)

    if args.dataset.dataset_name == "brats2019":
        from .brats2019 import BraTS2019Dataset

        return BraTS2019Dataset(args.dataset, mode=mode, transform_name=transform_name)

    raise ValueError(f"Unsupported dataset: {args.dataset.dataset_name}")
