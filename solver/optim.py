from torch import optim


def build_optimizer(args, parameters):
    if args.optimizer.optimizer_type == "sgd":
        return optim.SGD(
            parameters,
            lr=args.optimizer.lr,
            momentum=args.optimizer.momentum,
            weight_decay=args.optimizer.weight_decay,
        )

    if args.optimizer.optimizer_type == "adam":
        return optim.Adam(
            parameters,
            lr=args.optimizer.lr,
            betas=tuple(args.optimizer.betas),
            weight_decay=args.optimizer.weight_decay,
        )

    if args.optimizer.optimizer_type == "adamw":
        return optim.AdamW(
            parameters,
            lr=args.optimizer.lr,
            betas=tuple(args.optimizer.optim_betas),
            weight_decay=args.optimizer.optim_weight_decay,
            eps=1e-5,
        )

    raise ValueError(f"Unsupported optimizer: {args.optimizer.optimizer_type}")
