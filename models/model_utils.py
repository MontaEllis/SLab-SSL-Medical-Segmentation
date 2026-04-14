import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock

def _canonical_model_name(model_name):
    alias_map = {
        "unet_3d": "UNet",
        "vnet_3d": "VNet",
        "attention_unet_3d": "AttentionUnet",
        "nnunet_3d": "nnUnet",
        "swinunet": "swinunet_2d",
    }
    return alias_map.get(model_name, model_name)


def get_model(args, model_name=None):
    model_name = _canonical_model_name(model_name or args.model.model_name)

    if args.model.dimension == "2d":
        if model_name == "unet_2d":
            from .two_d.Unet.unet import Unet

            return Unet(in_channels=args.model.in_channels, classes=args.model.out_channels)

        if model_name == "unetpp_2d":
            from .two_d.UNetPP.unetpp import ResNet34UnetPlus

            return ResNet34UnetPlus(num_channels=args.model.in_channels, num_class=args.model.out_channels)

        if model_name == "deeplabv3_resnet50":
            from .two_d.deeplabv3.modeling import deeplabv3_resnet50
            import torch.nn as local_nn

            model = deeplabv3_resnet50(
                num_classes=args.model.out_channels,
                output_stride=8,
                pretrained_backbone=False,
            )
            if args.model.in_channels != 3:
                model.backbone["conv1"] = local_nn.Conv2d(
                    args.model.in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            return model

        if model_name == "deeplabv3_resnet101":
            from .two_d.deeplabv3.modeling import deeplabv3_resnet101
            import torch.nn as local_nn

            model = deeplabv3_resnet101(
                num_classes=args.model.out_channels,
                output_stride=8,
                pretrained_backbone=False,
            )
            if args.model.in_channels != 3:
                model.backbone["conv1"] = local_nn.Conv2d(
                    args.model.in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            return model

        if model_name == "fcn_2d":
            from .two_d.FCN.fcn import FCN32s

            return FCN32s(in_class=args.model.in_channels, n_class=args.model.out_channels)

        if model_name == "segnet":
            from .two_d.SegNet.segnet import SegNet

            return SegNet(input_nbr=args.model.in_channels, label_nbr=args.model.out_channels)

        if model_name == "pspnet":
            from .two_d.PSPNet.pspnet import PSPNet

            return PSPNet(in_class=args.model.in_channels, n_classes=args.model.out_channels)

        if model_name == "highresnet":
            from .two_d.highresnet.highresnet import HighResNet

            return HighResNet(
                in_channels=args.model.in_channels,
                out_channels=args.model.out_channels,
                dimensions=2,
            )

        if model_name == "miniseg":
            from .two_d.MiniSeg.miniseg import MiniSeg

            return MiniSeg(in_input=args.model.in_channels, classes=args.model.out_channels)

        if model_name == "attention_unet_2d":
            from .two_d.attention_unet.attention_unet import AttentionUNet

            return AttentionUNet(
                in_ch=args.model.in_channels,
                num_classes=args.model.out_channels,
                base_ch=args.model.base_chan,
                block=args.model.block,
            )

        if model_name == "danet_2d":
            from .two_d.DANet.dual_attention_unet import DAUNet

            return DAUNet(
                in_ch=args.model.in_channels,
                num_classes=args.model.out_channels,
                base_ch=args.model.base_chan,
                block=args.model.block,
            )

        if model_name == "transunet_2d":
            from .two_d.TransUnet.transunet import VisionTransformer, get_r50_b16_config

            config_vit = get_r50_b16_config()
            config_vit.n_classes = args.model.out_channels
            config_vit.n_skip = 3

            if hasattr(args.dataset, "img_size"):
                img_size = args.dataset.img_size
            elif hasattr(args.dataset, "image_size"):
                img_size = args.dataset.image_size
            elif hasattr(args.dataset, "patch_size"):
                patch_size_val = (
                    args.dataset.patch_size[0]
                    if isinstance(args.dataset.patch_size, (list, tuple))
                    else args.dataset.patch_size
                )
                img_size = patch_size_val if patch_size_val >= 256 else 512
            else:
                img_size = 512

            return VisionTransformer(config_vit, img_size=img_size, num_classes=args.model.out_channels)

        if model_name == "swinunet_2d":
            from .two_d.SwinUnet.swin_unet import SwinTransformerSys

            if hasattr(args.dataset, "img_size"):
                img_size = args.dataset.img_size
            elif hasattr(args.dataset, "image_size"):
                img_size = args.dataset.image_size
            elif hasattr(args.dataset, "patch_size"):
                patch_size = args.dataset.patch_size
                img_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
            else:
                img_size = 224

            return SwinTransformerSys(
                img_size=img_size,
                patch_size=args.model.patch_size,
                in_chans=args.model.in_channels,
                num_classes=args.model.out_channels,
                embed_dim=args.model.embed_dim,
                depths=args.model.depths,
                depths_decoder=args.model.depths_decoder,
                num_heads=args.model.num_heads,
                window_size=args.model.window_size,
                mlp_ratio=args.model.mlp_ratio,
                qkv_bias=args.model.qkv_bias,
                qk_scale=args.model.qk_scale,
                drop_rate=args.model.drop_rate,
                attn_drop_rate=args.model.attn_drop_rate,
                drop_path_rate=args.model.drop_path_rate,
                norm_layer=args.model.norm_layer,
                patch_norm=args.model.patch_norm,
                use_checkpoint=args.model.use_checkpoint,
                frozen_stages=args.model.frozen_stages,
                final_upsample=args.model.final_upsample,
            )

        if model_name == "medformer_2d":
            from .two_d.medformer.medformer import MedFormer

            return MedFormer(
                in_chan=args.model.in_channels,
                num_classes=args.model.out_channels,
                base_chan=args.model.base_chan,
                map_size=args.model.map_size,
                conv_block=args.model.conv_block,
                conv_num=args.model.conv_num,
                trans_num=args.model.trans_num,
                num_heads=args.model.num_heads,
                fusion_depth=args.model.fusion_depth,
                fusion_dim=args.model.fusion_dim,
                fusion_heads=args.model.fusion_heads,
                expansion=args.model.expansion,
                attn_drop=args.model.attn_drop,
                proj_drop=args.model.proj_drop,
                proj_type=args.model.proj_type,
            )

        raise ValueError(f"2D model {model_name} not supported")

    if args.model.dimension == "3d":
        if model_name == "Unetr":
            from .three_d.unetr.unetr import UNETR

            return UNETR(
                args.model.in_channels,
                args.model.out_channels,
                args.dataset.patch_size,
                feature_size=args.model.feature_size,
                hidden_size=args.model.hidden_size,
                mlp_dim=args.model.mlp_dim,
                num_heads=args.model.num_heads,
                proj_type=args.model.proj_type,
                pos_embed_type=args.model.pos_embed_type,
                norm_name=args.model.norm_name,
                res_block=args.model.res_block,
            )

        if model_name == "UNet":
            from .three_d.UNet.unet import UNet

            return UNet(
                args.model.in_channels,
                num_classes=args.model.out_channels,
                base_ch=args.model.base_chan,
                scale=args.model.down_scale,
                norm=args.model.norm,
                kernel_size=args.model.kernel_size,
                block=args.model.block,
            )

        if model_name == "UNETR_PP":
            from .three_d.unetr_pp.synapse.unetr_pp_synapse import UNETR_PP

            return UNETR_PP(
                in_channels=args.model.in_channels,
                out_channels=args.model.out_channels,
                img_size=args.dataset.patch_size,
                feature_size=args.model.feature_size,
                hidden_size=args.model.hidden_size,
                dims=args.model.dims,
                num_heads=args.model.num_heads,
                pos_embed=args.model.pos_embed,
                norm_name=args.model.norm_name,
                dropout_rate=args.model.dropout_rate,
                do_ds=args.model.do_ds,
            )

        if model_name == "segformer3d":
            from .three_d.segformer.segformer import build_segformer3d_model

            return build_segformer3d_model(
                {
                    "in_channels": args.model.in_channels,
                    "sr_ratios": args.model.sr_ratios,
                    "embed_dims": args.model.embed_dims,
                    "patch_kernel_size": args.model.patch_kernel_size,
                    "patch_stride": args.model.patch_stride,
                    "patch_padding": args.model.patch_padding,
                    "mlp_ratios": args.model.mlp_ratios,
                    "num_heads": args.model.num_heads,
                    "depths": args.model.depths,
                    "num_classes": args.model.out_channels,
                    "decoder_head_embedding_dim": args.model.decoder_head_embedding_dim
                    * (int(128 / args.dataset.patch_size[0]) ** 3),
                    "decoder_dropout": args.model.decoder_dropout,
                }
            )

        if model_name == "segmamba":
            from .three_d.segmamba.segmamba import SegMamba

            return SegMamba(
                in_chans=args.model.in_channels,
                out_chans=args.model.out_channels,
                depths=args.model.depths,
                feat_size=args.model.feat_size,
            )

        if model_name == "slim_unetr":
            from .three_d.slim_unetr.SlimUNETR import SlimUNETR

            return SlimUNETR(
                in_channels=args.model.in_channels,
                out_channels=args.model.out_channels,
                embed_dim=args.model.embed_dim,
                embedding_dim=args.model.embedding_dim,
                channels=args.model.channels,
                blocks=args.model.blocks,
                heads=args.model.heads,
                r=args.model.r,
                dropout=args.model.dropout,
            )

        if model_name == "VTUNET":
            from .three_d.VTUNET.vtunet import SwinTransformerSys3D

            model = SwinTransformerSys3D(
                img_size=args.dataset.patch_size,
                patch_size=args.model.patch_size,
                in_chans=args.model.in_channels,
                num_classes=args.model.out_channels if not args.model.pretrained else args.model.pretrain_classes,
                embed_dim=args.model.embed_dim,
                depths=args.model.depths,
                depths_decoder=args.model.depths_decoder,
                num_heads=args.model.num_heads,
                window_size=args.model.window_size,
                mlp_ratio=args.model.mlp_ratio,
                qkv_bias=args.model.qkv_bias,
                qk_scale=args.model.qk_scale,
                drop_rate=args.model.drop_rate,
                attn_drop_rate=args.model.attn_drop_rate,
                drop_path_rate=args.model.drop_path_rate,
                norm_layer=args.model.norm_layer,
                patch_norm=args.model.patch_norm,
                use_checkpoint=args.model.use_checkpoint,
                frozen_stages=args.model.frozen_stages,
                final_upsample=args.model.final_upsample,
            )
            if args.model.pretrained:
                model.load_state_dict(torch.load(args.model.pretrained_weights))
            return model

        if model_name == "AttentionUnet":
            from .three_d.Attention_Unet.attention_unet import AttentionUNet

            return AttentionUNet(
                args.model.in_channels,
                num_classes=args.model.out_channels,
                base_ch=args.model.base_chan,
                scale=args.model.down_scale,
                norm=args.model.norm,
                kernel_size=args.model.kernel_size,
                block=args.model.block,
            )

        if model_name == "SwinUNETR":
            from .three_d.swin_unetr.swin_unetr import SwinUNETR

            if args.model.pretrained_weights:
                model = SwinUNETR(
                    img_size=args.dataset.patch_size[0],
                    in_channels=args.model.in_channels,
                    out_channels=args.model.pretrain_classes,
                    feature_size=args.model.feature_size,
                    use_checkpoint=False,
                )
                model.load_state_dict(torch.load(args.model.pretrained_weights))
                model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=args.model.out_channels)
                return model

            return SwinUNETR(
                img_size=args.dataset.patch_size,
                in_channels=args.model.in_channels,
                out_channels=args.model.out_channels,
                feature_size=args.model.feature_size,
                use_checkpoint=False,
            )

        if model_name == "UNetPP":
            from .three_d.UNetPP.unetpp import UNetPlusPlus

            return UNetPlusPlus(
                args.model.in_channels,
                num_classes=args.model.out_channels,
                base_ch=args.model.base_chan,
                block=args.model.block,
            )

        if model_name == "3DUXNET":
            from .three_d.UXNet_3D.network_backbone import UXNET

            if args.model.pretrained_weights:
                model = UXNET(
                    in_chans=args.model.in_channels,
                    out_chans=args.model.pretrain_classes,
                    depths=args.model.depths,
                    feat_size=args.model.feat_size,
                    drop_path_rate=args.model.drop_path_rate,
                    layer_scale_init_value=args.model.layer_scale_init_value,
                    spatial_dims=args.model.spatial_dims,
                )
                model.load_state_dict(torch.load(args.model.pretrained_weights))
                model.out = UnetOutBlock(
                    spatial_dims=args.model.spatial_dims,
                    in_channels=args.model.in_channels,
                    out_channels=args.model.out_channels,
                )
                return model

            return UXNET(
                in_chans=args.model.in_channels,
                out_chans=args.model.out_channels,
                depths=args.model.depths,
                feat_size=args.model.feat_size,
                drop_path_rate=args.model.drop_path_rate,
                layer_scale_init_value=args.model.layer_scale_init_value,
                spatial_dims=args.model.spatial_dims,
            )

        if model_name == "nnFormer":
            from .three_d.nnFormer.nnFormer_seg import nnFormer

            if args.model.pretrained_weights:
                from .three_d.nnFormer.nnFormer_seg import final_patch_expanding

                final_layer = []
                model = nnFormer(input_channels=args.model.in_channels, num_classes=args.model.pretrain_classes)
                model.load_state_dict(torch.load(args.model.pretrained_weights))
                final_layer.append(final_patch_expanding(192, args.model.out_channels, patch_size=[2, 4, 4]))
                model.final = nn.ModuleList(final_layer)
                return model

            return nnFormer(
                input_channels=args.model.in_channels,
                num_classes=args.model.out_channels,
                crop_size=args.dataset.patch_size,
            )

        if model_name == "EfficientMedNeXt_T":
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext

            return create_efficient_mednext(
                args.model.in_channels,
                args.model.out_channels,
                "T",
                n_channels=args.model.feature_size,
                kernel_sizes=args.model.kernel_sizes,
                strides=args.model.strides,
                uniform_dec_channels=args.model.n_decoder_channels,
                deep_supervision=args.model.deep_supervision,
            )

        if model_name == "EfficientMedNeXt_S":
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext

            return create_efficient_mednext(
                args.model.in_channels,
                args.model.out_channels,
                "S",
                n_channels=args.model.feature_size,
                kernel_sizes=args.model.kernel_sizes,
                strides=args.model.strides,
                uniform_dec_channels=args.model.n_decoder_channels,
                deep_supervision=args.model.deep_supervision,
            )

        if model_name == "EfficientMedNeXt_M":
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext

            return create_efficient_mednext(
                args.model.in_channels,
                args.model.out_channels,
                "M",
                n_channels=args.model.feature_size,
                kernel_sizes=args.model.kernel_sizes,
                strides=args.model.strides,
                uniform_dec_channels=args.model.n_decoder_channels,
                deep_supervision=args.model.deep_supervision,
            )

        if model_name == "EfficientMedNeXt_L":
            from .three_d.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext

            return create_efficient_mednext(
                args.model.in_channels,
                args.model.out_channels,
                "L",
                n_channels=args.model.feature_size,
                kernel_sizes=args.model.kernel_sizes,
                strides=args.model.strides,
                uniform_dec_channels=args.model.n_decoder_channels,
                deep_supervision=args.model.deep_supervision,
            )

        if model_name == "nnUnet":
            from .three_d.nnunet.network_architecture.generic_UNet import Generic_UNet

            return Generic_UNet(
                input_channels=args.model.in_channels,
                base_num_features=args.model.base_num_features,
                num_classes=args.model.out_channels,
                num_pool=args.model.num_pool,
                num_conv_per_stage=args.model.num_conv_per_stage,
                conv_op=nn.Conv3d,
                norm_op=nn.BatchNorm3d,
                dropout_op=nn.Dropout3d,
                max_num_features=args.model.max_num_features,
                deep_supervision=args.model.deep_supervision,
            )

        if model_name == "VNet":
            from .three_d.VNet.vnet import VNet

            return VNet(
                args.model.in_channels,
                args.model.out_channels,
                scale=args.model.downsample_scale,
                baseChans=args.model.base_chan,
            )

        if model_name == "MedFormer":
            from .three_d.MedFormer.medformer import MedFormer

            return MedFormer(
                args.model.in_channels,
                args.model.out_channels,
                args.model.base_chan,
                map_size=args.model.map_size,
                conv_block=args.model.conv_block,
                conv_num=args.model.conv_num,
                trans_num=args.model.trans_num,
                num_heads=args.model.num_heads,
                fusion_depth=args.model.fusion_depth,
                fusion_dim=args.model.fusion_dim,
                fusion_heads=args.model.fusion_heads,
                expansion=args.model.expansion,
                attn_drop=args.model.attn_drop,
                proj_drop=args.model.proj_drop,
                proj_type=args.model.proj_type,
                norm=args.model.norm,
                act=args.model.act,
                kernel_size=args.model.kernel_size,
                scale=args.model.down_scale,
            )

        if model_name == "TransBTS":
            from .three_d.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

            model = TransBTS(
                num_channels=args.model.in_channels,
                num_classes=args.model.pretrain_classes if args.model.pretrained_weights else args.model.out_channels,
                img_dim=args.dataset.patch_size[0],
                patch_dim=args.model.patch_dim,
                embedding_dim=args.model.embedding_dim,
                num_heads=args.model.num_heads,
                num_layers=args.model.num_layers,
                hidden_dim=args.model.hidden_dim,
                dropout_rate=args.model.dropout_rate,
                attn_dropout_rate=args.model.attn_dropout_rate,
                _conv_repr=args.model._conv_repr,
                _pe_type=args.model._pe_type,
            )[1]
            if args.model.pretrained_weights:
                model.load_state_dict(torch.load(args.model.pretrained_weights))
                model.endconv = nn.Conv3d(512 // 32, args.model.out_channels, kernel_size=1)
            return model

        raise ValueError(f"3D model {model_name} not supported")

    raise ValueError("Invalid dimension, should be '2d' or '3d'")


def get_discriminator(args):
    from .discriminator import FC3DDiscriminator, FCDiscriminator

    if args.model.dimension == "2d":
        return FCDiscriminator(num_classes=args.model.out_channels)

    return FC3DDiscriminator(num_classes=args.model.out_channels)


def initialize_weights(model, init_type=None):
    if init_type is None:
        return model

    conv_types = (nn.Conv2d, nn.Conv3d)
    norm_types = (nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, conv_types):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight)
        elif isinstance(module, norm_types):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
    return model
