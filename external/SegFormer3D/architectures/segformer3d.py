
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODEL BUILDER
# ============================================================

def build_segformer3d_model(config=None):
    if config is None:
        raise ValueError("A model configuration must be supplied.")

    params = config["model_parameters"]

    return SegFormer3D(
        in_channels=params["in_channels"],
        sr_ratios=params["sr_ratios"],
        embed_dims=params["embed_dims"],
        patch_kernel_size=params["patch_kernel_size"],
        patch_stride=params["patch_stride"],
        patch_padding=params["patch_padding"],
        mlp_ratios=params["mlp_ratios"],
        num_heads=params["num_heads"],
        depths=params["depths"],
        decoder_head_embedding_dim=params["decoder_head_embedding_dim"],
        num_classes=params["num_classes"],
        decoder_dropout=params["decoder_dropout"],
    )


# ============================================================
# SEGFORMER3D
# ============================================================

class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        sr_ratios=(4, 2, 1, 1),
        embed_dims=(16, 32, 64, 128),
        patch_kernel_size=(7, 3, 3, 3),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 1, 1, 1),
        mlp_ratios=(4, 4, 4, 4),
        num_heads=(1, 2, 4, 8),
        depths=(2, 2, 2, 2),
        decoder_head_embedding_dim=128,
        num_classes=2,
        decoder_dropout=0.0,
    ):
        super().__init__()

        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )

        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=list(embed_dims[::-1]),
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm3d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

        elif isinstance(module, nn.Conv3d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.kernel_size[2]
                * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.segformer_encoder(x)

        c1, c2, c3, c4 = features

        output = self.segformer_decoder(c1, c2, c3, c4)

        # Guarantee exact input spatial dimensions, including
        # dimensions that are not divisible by the decoder scale.
        if output.shape[2:] != input_size:
            output = F.interpolate(
                output,
                size=input_size,
                mode="trilinear",
                align_corners=False,
            )

        return output


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel,
        embed_dim,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()

        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_shape=False):
        x = self.patch_embeddings(x)

        spatial_shape = x.shape[2:]

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        if return_shape:
            return x, spatial_shape

        return x


# ============================================================
# SELF ATTENTION
# ============================================================

class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        sr_ratio,
        qkv_bias=True,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.attn_dropout = attn_dropout

        self.query = nn.Linear(
            embed_dim,
            embed_dim,
            bias=qkv_bias,
        )

        self.key_value = nn.Linear(
            embed_dim,
            2 * embed_dim,
            bias=qkv_bias,
        )

        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )

            self.sr_norm = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, spatial_shape=None):
        batch_size, num_tokens, channels = x.shape

        q = self.query(x).reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        q = q.permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape is required for spatial-reduction attention."
                )

            depth, height, width = spatial_shape

            if (
                depth < self.sr_ratio
                or height < self.sr_ratio
                or width < self.sr_ratio
            ):
                raise ValueError(
                    f"Feature dimensions {spatial_shape} are too small "
                    f"for reduction ratio {self.sr_ratio}."
                )

            x_reduced = x.transpose(1, 2).reshape(
                batch_size,
                channels,
                depth,
                height,
                width,
            ).contiguous()

            x_reduced = self.sr(x_reduced)

            x_reduced = x_reduced.flatten(2).transpose(1, 2).contiguous()

            x_reduced = self.sr_norm(x_reduced)

        else:
            x_reduced = x

        kv = self.key_value(x_reduced).reshape(
            batch_size,
            -1,
            2,
            self.num_heads,
            self.head_dim,
        )

        kv = kv.permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attention = (q @ k.transpose(-2, -1)) * self.scale
            attention = attention.softmax(dim=-1)

            if self.training and self.attn_dropout > 0:
                attention = F.dropout(
                    attention,
                    p=self.attn_dropout,
                    training=True,
                )

            output = attention @ v

        output = output.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels,
        ).contiguous()

        output = self.proj(output)
        output = self.proj_dropout(output)

        return output


# ============================================================
# DEPTHWISE 3D CONVOLUTION
# ============================================================

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=True,
        )

        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, spatial_shape=None):
        if spatial_shape is None:
            raise ValueError(
                "spatial_shape is required by the 3D depthwise convolution."
            )

        batch_size, num_tokens, channels = x.shape

        depth, height, width = spatial_shape

        if depth * height * width != num_tokens:
            raise ValueError(
                "Token count does not match the supplied spatial dimensions."
            )

        x = x.transpose(1, 2).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        ).contiguous()

        x = self.dwconv(x)
        x = self.bn(x)

        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


# ============================================================
# MIXED FEED-FORWARD NETWORK
# ============================================================

class _MLP(nn.Module):
    def __init__(
        self,
        in_feature,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()

        out_feature = mlp_ratio * in_feature

        self.fc1 = nn.Linear(in_feature, out_feature)

        self.dwconv = DWConv(dim=out_feature)

        self.fc2 = nn.Linear(out_feature, in_feature)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_shape=None):
        x = self.fc1(x)

        x = self.dwconv(x, spatial_shape)

        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        num_heads,
        sr_ratio,
        qkv_bias=True,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = _MLP(
            in_feature=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=proj_dropout,
        )

    def forward(self, x, spatial_shape=None):
        x = x + self.attention(
            self.norm1(x),
            spatial_shape,
        )

        x = x + self.mlp(
            self.norm2(x),
            spatial_shape,
        )

        return x


# ============================================================
# MIX VISION TRANSFORMER ENCODER
# ============================================================

class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        sr_ratios=(4, 2, 1, 1),
        embed_dims=(16, 32, 64, 128),
        patch_kernel_size=(7, 3, 3, 3),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 1, 1, 1),
        mlp_ratios=(4, 4, 4, 4),
        num_heads=(1, 2, 4, 8),
        depths=(2, 2, 2, 2),
    ):
        super().__init__()

        self.embed_1 = PatchEmbedding(
            in_channels,
            embed_dims[0],
            patch_kernel_size[0],
            patch_stride[0],
            patch_padding[0],
        )

        self.embed_2 = PatchEmbedding(
            embed_dims[0],
            embed_dims[1],
            patch_kernel_size[1],
            patch_stride[1],
            patch_padding[1],
        )

        self.embed_3 = PatchEmbedding(
            embed_dims[1],
            embed_dims[2],
            patch_kernel_size[2],
            patch_stride[2],
            patch_padding[2],
        )

        self.embed_4 = PatchEmbedding(
            embed_dims[2],
            embed_dims[3],
            patch_kernel_size[3],
            patch_stride[3],
            patch_padding[3],
        )

        self.tf_block1 = self._make_stage(
            embed_dims[0], mlp_ratios[0],
            num_heads[0], sr_ratios[0], depths[0],
        )

        self.tf_block2 = self._make_stage(
            embed_dims[1], mlp_ratios[1],
            num_heads[1], sr_ratios[1], depths[1],
        )

        self.tf_block3 = self._make_stage(
            embed_dims[2], mlp_ratios[2],
            num_heads[2], sr_ratios[2], depths[2],
        )

        self.tf_block4 = self._make_stage(
            embed_dims[3], mlp_ratios[3],
            num_heads[3], sr_ratios[3], depths[3],
        )

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    @staticmethod
    def _make_stage(
        embed_dim,
        mlp_ratio,
        num_heads,
        sr_ratio,
        depth,
    ):
        return nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qkv_bias=True,
            )
            for _ in range(depth)
        ])

    @staticmethod
    def _tokens_to_volume(x, spatial_shape):
        batch_size, _, channels = x.shape

        depth, height, width = spatial_shape

        return x.transpose(1, 2).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        ).contiguous()

    def _run_stage(self, x, embedding, blocks, norm):
        x, spatial_shape = embedding(
            x,
            return_shape=True,
        )

        for block in blocks:
            x = block(x, spatial_shape)

        x = norm(x)

        return self._tokens_to_volume(x, spatial_shape)

    def forward(self, x):
        outputs = []

        x = self._run_stage(
            x,
            self.embed_1,
            self.tf_block1,
            self.norm1,
        )
        outputs.append(x)

        x = self._run_stage(
            x,
            self.embed_2,
            self.tf_block2,
            self.norm2,
        )
        outputs.append(x)

        x = self._run_stage(
            x,
            self.embed_3,
            self.tf_block3,
            self.norm3,
        )
        outputs.append(x)

        x = self._run_stage(
            x,
            self.embed_4,
            self.tf_block4,
            self.norm4,
        )
        outputs.append(x)

        return outputs


# ============================================================
# DECODER MLP
# ============================================================

class MLP_(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()

        x = self.proj(x)
        x = self.norm(x)

        return x


# ============================================================
# SEGFORMER DECODER HEAD
# ============================================================

class SegFormerDecoderHead(nn.Module):
    def __init__(
        self,
        input_feature_dims=(128, 64, 32, 16),
        decoder_head_embedding_dim=128,
        num_classes=2,
        dropout=0.0,
    ):
        super().__init__()

        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )

        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )

        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )

        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                4 * decoder_head_embedding_dim,
                decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim,
            num_classes,
            kernel_size=1,
        )

    @staticmethod
    def _project_and_resize(mlp, feature, target_size):
        batch_size = feature.shape[0]

        projected = mlp(feature)

        projected = projected.transpose(1, 2).reshape(
            batch_size,
            -1,
            feature.shape[2],
            feature.shape[3],
            feature.shape[4],
        ).contiguous()

        if projected.shape[2:] != target_size:
            projected = F.interpolate(
                projected,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        return projected

    def forward(self, c1, c2, c3, c4):
        target_size = c1.shape[2:]

        _c4 = self._project_and_resize(
            self.linear_c4,
            c4,
            target_size,
        )

        _c3 = self._project_and_resize(
            self.linear_c3,
            c3,
            target_size,
        )

        _c2 = self._project_and_resize(
            self.linear_c2,
            c2,
            target_size,
        )

        _c1 = self._project_and_resize(
            self.linear_c1,
            c1,
            target_size,
        )

        x = torch.cat(
            [_c4, _c3, _c2, _c1],
            dim=1,
        )

        x = self.linear_fuse(x)

        x = self.dropout(x)

        x = self.linear_pred(x)

        x = F.interpolate(
            x,
            scale_factor=4,
            mode="trilinear",
            align_corners=False,
        )

        return x


# ============================================================
# QUICK MODEL TEST
# ============================================================

if __name__ == "__main__":
    torch.set_num_threads(6)

    config = {
        "model_name": "segformer3d",
        "model_parameters": {
            "in_channels": 1,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [16, 32, 64, 128],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 4, 8],
            "depths": [2, 2, 2, 2],
            "decoder_head_embedding_dim": 128,
            "num_classes": 2,
            "decoder_dropout": 0.0,
        },
    }

    model = build_segformer3d_model(config)
    model.eval()

    sample = torch.randn(1, 1, 64, 64, 32)

    with torch.no_grad():
        prediction = model(sample)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(prediction.shape))
    print("Finite output:", torch.isfinite(prediction).all().item())
    print("Parameters:", f"{parameter_count:,}")

    assert prediction.shape == (1, 2, 64, 64, 32)
    assert torch.isfinite(prediction).all()

    print("SegFormer3D model test passed.")
