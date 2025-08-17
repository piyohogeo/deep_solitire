import json
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pt_utils import (
    filter_kwargs_for_function,
    nearest_embedding_indices_from_embedding_layer,
)

try:
    from flash_attn import flash_attn_func

    print("flash_attn is found")
except ImportError:
    flash_attn_func = None
    print("flash_attn is not found")


def get_rope_embedding(
    seq_len,
    head_dim,
    device,
    base: float = 10000.0,
    short_cycle_scale: float = 1.0,
):
    """オフセットを加えたRoPE embeddingを作成"""
    position = torch.arange(seq_len, device=device).float()
    dim = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (dim / head_dim))
    inv_freq = inv_freq * short_cycle_scale
    sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)

    sin_emb = torch.zeros(seq_len, head_dim, device=device)
    cos_emb = torch.zeros(seq_len, head_dim, device=device)
    sin_emb[:, ::2] = sinusoid_inp.sin()
    cos_emb[:, ::2] = sinusoid_inp.cos()
    sin_emb[:, 1::2] = sinusoid_inp.sin()
    cos_emb[:, 1::2] = sinusoid_inp.cos()

    return sin_emb, cos_emb


def apply_rope(x, sin_emb, cos_emb):
    # x: [batch, heads, seq_len, head_dim]
    # sin_emb, cos_emb: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)
    cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)

    x1, x2 = x[..., ::2], x[..., 1::2]  # [batch, heads, seq_len, head_dim//2]
    sin, cos = sin_emb[..., ::2], cos_emb[..., ::2]

    # 正しいRoPE適用方法
    x_rotated_even = x1 * cos - x2 * sin
    x_rotated_odd = x1 * sin + x2 * cos

    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
    return x_out  # [batch, heads, seq_len, head_dim]


class TransformerEncoderBlockRoPEPostLN(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        init_base=10000,
        layer_id=None,
        layer_scale_init=1e-2,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.base = torch.tensor(init_base)

        # LayerScaleパラメータの定義（Attention用とMLP用に別々に持つ）
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if (
                "weight" in name and param.dim() >= 2
            ):  # 2次元以上の重みのみ Xavier 初期化
                nn.init.xavier_uniform_(param)
            elif "bias" in name:  # バイアスはゼロ初期化
                nn.init.zeros_(param)

    def forward(self, x, is_return_attention=False):
        batch_size, seq_len, _ = x.shape

        # **Query, Key, Value を計算**
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        base = self.base

        # rope_sine [seq_len, head_dim]
        rope_sin, rope_cos = get_rope_embedding(
            seq_len,
            self.head_dim,
            q.device,
            base=base,
        )

        q = apply_rope(q, rope_sin, rope_cos)
        k = apply_rope(k, rope_sin, rope_cos)

        # スケーリング係数を掛ける
        q = q * self.scale
        # **Self-Attention**
        if flash_attn_func is not None:
            attn_output = flash_attn_func(
                q.permute(
                    0, 2, 1, 3
                ).bfloat16(),  # (batch, num_heads, seq_len, head_dim)
                k.permute(
                    0, 2, 1, 3
                ).bfloat16(),  # (batch, num_heads, seq_len, head_dim)
                v.permute(
                    0, 2, 1, 3
                ).bfloat16(),  # (batch, num_heads, seq_len, head_dim)
                dropout_p=0.1 if self.training else 0.0,
                causal=False,
            ).permute(
                0, 2, 1, 3
            )  # (batch, seq_len, num_heads, head_dim)
        else:
            attn_output = F.scaled_dot_product_attention(
                q.bfloat16(),
                k.bfloat16(),
                v.bfloat16(),
                dropout_p=0.1 if self.training else 0.0,
                is_causal=False,
            )

        # **元の形状に戻す**
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn_weights = None
        if is_return_attention:
            with torch.no_grad():
                attn_weights = (
                    torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k), dim=-1)
                    .detach()
                    .cpu()
                )
        attn_output_mean = attn_output.mean().detach()
        attn_output_std = attn_output.std().detach()
        attn_output = self.out_proj(attn_output)

        # LayerScaleを適用して残差に加える
        attn_output = self.layer_scale_1 * attn_output

        # **残差接続**
        x = self.norm1(x + attn_output)

        # **MLP**
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))

        # ログ取得（tensorboard用）
        self.logs = {
            "attn_output_mean": attn_output_mean,
            "attn_output_std": attn_output_std,
            "post_ln_mean": self.norm1(x).mean().detach(),
            "post_ln_std": self.norm1(x).std().detach(),
            "layer_scale_1_mean": self.layer_scale_1.mean().detach(),
            "layer_scale_2_mean": self.layer_scale_2.mean().detach(),
            "base": base.detach(),
        }

        return (x, attn_weights) if is_return_attention else x


def get_encoder_block_type(is_use_rope_post):
    if is_use_rope_post:
        return TransformerEncoderBlockRoPEPostLN
    else:
        raise NotImplementedError(
            "Only TransformerEncoderBlockRoPEPostLN is implemented for now."
        )


class SolitireMAEModel(nn.Module):
    def __init__(
        self,
        dim=128,
        latent_dim=32,
        num_layer=6,
        num_heads=2,
        input_proj_dim=None,
        is_use_rope_post=False,
        embeddings_len=128,
    ):
        super().__init__()
        self.dim = dim
        self.input_proj_dim = input_proj_dim
        encoder_type = get_encoder_block_type(
            is_use_rope_post,
        )
        params = {
            "dim": dim,
            "num_heads": num_heads,
        }
        self.enc_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.latent_proj = nn.Linear(dim, latent_dim)  # **最後に圧縮**
        if input_proj_dim is not None and input_proj_dim != dim:
            self.enc_input_proj = nn.Sequential(
                nn.Linear(input_proj_dim, dim),
                nn.LayerNorm(dim),
            )

        self.dec_proj = nn.Linear(latent_dim, dim)  # **デコード時に元の次元に戻す**
        if input_proj_dim is not None and input_proj_dim != dim:
            self.dec_input_proj = nn.Linear(dim, input_proj_dim)
        self.dec_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))  # マスクトークンのパラメータ
        if input_proj_dim is not None and input_proj_dim != dim:
            self.embeddings = nn.Embedding(embeddings_len, input_proj_dim)
        else:
            self.embeddings = nn.Embedding(embeddings_len, dim)

    def encode(
        self,
        x,
        mask=None,
        is_return_attention=False,
    ):
        attentions = []
        batch_size, seq_len, _ = x.shape
        if self.input_proj_dim is not None and self.input_proj_dim != self.dim:
            x = self.enc_input_proj(x)

        if mask is not None:
            # マスクトークンの挿入（256次元での挿入）
            mask_token_expanded = (
                self.mask_token.view(1, 1, -1)
                .expand(batch_size, seq_len, -1)
                .to(x.dtype)
            )
            x = torch.where(mask.unsqueeze(-1), mask_token_expanded, x)
        """入力トークン列から latent を計算"""
        for layer in self.enc_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Encoder

        x = self.norm(x)  # Transformer 出力を正規化
        latent = self.latent_proj(x)  # **最後に圧縮**
        return (latent, attentions) if is_return_attention else latent

    def decode(self, latent, is_return_attention=False):
        attentions = []
        """latent から元のトークン列を復元"""
        x = self.dec_proj(latent)  # **元の次元に復元**
        for layer in self.dec_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Decoder
        if self.input_proj_dim is not None and self.input_proj_dim != self.dim:
            x = self.dec_input_proj(x)
        return (x, attentions) if is_return_attention else x

    def decode_to_indices(self, latent, max_index=None, is_return_attention=False):
        """
        latent からトークンのインデックスを復元する。
        """
        if is_return_attention:
            x, attentions = self.decode(latent, is_return_attention=is_return_attention)
        else:
            x = self.decode(latent)
        indices = nearest_embedding_indices_from_embedding_layer(
            self.embeddings, x, max_emb_index=max_index
        )
        if is_return_attention:
            return indices, attentions
        else:
            return indices

    def forward(self, x, is_return_attention=False):
        """エンドツーエンドの AE"""
        if is_return_attention:
            latent, enc_attns = self.encode(x, is_return_attention=True)
            output, dec_attns = self.decode(latent, is_return_attention=True)
            return output, {
                "encoder_attentions": enc_attns,
                "decoder_attentions": dec_attns,
            }
        else:
            latent = self.encode(x)
            output = self.decode(latent)
            return output

    @classmethod
    def load_from_file(cls, path, prefix="best_"):
        with open(os.path.join(path, f"{prefix}model_params.json"), "r") as f:
            model_params = json.load(f)
        filtered_model_params = filter_kwargs_for_function(cls.__init__, model_params)
        model = cls(**filtered_model_params)
        model.load_state_dict(torch.load(os.path.join(path, f"{prefix}model.pth")))
        return model

    def save_to_file(self, path, model_params, prefix=""):
        """
        モデルを指定されたパスに保存します。
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f"{prefix}model.pth"))
        with open(os.path.join(path, f"{prefix}model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)


class SolitireMAEClassifyModel(nn.Module):
    def __init__(
        self,
        dim=128,
        latent_dim=32,
        num_layer=6,
        num_heads=2,
        is_use_rope_post=False,
        embeddings_len=128,
    ):
        super().__init__()
        self.dim = dim
        encoder_type = get_encoder_block_type(
            is_use_rope_post,
        )
        params = {
            "dim": dim,
            "num_heads": num_heads,
        }
        self.enc_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.latent_proj = nn.Linear(dim, latent_dim)  # **最後に圧縮**

        self.dec_proj = nn.Linear(latent_dim, dim)  # **デコード時に元の次元に戻す**
        self.dec_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.norm = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))  # マスクトークンのパラメータ
        self.embeddings = nn.Embedding(embeddings_len, dim)
        self.dec_ln = nn.LayerNorm(dim)  # デコード時の正規化
        self.classify_proj = nn.Linear(dim, embeddings_len)  # 復元用のプロジェクション

    def encode(
        self,
        x,
        mask=None,
        is_return_attention=False,
    ):
        attentions = []
        batch_size, seq_len, _ = x.shape

        if mask is not None:
            # マスクトークンの挿入（256次元での挿入）
            mask_token_expanded = (
                self.mask_token.view(1, 1, -1)
                .expand(batch_size, seq_len, -1)
                .to(x.dtype)
            )
            x = torch.where(mask.unsqueeze(-1), mask_token_expanded, x)
        """入力トークン列から latent を計算"""
        for layer in self.enc_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Encoder

        x = self.norm(x)  # Transformer 出力を正規化
        latent = self.latent_proj(x)  # **最後に圧縮**
        return (latent, attentions) if is_return_attention else latent

    def decode(self, latent, is_return_attention=False):
        attentions = []
        """latent から元のトークン列を復元"""
        x = self.dec_proj(latent)  # **元の次元に復元**
        for layer in self.dec_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Decoder
        x = self.dec_ln(x)
        logits = self.classify_proj(x)  # 復元用のプロジェクション
        return (logits, attentions) if is_return_attention else logits

    def decode_to_indices(self, latent, max_index=None, is_return_attention=False):
        """
        latent からトークンのインデックスを復元する。
        """
        if is_return_attention:
            logits, attentions = self.decode(
                latent, is_return_attention=is_return_attention
            )
        else:
            logits = self.decode(latent, is_return_attention=is_return_attention)
        if max_index is not None:
            logits = logits[:, :, : max_index + 1]
        indices = logits.argmax(dim=-1)
        if is_return_attention:
            return indices, attentions
        else:
            return indices

    def forward(self, x, is_return_attention=False):
        """エンドツーエンドの AE"""
        if is_return_attention:
            latent, enc_attns = self.encode(x, is_return_attention=True)
            output, dec_attns = self.decode(latent, is_return_attention=True)
            return output, {
                "encoder_attentions": enc_attns,
                "decoder_attentions": dec_attns,
            }
        else:
            latent = self.encode(x)
            output = self.decode(latent)
            return output

    @classmethod
    def load_from_file(cls, path, prefix="best_"):
        with open(os.path.join(path, f"{prefix}model_params.json"), "r") as f:
            model_params = json.load(f)
        filtered_model_params = filter_kwargs_for_function(cls.__init__, model_params)
        model = cls(**filtered_model_params)
        model.load_state_dict(torch.load(os.path.join(path, f"{prefix}model.pth")))
        return model

    def save_to_file(self, path, model_params, prefix=""):
        """
        モデルを指定されたパスに保存します。
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f"{prefix}model.pth"))
        with open(os.path.join(path, f"{prefix}model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)


SolitireAbstractMEAModel = Union[SolitireMAEModel, SolitireMAEClassifyModel]


class SolitireValueModel(nn.Module):
    def __init__(
        self,
        dim=128,
        num_layer=6,
        num_heads=2,
        input_proj_dim=None,
        is_use_rope_post=False,
        embeddings_len=128,
    ):
        super().__init__()
        self.dim = dim
        self.input_proj_dim = input_proj_dim
        encoder_type = get_encoder_block_type(
            is_use_rope_post,
        )
        params = {
            "dim": dim,
            "num_heads": num_heads,
        }
        self.enc_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.regression_proj = nn.Linear(dim, 1)  # regression用のプロジェクション
        if input_proj_dim is not None and input_proj_dim != dim:
            self.enc_input_proj = nn.Sequential(
                nn.Linear(input_proj_dim, dim),
                nn.LayerNorm(dim),
            )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x,
        is_return_attention=False,
    ):
        attentions = []
        batch_size, seq_len, _ = x.shape
        if self.input_proj_dim is not None and self.input_proj_dim != self.dim:
            x = self.enc_input_proj(x)

        for layer in self.enc_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Encoder

        x = self.norm(x)  # Transformer 出力を正規化
        x = self.dropout(x)
        target = self.regression_proj(x[:, 0, :])  # **最後に圧縮**
        target = target.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        return (target, attentions) if is_return_attention else target

    @classmethod
    def load_from_file(cls, path, prefix="best_"):
        model, _ = cls.load_from_file_and_model_params(path, prefix=prefix)
        return model

    @classmethod
    def load_from_file_and_model_params(cls, path, prefix="best_"):
        with open(os.path.join(path, f"{prefix}model_params.json"), "r") as f:
            model_params = json.load(f)
        filtered_model_params = filter_kwargs_for_function(cls.__init__, model_params)
        model = cls(**filtered_model_params)
        model.load_state_dict(torch.load(os.path.join(path, f"{prefix}model.pth")))
        return model, model_params

    def save_to_file(self, path, model_params, prefix=""):
        """
        モデルを指定されたパスに保存します。
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f"{prefix}model.pth"))
        with open(os.path.join(path, f"{prefix}model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)


class SolitireTokenToValueModel(nn.Module):
    def __init__(
        self,
        ae_model: SolitireMAEModel,
        value_model: SolitireValueModel,
    ):
        super().__init__()
        self.ae_model = ae_model
        self.value_model = value_model

    def forward(
        self,
        x,
    ):
        x = self.ae_model.embeddings(x)
        latent = self.ae_model.encode(x)
        value = self.value_model(latent)
        return value

    @classmethod
    def load_from_file_by_value_model_name(
        cls, model_path, value_model_name, prefix="best_"
    ):
        value_model_path = os.path.join(model_path, "value", value_model_name)
        value_model, value_model_params = (
            SolitireValueModel.load_from_file_and_model_params(
                value_model_path, prefix=prefix
            )
        )
        ae_model_path = os.path.join(model_path, "mae", value_model_params["ae_model"])
        ae_model = SolitireMAEModel.load_from_file(ae_model_path, prefix=prefix)
        return cls(ae_model=ae_model, value_model=value_model)


class SolitireEndToEndValueModel(nn.Module):
    def __init__(
        self,
        dim=128,
        num_layer=6,
        num_heads=2,
        input_proj_dim=None,
        is_use_rope_post=False,
        embeddings_len=128,
    ):
        super().__init__()
        self.dim = dim
        self.input_proj_dim = input_proj_dim
        encoder_type = get_encoder_block_type(
            is_use_rope_post,
        )
        params = {
            "dim": dim,
            "num_heads": num_heads,
        }
        self.enc_layers = nn.ModuleList(
            [encoder_type(**params) for _ in range(num_layer)]
        )

        self.regression_proj = nn.Linear(dim, 1)  # regression用のプロジェクション
        if input_proj_dim is not None and input_proj_dim != dim:
            self.enc_input_proj = nn.Sequential(
                nn.Linear(input_proj_dim, dim),
                nn.LayerNorm(dim),
            )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

        if input_proj_dim is not None and input_proj_dim != dim:
            self.embeddings = nn.Embedding(embeddings_len, input_proj_dim)
        else:
            self.embeddings = nn.Embedding(embeddings_len, dim)

    def forward(
        self,
        x,
        is_return_attention=False,
    ):
        x = self.embeddings(x)  # トークンを埋め込みベクトルに変換
        attentions = []
        batch_size, seq_len, _ = x.shape
        if self.input_proj_dim is not None and self.input_proj_dim != self.dim:
            x = self.enc_input_proj(x)

        """入力トークン列から latent を計算"""
        for layer in self.enc_layers:
            if is_return_attention:
                x, attn_weights = layer(x, is_return_attention=True)
                attentions.append(attn_weights)
            else:
                x = layer(x)  # Transformer Encoder

        x = self.norm(x)  # Transformer 出力を正規化
        x = self.dropout(x)
        target = self.regression_proj(x[:, 0, :])  # **最後に圧縮**
        target = target.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        return (target, attentions) if is_return_attention else target

    @classmethod
    def load_from_file(cls, path, prefix="best_"):
        with open(os.path.join(path, f"{prefix}model_params.json"), "r") as f:
            model_params = json.load(f)
        filtered_model_params = filter_kwargs_for_function(cls.__init__, model_params)
        model = cls(**filtered_model_params)
        model.load_state_dict(torch.load(os.path.join(path, f"{prefix}model.pth")))
        return model

    def save_to_file(self, path, model_params, prefix=""):
        """
        モデルを指定されたパスに保存します。
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f"{prefix}model.pth"))
        with open(os.path.join(path, f"{prefix}model_params.json"), "w") as f:
            json.dump(model_params, f, indent=4)


SolitireAbstractEndToEndValueModel = Union[
    SolitireEndToEndValueModel,
    SolitireTokenToValueModel,
]
