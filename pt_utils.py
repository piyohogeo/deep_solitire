import inspect
from typing import Any, Callable, Optional

import torch
from torch import nn


def build_mask(batch_size, seq_len, mask_ratio):
    num_mask_tokens = int(seq_len * mask_ratio)
    mask_indices = torch.rand(batch_size, seq_len).argsort(dim=1)[:, :num_mask_tokens]

    # マスク位置をTrueで置換
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_mask_tokens)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[batch_indices, mask_indices] = True
    return mask


def build_mask_like(x, mask_ratio):
    batch_size, seq_len = x.shape[:2]
    return build_mask(batch_size, seq_len, mask_ratio)


def nearest_embedding_indices_from_embedding_layer(
    emb: nn.Embedding, output: torch.Tensor, max_emb_index: Optional[int] = None
) -> torch.Tensor:
    """
    emb: torch.nn.Embedding, [num_tokens, embed_dim]
    output: torch.Tensor, [batch_size, token_seq_length, embed_dim]
    Returns: torch.Tensor, [batch_size, token_seq_length]
      各出力ベクトルに対する最近傍トークンID
    """
    embedding_matrix = emb.weight  # [num_tokens, embed_dim]
    if max_emb_index is not None:
        embedding_matrix = embedding_matrix[: max_emb_index + 1]
    batch_size, seq_len, embed_dim = output.shape

    # (output - embedding_matrix)の全組み合わせ距離計算
    # output: [batch, seq, 1, embed_dim]
    # embedding_matrix: [1, 1, num_tokens, embed_dim]
    diff = output.unsqueeze(2) - embedding_matrix.unsqueeze(0).unsqueeze(
        0
    )  # [batch, seq, num_tokens, embed_dim]
    dist = (diff**2).sum(dim=-1)  # [batch, seq, num_tokens]
    indices = dist.argmin(dim=-1)  # [batch, seq]
    return indices


def filter_kwargs_for_function(
    func: Callable, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    指定された関数で受け取れる引数のみを抽出して返します。

    Parameters:
        func (callable): 引数を解析する対象の関数。
        kwargs (dict): 関数に渡す引数が含まれた辞書。

    Returns:
        dict: 関数が受け取れる引数のみを含む辞書。
    """
    # 関数のシグネチャを取得
    sig = inspect.signature(func)

    # 関数のパラメータ名を取得
    valid_params = set(sig.parameters.keys())

    # kwargsから関数で受け取れるものだけをフィルタ
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in valid_params
    }

    return filtered_kwargs
