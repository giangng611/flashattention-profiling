"""Attention implementations used by the benchmark."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Straightforward scaled dot-product attention.

    Shapes use the PyTorch convention:
    batch x heads x sequence length x head dimension.
    """

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.shape[-2]
        mask = torch.ones(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=q.device,
        ).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


@contextmanager
def sdpa_flash_context() -> Iterator[None]:
    """Force PyTorch SDPA to use FlashAttention when the API is available."""

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
        return
    except (ImportError, AttributeError):
        pass

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            yield
        return

    raise RuntimeError("This PyTorch version does not expose an SDPA flash backend.")


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    force_flash: bool = True,
) -> torch.Tensor:
    """Run PyTorch scaled dot-product attention with FlashAttention backend."""

    if force_flash:
        with sdpa_flash_context():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Run PyTorch SDPA with the backend selected by the current device."""

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
