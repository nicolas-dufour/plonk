import torch
import torch.nn as nn
from torch import Tensor
import math

from plonk.models.positional_embeddings import PositionalEmbedding, FourierEmbedding
from einops import rearrange

torch.fx.wrap("rearrange")
from typing import Tuple, Optional
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

allow_ops_in_compiled_graph()


class FusedMLP(nn.Sequential):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__(
            nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias),
        )


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LayerNorm16Bits(torch.nn.LayerNorm):
    """
    16-bit friendly version of torch.nn.LayerNorm
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class StochatichDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.survival_prob = 1.0 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.survival_prob < 1:
            mask = (
                torch.empty(x.shape[0], 1, 1, device=x.device).uniform_()
                + self.survival_prob
            )
            mask = mask.floor()
            if self.survival_prob > 0:
                mask = mask / self.survival_prob
            return x * mask
        else:
            return x


class CrossAttentionOp(nn.Module):
    def __init__(
        self, attention_dim, num_heads, dim_q, dim_kv, use_biases=True, is_sa=False
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.use_biases = use_biases
        self.is_sa = is_sa
        if self.is_sa:
            self.qkv = nn.Linear(dim_q, attention_dim * 3, bias=use_biases)
        else:
            self.q = nn.Linear(dim_q, attention_dim, bias=use_biases)
            self.kv = nn.Linear(dim_kv, attention_dim * 2, bias=use_biases)
        self.out = nn.Linear(attention_dim, dim_q, bias=use_biases)

    def forward(self, x_to, x_from=None, attention_mask=None, materialize_sdpa=False):
        if x_from is None:
            x_from = x_to
        if self.is_sa:
            q, k, v = self.qkv(x_to).chunk(3, dim=-1)
        else:
            q = self.q(x_to)
            k, v = self.kv(x_from).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        if materialize_sdpa:
            x = self.materialize_sdpa(q, k, v, attention_mask)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def materialize_sdpa(self, q, k, v, attn_mask=None):
        scale = 1.0 / math.sqrt(q.shape[-1])

        attn_matrix = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if attn_mask is not None:
            attn_matrix = attn_matrix * attn_mask
        attn_matrix = torch.nn.functional.softmax(attn_matrix, dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", attn_matrix, v)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        retrieve_attention_scores: bool = False,
        use_16_bits_layer_norm: bool = False,
    ):
        super().__init__()
        if use_16_bits_layer_norm and not retrieve_attention_scores:
            LayerNorm = LayerNorm16Bits
        else:
            LayerNorm = nn.LayerNorm
        self.retrieve_attention_scores = retrieve_attention_scores
        self.initial_to_ln = LayerNorm(dim_q, eps=1e-6)
        attention_dim = min(dim_q, dim_kv) if attention_dim == 0 else attention_dim
        self.ca = CrossAttentionOp(
            attention_dim, num_heads, dim_q, dim_kv, is_sa=False, use_biases=use_biases
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = LayerNorm(dim_q, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_q,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)

        self.register_parameter(
            "attention_mask_dummy",
            nn.Parameter(torch.ones(1, 1, dtype=torch.bool), requires_grad=False),
        )

    def forward(
        self,
        to_tokens: Tensor,
        from_tokens: Tensor,
        to_token_mask: Optional[Tensor] = None,
        from_token_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if to_token_mask is None and from_token_mask is None:
            attention_mask = None
        else:
            if to_token_mask is None:
                to_token_mask = self.attention_mask_dummy.expand(
                    to_tokens.shape[0],
                    to_tokens.shape[1],
                )
            if from_token_mask is None:
                from_token_mask = self.attention_mask_dummy.expand(
                    from_tokens.shape[0],
                    from_tokens.shape[1],
                )
            attention_mask = from_token_mask.unsqueeze(1) * to_token_mask.unsqueeze(2)
        if self.retrieve_attention_scores:
            attention_output = self.ca(
                self.initial_to_ln(to_tokens),
                from_tokens,
                attention_mask=attention_mask,
                materialize_sdpa=True,
            )
        else:
            attention_output = self.ca(
                self.initial_to_ln(to_tokens),
                from_tokens,
                attention_mask=attention_mask,
            )
        to_tokens = to_tokens + self.ca_stochastic_depth(attention_output)
        to_tokens = to_tokens + self.ffn_stochastic_depth(
            self.ffn(self.middle_ln(to_tokens))
        )
        return to_tokens


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.1,
        retrieve_attention_scores: bool = False,
        use_16_bits_layer_norm: bool = False,
    ):
        super().__init__()
        if use_16_bits_layer_norm and not retrieve_attention_scores:
            LayerNorm = LayerNorm16Bits
        else:
            LayerNorm = nn.LayerNorm
        self.retrieve_attention_scores = retrieve_attention_scores
        self.initial_ln = LayerNorm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = LayerNorm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

        self.register_parameter(
            "attention_mask_dummy",
            nn.Parameter(torch.ones(1, 1, dtype=torch.bool), requires_grad=False),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * self.attention_mask_dummy.expand(
                tokens.shape[0],
                tokens.shape[1],
            ).unsqueeze(2)
        if self.retrieve_attention_scores:
            attention_output = self.sa(
                self.initial_ln(tokens),
                attention_mask=attention_mask,
                materialize_sdpa=True,
            )
        else:
            attention_output = self.sa(
                self.initial_ln(tokens),
                attention_mask=attention_mask,
            )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens
