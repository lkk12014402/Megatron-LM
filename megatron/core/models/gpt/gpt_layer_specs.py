# © 2024-2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add, get_bias_dropout_norm_add
from megatron.core.fusions.fused_dot_product_attention import FusedDotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.rmsnorm import RMSNorm
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_real_cuda_device_available

try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelLinear,
        IntelTEDotProductAttention,
        IntelTEDotProductAttentionFp8Disabled,
        IntelTENorm,
        IntelTERowParallelLinear,
        IntelTERowParallelLinearFp8Disabled,
    )
except:
    pass

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn('Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    fp8: Optional[str] = None,
    enable_fsdpa: bool = False,
    fp8_coverage: dict = {},
    context_parallel_size: int = 1,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    mlp = _get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        fp8_coverage=fp8_coverage,
    )

    use_intel_te = not is_real_cuda_device_available()
    if use_intel_te:
        from intel_transformer_engine.utils import is_gaudi3

        cp_enabled = context_parallel_size > 1
        if (is_gaudi3() or cp_enabled) and enable_fsdpa:
            core_attention_class = (
                IntelTEDotProductAttention
                if fp8_coverage.get('attention', True)
                else IntelTEDotProductAttentionFp8Disabled
            )
        elif enable_fsdpa:
            core_attention_class = FusedDotProductAttention
        else:
            core_attention_class = DotProductAttention
        linear_proj = IntelTERowParallelLinear
        linear_qkv = IntelTEColumnParallelLinear
        normalization_class = IntelTENorm
    else:
        core_attention_class = TEDotProductAttention
        linear_proj = TERowParallelLinear
        linear_qkv = TELayerNormColumnParallelLinear
        normalization_class = TENorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class if use_intel_te else IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=linear_qkv,
                    core_attention=core_attention_class,
                    linear_proj=linear_proj,
                    # TENorm significantly harms convergence when used
                    # for QKLayerNorm; we instead use the Apex implementation.
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=normalization_class if use_intel_te or num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    normalization_type: str = 'LayerNorm',
    enable_fsdpa: bool = False,
    use_pre_norm=True,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    if normalization_type not in ('LayerNorm', 'RMSNorm'):
        raise Exception(
            f'Only LayerNorm and RMSNorm are currently supported, configured {normalization_type}'
        )
    normalization_class = None
    if normalization_type == "LayerNorm":
        normalization_class = LNImpl
    elif normalization_type == "RMSNorm":
        normalization_class = RMSNorm
    core_attention_class = None
    if is_real_cuda_device_available() or not enable_fsdpa:
        core_attention_class = DotProductAttention
    else:
        core_attention_class = FusedDotProductAttention
    get_bda = get_bias_dropout_add if use_pre_norm else get_bias_dropout_norm_add
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=normalization_class,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=core_attention_class,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bda,
            pre_mlp_layernorm=normalization_class,
            mlp=mlp,
            mlp_bda=get_bda,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
    fp8_coverage: dict = {},
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        if use_te:
            if is_real_cuda_device_available():
                linear_fc1 = TELayerNormColumnParallelLinear
                linear_fc2 = TERowParallelLinear
            else:
                linear_fc1 = IntelTEColumnParallelLinear
                linear_fc2 = (
                    IntelTERowParallelLinear
                    if fp8_coverage.get('mlp_row_parallel', True)
                    else IntelTERowParallelLinearFp8Disabled
                )
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear
        return ModuleSpec(
            module=MLP, submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te:
            if is_real_cuda_device_available():
                if moe_grouped_gemm:
                    linear_fc1 = TEColumnParallelGroupedLinear
                    linear_fc2 = TERowParallelGroupedLinear
                elif fp8:
                    linear_fc1 = TEColumnParallelLinear
                    linear_fc2 = TERowParallelLinear
                else:
                    linear_fc1 = ColumnParallelLinear
                    linear_fc2 = RowParallelLinear
            else:
                # TODO:
                # linear_fc1 = IntelTEColumnParallelLinear
                # linear_fc2 = (
                #     IntelTERowParallelLinear
                #     if fp8_coverage.get('mlp_row_parallel', True)
                #     else IntelTERowParallelLinearFp8Disabled
                # )
                linear_fc1 = ColumnParallelLinear
                linear_fc2 = RowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and HAVE_TE and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=(
                MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
        )
