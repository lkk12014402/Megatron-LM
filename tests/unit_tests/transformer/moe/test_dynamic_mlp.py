# Copyright (C) 2025 Intel Corporation

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import call_mark_step
from megatron.legacy.model import Float16Module
from megatron.training.arguments import parse_args
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestIntelDynamicMLP:

    def __init__(
        self,
        tp_size=1,
        pp_size=1,
        fused_weights=True,
        permuted_weights=True,
        cpu_init=False,
        activation_func=F.silu,
    ):

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
        )

        num_layers = 1
        self.hidden_size = 16
        self.num_experts = 8
        self.activation_func = activation_func
        self.tp_size = tp_size
        self.permuted_weights = permuted_weights
        self.fused_weights = fused_weights

        tf_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            activation_func=self.activation_func,
            use_cpu_initialization=cpu_init,
            gated_linear_unit=True,
            add_bias_linear=False,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
        )

        self.fc1_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc2_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc1_ffn_hidden_size *= 2

        # Vanilla Sequential GEMM
        # Set random seed for reproducability
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        transformer_layer_spec = get_gpt_layer_local_spec(self.num_experts, moe_grouped_gemm=False)
        self.sequential_mlp = self.new_moe_layer(
            tf_config, transformer_layer_spec.submodules.mlp.submodules
        )

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        print("done intializing for sequential gemm")

        # Grouped HPU GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        tf_config.moe_dynamic_hpu = True
        tf_config.moe_permuted_weights = self.permuted_weights
        tf_config.moe_fused_weights = self.fused_weights
        self.dynamic_mlp = self.new_moe_layer(tf_config)
        self.weight1 = self.dynamic_mlp.experts.expert_weights[0]
        self.weight2 = (
            self.dynamic_mlp.experts.expert_weights[1]
            if self.fused_weights
            else self.dynamic_mlp.experts.expert_weights[2]
        )

        print("done intializing for grouped hpu gemm")

    def new_moe_layer(self, tf_config, mlp_submodules=None):
        moe_layer = MoELayer(copy.deepcopy(tf_config), mlp_submodules).cuda()
        moe_layer.set_layer_number(0)
        return moe_layer

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.sequential_mlp, MoELayer)
        assert isinstance(self.dynamic_mlp, MoELayer)
        assert torch.equal(self.sequential_mlp.router.weight, self.dynamic_mlp.router.weight)
        expected_num_weights = (
            self.hidden_size * self.num_experts
            + self.hidden_size
            * (self.fc1_ffn_hidden_size + self.fc2_ffn_hidden_size)
            * self.num_experts
            // self.tp_size
        )
        num_weights_smm = sum([p.numel() for p in self.sequential_mlp.parameters()])
        num_weights_gmm = self.dynamic_mlp.router.weight.numel()
        for e in range(len(self.dynamic_mlp.experts.expert_weights)):
            num_weights_gmm += sum([p.numel() for p in self.dynamic_mlp.experts.expert_weights[e]])
        # expected num weights: router linear weights+bias + MLP weights(no bias) of all experts
        assert num_weights_smm == expected_num_weights
        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # HPU GEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm

        assert len(self.weight1) == len(self.weight2) == self.num_experts
        assert all([w1.shape == self.weight1[0].shape for w1 in self.weight1])
        # All expert weights should have the same shape
        fc1_shape = self.fc1_ffn_hidden_size // self.tp_size
        # unpermuted weights sizes:
        # weight1: [h, 8h] * num_experts
        # weight2: [4h, h] * num_experts
        if self.permuted_weights:
            assert self.weight1[0].shape[1] == self.hidden_size
            assert self.weight1[0].shape[0] == fc1_shape if self.fused_weights else (fc1_shape // 2)
            assert self.weight2[0].shape[0] == self.hidden_size
        else:
            assert self.weight1[0].shape[0] == self.hidden_size
            assert self.weight1[0].shape[1] == fc1_shape if self.fused_weights else (fc1_shape // 2)
            assert self.weight2[0].shape[1] == self.hidden_size

    def test_hpu_forward_backward(self):
        self.sequential_mlp.cuda()
        self.dynamic_mlp.cuda()
        # [sequence length, batch size, hidden size]
        seq_len = 32
        batch_size = 1
        dtype = torch.bfloat16
        hidden_states = torch.rand(
            (seq_len, batch_size, self.sequential_mlp.config.hidden_size),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        # calc sequential fwd + bwd
        self.sequential_mlp.config.moe_dynamic_hpu = False
        output_smm, _ = self.sequential_mlp(hidden_states)
        output_smm.mean().backward()

        # calc grouped hpu fwd + bwd
        self.dynamic_mlp.config.moe_dynamic_hpu = True
        hidden_states_hpu = hidden_states.detach().to('cuda').requires_grad_(True)
        output_gmm, _ = self.dynamic_mlp(hidden_states_hpu)
        call_mark_step()
        output_gmm.mean().backward()
        assert output_smm.shape == output_gmm.shape
        atol = 1e-2 if dtype == torch.float else 1.6e-1
        rtol = atol
        torch.testing.assert_close(output_gmm, output_smm, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            hidden_states_hpu.grad.cpu(), hidden_states.grad.cpu(), atol=atol, rtol=rtol
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
@pytest.mark.timeout(120)
@pytest.mark.parametrize("tp_size,pp_size", [(1, 1), (8, 1), (4, 2)])
@pytest.mark.parametrize("fused_weights", [True, False])
@pytest.mark.parametrize("permuted_weights", [True, False])
@pytest.mark.parametrize("cpu_init", [True, False])
def test_dynamic_mlp_init(tp_size, pp_size, fused_weights, permuted_weights, cpu_init):
    container = TestIntelDynamicMLP(
        tp_size,
        pp_size,
        fused_weights=fused_weights,
        permuted_weights=permuted_weights,
        cpu_init=cpu_init,
        activation_func=F.silu,
    )
    container.test_constructor()
    container.teardown_method(method=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.internal
@pytest.mark.timeout(120)
@pytest.mark.parametrize("tp_size,pp_size", [(1, 1), (8, 1), (4, 2)])
@pytest.mark.parametrize("act_fun", [F.silu, F.gelu, F.relu])
def test_dynamic_mlp_fwd_bwd(tp_size, pp_size, act_fun):
    container = TestIntelDynamicMLP(tp_size, pp_size, activation_func=act_fun)
    container.test_hpu_forward_backward()
    container.teardown_method(method=None)
