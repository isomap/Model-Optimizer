# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU tests for Triton prefill attention kernel.

Part 1 (correctness): Compare Triton kernel output to PyTorch SDPA (Flash Attention).
Part 2 (integration): Compare full model output when attention uses Triton vs SDPA.
Part 3 (HF integration): Load HF model via from_pretrained, run model.generate with SDPA vs
Triton (first layer prefill); compare generated output text directly (no numerical tolerance).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress optional-plugin and deprecation warnings when running only this test file
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# Only run if Triton kernel is available (CUDA + triton)
from modelopt.torch.sparsity.attention_sparsity.kernels import (
    IS_AVAILABLE as TRITON_KERNEL_AVAILABLE,
)

if TRITON_KERNEL_AVAILABLE:
    from modelopt.torch.sparsity.attention_sparsity.kernels import context_attention_fwd


def _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len):
    """Reference: PyTorch F.scaled_dot_product_attention (Flash Attention when available).

    Causal attention, same layout as Triton kernel. Supports GQA by repeating k,v
    to match q head count. Returns [total_tokens, num_heads, dim].
    """
    batch = b_seq_len.shape[0]
    num_heads_q = q.shape[1]
    num_heads_kv = k.shape[1]
    out_list = []
    for b in range(batch):
        start = b_start_loc[b].item()
        length = b_seq_len[b].item()
        q_b = q[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        k_b = k[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        v_b = v[start : start + length].unsqueeze(0).permute(0, 2, 1, 3)
        if num_heads_q != num_heads_kv:
            repeat = num_heads_q // num_heads_kv
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)
        o_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        o_b = o_b.permute(0, 2, 1, 3).squeeze(0)
        out_list.append(o_b)
    return torch.cat(out_list, dim=0)


# -----------------------------------------------------------------------------
# Part 1: Kernel correctness (Triton vs Flash Attention / SDPA)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelCorrectness:
    """Compare Triton prefill kernel output to PyTorch SDPA (Flash Attention)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    def test_triton_vs_sdpa_causal_fp32(self):
        """Triton vs SDPA causal attention, float32."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 2, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(123)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=1e-2, atol=1e-2)

    def test_triton_vs_sdpa_causal_fp16(self):
        """Triton vs SDPA causal attention, float16."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 4, 2, 64
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(456)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=2e-2, atol=2e-2)

    def test_triton_vs_sdpa_causal_gqa(self):
        """Triton vs SDPA causal attention with GQA (4 q-heads, 2 kv-heads)."""
        max_seq_len = 16
        seq_lens = [8, 12]
        total_tokens = sum(seq_lens)
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(789)
        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float32)
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float32
        )
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        o_triton = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o_triton,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o_sdpa = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)
        torch.testing.assert_close(o_triton, o_sdpa, rtol=1e-2, atol=1e-2)

    def test_triton_bidirectional_forward(self):
        """Triton kernel runs bidirectional (is_causal=False); shape and finite."""
        max_seq_len = 8
        seq_lens = [4, 6]
        total_tokens = sum(seq_lens)
        num_heads, head_dim = 2, 32

        q = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        k = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        v = torch.randn(total_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16)
        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)
        o = torch.empty_like(q)

        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=max_seq_len,
            is_causal=False,
        )
        assert o.shape == q.shape
        assert not torch.isnan(o).any() and not torch.isinf(o).any()


# -----------------------------------------------------------------------------
# Part 2: Integration (small model output: Triton vs SDPA)
# -----------------------------------------------------------------------------


class _AttentionBlockModel(nn.Module):
    """Minimal model: linear Q,K,V + attention (SDPA or Triton) + output linear.

    Same weights for both modes; only the attention implementation differs.
    """

    def __init__(self, hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, use_triton=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_triton = use_triton
        assert num_heads % num_kv_heads == 0
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)

    def forward(self, x, b_start_loc, b_seq_len, max_seq_len):
        """x: [total_tokens, hidden_size]. Returns [total_tokens, hidden_size]."""
        total_tokens = x.shape[0]
        scale = 1.0 / (self.head_dim**0.5)

        q = self.q_proj(x).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(total_tokens, self.num_kv_heads, self.head_dim)

        if self.use_triton:
            o = torch.empty_like(q)
            context_attention_fwd(
                q,
                k,
                v,
                o,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_input_len=max_seq_len,
                is_causal=True,
                softmax_scale=scale,
            )
        else:
            o = _sdpa_attention_causal(q, k, v, b_start_loc, b_seq_len)

        o_flat = o.reshape(total_tokens, self.num_heads * self.head_dim)
        return self.out_proj(o_flat)


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelIntegration:
    """Compare full model output when attention uses Triton vs SDPA."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    def test_small_model_triton_vs_sdpa_output_match(self):
        """Same small model and input; Triton vs SDPA attention => same output."""
        hidden_size = 32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        seq_lens = [8, 6]
        total_tokens = sum(seq_lens)
        max_seq_len = max(seq_lens)

        b_start_loc = torch.tensor([0, seq_lens[0]], device=self.device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)

        model_sdpa = _AttentionBlockModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_triton=False,
        ).to(self.device)

        model_triton = _AttentionBlockModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_triton=True,
        ).to(self.device)

        model_triton.load_state_dict(model_sdpa.state_dict())

        torch.manual_seed(99)
        x = torch.randn(total_tokens, hidden_size, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            out_sdpa = model_sdpa(x, b_start_loc, b_seq_len, max_seq_len)
            out_triton = model_triton(x, b_start_loc, b_seq_len, max_seq_len)

        torch.testing.assert_close(out_triton, out_sdpa, rtol=1e-2, atol=1e-2)


# -----------------------------------------------------------------------------
# Part 3: HF model loading + Triton kernel (integration with AutoModelForCausalLM)
# -----------------------------------------------------------------------------


def _hf_llama_qkv_to_kernel_layout(
    module,
    hidden_states,
    position_embeddings,
):
    """Compute Q,K,V from HF Llama first-layer hidden_states and RoPE; return kernel layout.

    Returns (q, k, v) each [total_tokens, num_heads_or_kv_heads, head_dim] on same device/dtype.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.shape
    head_dim = getattr(
        module.config, "head_dim", module.config.hidden_size // module.config.num_attention_heads
    )
    num_heads = module.config.num_attention_heads
    num_kv_heads = module.config.num_key_value_heads

    # Same as LlamaAttention: view then transpose -> (bsz, num_heads, q_len, head_dim)
    query_states = (
        module.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    )
    key_states = (
        module.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    )
    value_states = (
        module.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Kernel layout: [total_tokens, num_heads, head_dim] (and num_kv_heads for k,v)
    q = query_states.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)
    k = key_states.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
    v = value_states.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_dim)
    return q, k, v


def _make_triton_first_layer_forward(original_forward):
    """Return a forward that uses context_attention_fwd for prefill (seq_len > 1); else original."""

    def triton_prefill_forward(module, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        position_embeddings = kwargs.get("position_embeddings")
        bsz, q_len, _ = hidden_states.shape
        # Use Triton only for prefill (multiple query positions); decode (q_len==1) uses original
        if position_embeddings is None or q_len <= 1:
            return original_forward(*args, **kwargs)
        q, k, v = _hf_llama_qkv_to_kernel_layout(module, hidden_states, position_embeddings)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        scale = 1.0 / (head_dim**0.5)
        b_seq_len = torch.full((bsz,), q_len, device=q.device, dtype=torch.int32)
        b_start_loc = torch.arange(bsz, device=q.device, dtype=torch.int32) * q_len
        o = torch.empty_like(q)
        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            max_input_len=q_len,
            is_causal=True,
            softmax_scale=scale,
        )
        o = o.to(hidden_states.dtype)
        attn_output = o.view(bsz, q_len, num_heads, head_dim).reshape(bsz, q_len, -1)
        attn_output = module.o_proj(attn_output)

        # Update KV cache so decode steps see correct key/value (avoids shape mismatch in later steps)
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is not None:
            num_kv_heads = module.config.num_key_value_heads
            key_states = (
                k.view(bsz, q_len, num_kv_heads, head_dim)
                .permute(0, 2, 1, 3)
                .to(hidden_states.dtype)
            )
            value_states = (
                v.view(bsz, q_len, num_kv_heads, head_dim)
                .permute(0, 2, 1, 3)
                .to(hidden_states.dtype)
            )
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

        return (attn_output, None)

    return triton_prefill_forward


@pytest.mark.skipif(
    not TRITON_KERNEL_AVAILABLE, reason="Triton kernel not available (need CUDA + triton)"
)
class TestTritonPrefillKernelHFIntegration:
    """HF model + Triton kernel: generate text and compare output (no numerical diff)."""

    @pytest.fixture(scope="class")
    def tiny_llama_dir(self, tmp_path_factory):
        """Create minimal Llama with head_dim=16 (Triton dot requires K>=16)."""
        from _test_utils.torch.transformers_models import create_tiny_llama_dir

        return create_tiny_llama_dir(
            tmp_path_factory.mktemp("tiny_llama_triton"),
            with_tokenizer=True,
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )

    @pytest.fixture(scope="class")
    def hf_llama_model(self, tiny_llama_dir):
        """Load HF Llama via AutoModelForCausalLM.from_pretrained (same as user flow)."""
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return model

    @pytest.fixture(scope="class")
    def hf_llama_tokenizer(self, tiny_llama_dir):
        """Load tokenizer for the tiny Llama (for decoding generated ids to text)."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tiny_llama_dir)

    def test_generate_text_sdpa_vs_triton_first_layer(
        self, tiny_llama_dir, hf_llama_model, hf_llama_tokenizer
    ):
        """Generate text with SDPA vs Triton (first layer); compare output text directly."""
        from transformers import AutoModelForCausalLM

        tokenizer = hf_llama_tokenizer
        device = next(hf_llama_model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Two models, same weights: one standard, one with first layer using Triton for prefill
        model_sdpa = hf_llama_model
        model_triton = AutoModelForCausalLM.from_pretrained(
            tiny_llama_dir,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model_triton.load_state_dict(model_sdpa.state_dict())

        first_attn = model_triton.model.layers[0].self_attn
        original_forward = first_attn.forward
        first_attn.forward = lambda *a, **kw: _make_triton_first_layer_forward(original_forward)(
            first_attn, *a, **kw
        )

        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        max_new_tokens = 5
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        torch.manual_seed(42)
        model_sdpa.eval()
        with torch.no_grad():
            out_sdpa = model_sdpa.generate(input_ids, **gen_kwargs)

        torch.manual_seed(42)
        model_triton.eval()
        with torch.no_grad():
            out_triton = model_triton.generate(input_ids, **gen_kwargs)

        text_sdpa = tokenizer.decode(out_sdpa[0], skip_special_tokens=True)
        text_triton = tokenizer.decode(out_triton[0], skip_special_tokens=True)
        assert text_sdpa == text_triton, (
            f"Generated text should match: SDPA got {text_sdpa!r}, Triton got {text_triton!r}"
        )
