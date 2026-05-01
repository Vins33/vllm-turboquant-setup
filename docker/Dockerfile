# syntax=docker/dockerfile:1
# Dockerfile — vLLM (Official Main) + Gemma 3 27B
# Target: NVIDIA RTX 5090 (Blackwell SM120)

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VLLM_TARGET_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=4 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the official vLLM repo (main branch for experimental TurboQuant features)
RUN git clone https://github.com/vllm-project/vllm.git

# Create venv, install PyTorch (CUDA 12.8 wheel), then vLLM
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip "setuptools>=74" wheel && \
    /app/.venv/bin/pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    /app/.venv/bin/pip install --upgrade "setuptools>=74" packaging ninja setuptools_scm && \
    cd vllm && \
    /app/.venv/bin/pip install -e . --no-build-isolation

RUN mkdir -p /models /logs /cache

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Patch A: add TQSlidingWindowSpec to kv_cache_interface.py
# Patch B: patch attention.py to return TQSlidingWindowSpec for TQ+SWA layers
#
# Without this, sliding-window layers get SlidingWindowSpec(bfloat16), while
# the TQ backend expects uint8. After LCM unification, _reshape_kv_cache_tensors
# tries to view the bfloat16-sized raw tensor with a uint8 shape → size mismatch.
RUN python3 <<'PYEOF'
from dataclasses import replace

# ── Part A: add TQSlidingWindowSpec ──────────────────────────────────────────
filepath_iface = '/app/vllm/vllm/v1/kv_cache_interface.py'
with open(filepath_iface) as f:
    content = f.read()
old_iface = (
    '        return max_blocks * self.page_size_bytes\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class SlidingWindowMLASpec(SlidingWindowSpec):'
)
new_iface = (
    '        return max_blocks * self.page_size_bytes\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class TQSlidingWindowSpec(SlidingWindowSpec):\n'
    '    """SlidingWindowSpec with TurboQuant-aware page size."""\n'
    '\n'
    '    tq_slot_size: int = 0\n'
    '\n'
    '    @property\n'
    '    def real_page_size_bytes(self) -> int:\n'
    '        if self.tq_slot_size > 0:\n'
    '            return self.block_size * self.num_kv_heads * self.tq_slot_size\n'
    '        return super().real_page_size_bytes\n'
    '\n'
    '    @classmethod\n'
    '    def merge(cls, specs: list["TQSlidingWindowSpec"]) -> "TQSlidingWindowSpec":\n'
    '        merged = super().merge(specs)\n'
    '        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs)\n'
    '        return replace(merged, tq_slot_size=specs[0].tq_slot_size)\n'
    '\n'
    '\n'
    '@dataclass(frozen=True, kw_only=True)\n'
    'class SlidingWindowMLASpec(SlidingWindowSpec):'
)
assert old_iface in content, 'Patch A: insertion point not found in kv_cache_interface.py!'
content = content.replace(old_iface, new_iface, 1)
with open(filepath_iface, 'w') as f:
    f.write(content)
print('Patch A: TQSlidingWindowSpec added to kv_cache_interface.py')

# ── Part B: patch attention.py ────────────────────────────────────────────────
filepath_attn = '/app/vllm/vllm/model_executor/layers/attention/attention.py'
with open(filepath_attn) as f:
    content = f.read()
old_attn = (
    '        if self.sliding_window is not None:\n'
    '            assert not vllm_config.model_config.use_mla, (\n'
    '                "MLA is not supported for slidingwindow"\n'
    '            )\n'
    '            return SlidingWindowSpec(\n'
    '                block_size=block_size,\n'
    '                num_kv_heads=self.num_kv_heads,\n'
    '                head_size=self.head_size,\n'
    '                head_size_v=self.head_size_v,\n'
    '                dtype=self.kv_cache_torch_dtype,\n'
    '                kv_quant_mode=quant_mode,\n'
    '                sliding_window=self.sliding_window,\n'
    '            )\n'
    '        elif self.kv_cache_dtype.startswith("turboquant_"):'
)
new_attn = (
    '        if self.sliding_window is not None:\n'
    '            assert not vllm_config.model_config.use_mla, (\n'
    '                "MLA is not supported for slidingwindow"\n'
    '            )\n'
    '            if self.kv_cache_dtype.startswith("turboquant_"):\n'
    '                from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig\n'
    '                from vllm.v1.kv_cache_interface import TQSlidingWindowSpec\n'
    '                tq_cfg = TurboQuantConfig.from_cache_dtype(self.kv_cache_dtype, self.head_size)\n'
    '                return TQSlidingWindowSpec(\n'
    '                    block_size=block_size,\n'
    '                    num_kv_heads=self.num_kv_heads,\n'
    '                    head_size=self.head_size,\n'
    '                    head_size_v=self.head_size,\n'
    '                    dtype=self.kv_cache_torch_dtype,\n'
    '                    tq_slot_size=tq_cfg.slot_size_aligned,\n'
    '                    sliding_window=self.sliding_window,\n'
    '                )\n'
    '            return SlidingWindowSpec(\n'
    '                block_size=block_size,\n'
    '                num_kv_heads=self.num_kv_heads,\n'
    '                head_size=self.head_size,\n'
    '                head_size_v=self.head_size_v,\n'
    '                dtype=self.kv_cache_torch_dtype,\n'
    '                kv_quant_mode=quant_mode,\n'
    '                sliding_window=self.sliding_window,\n'
    '            )\n'
    '        elif self.kv_cache_dtype.startswith("turboquant_"):'
)
assert old_attn in content, 'Patch B: target not found in attention.py!'
content = content.replace(old_attn, new_attn, 1)
with open(filepath_attn, 'w') as f:
    f.write(content)
print('Patch B: attention.py patched for TQ+SWA layers')
PYEOF

# Patch D: make get_manager_for_kv_cache_spec use MRO-based lookup so that
# TQSlidingWindowSpec (a subclass of SlidingWindowSpec) resolves to the same
# manager without needing an explicit registration entry.
RUN python3 <<'PYEOF'
filepath_stm = '/app/vllm/vllm/v1/core/single_type_kv_cache_manager.py'
with open(filepath_stm) as f:
    content = f.read()
old_stm = '    manager_class = spec_manager_map[type(kv_cache_spec)]'
new_stm = (
    '    spec_type = type(kv_cache_spec)\n'
    '    for _cls in spec_type.__mro__:\n'
    '        if _cls in spec_manager_map:\n'
    '            manager_class = spec_manager_map[_cls]\n'
    '            break\n'
    '    else:\n'
    '        raise KeyError(\n'
    '            f"No KV cache manager registered for spec type {spec_type}. "\n'
    '            f"Checked MRO: {spec_type.__mro__}"\n'
    '        )'
)
assert old_stm in content, 'Patch D: target line not found in single_type_kv_cache_manager.py!'
content = content.replace(old_stm, new_stm, 1)
with open(filepath_stm, 'w') as f:
    f.write(content)
print('Patch D: MRO-based spec_manager_map lookup in single_type_kv_cache_manager.py')
PYEOF

# Patch C: use LCM instead of MAX in unify_kv_cache_spec_page_size.
# TurboQuant boundary skip layers (BF16) + TQ layers produce page sizes
# that are not pairwise divisible → LCM ensures clean integer ratios.
RUN python3 <<'PYEOF'
import math
filepath = '/app/vllm/vllm/v1/core/kv_cache_utils.py'
with open(filepath, 'r') as f:
    content = f.read()
old = (
    '    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}\n'
    '    if len(page_sizes) <= 1:\n'
    '        # All layers have the same page size, no need to unify.\n'
    '        return kv_cache_spec\n'
    '\n'
    '    max_page_size = max(page_sizes)\n'
    '    new_kv_cache_spec = {}\n'
    '    for layer_name, layer_spec in kv_cache_spec.items():\n'
    '        if layer_spec.page_size_bytes == max_page_size:\n'
    '            new_kv_cache_spec[layer_name] = layer_spec\n'
    '        else:\n'
    '            layer_page_size = layer_spec.page_size_bytes\n'
    '            if max_page_size % layer_page_size != 0:\n'
    '                raise NotImplementedError(\n'
    '                    "The page size of the layer is not divisible by the "\n'
    '                    "maximum page size. Cannot unify by adjusting block_size."\n'
    '                )\n'
    '            ratio = max_page_size // layer_page_size\n'
    '            new_block_size = layer_spec.block_size * ratio\n'
    '            new_spec = replace(layer_spec, block_size=new_block_size)\n'
    '            assert new_spec.page_size_bytes == max_page_size\n'
    '            new_kv_cache_spec[layer_name] = new_spec\n'
    '    return new_kv_cache_spec'
)
new = (
    '    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}\n'
    '    if len(page_sizes) <= 1:\n'
    '        # All layers have the same page size, no need to unify.\n'
    '        return kv_cache_spec\n'
    '\n'
    '    # Use LCM so that mixed-dtype models (e.g. TurboQuant TQ layers +\n'
    '    # BF16 boundary skip layers) always satisfy divisibility.\n'
    '    import math as _math\n'
    '    lcm_page_size = _math.lcm(*page_sizes)\n'
    '    new_kv_cache_spec = {}\n'
    '    for layer_name, layer_spec in kv_cache_spec.items():\n'
    '        if layer_spec.page_size_bytes == lcm_page_size:\n'
    '            new_kv_cache_spec[layer_name] = layer_spec\n'
    '        else:\n'
    '            ratio = lcm_page_size // layer_spec.page_size_bytes\n'
    '            new_block_size = layer_spec.block_size * ratio\n'
    '            new_spec = replace(layer_spec, block_size=new_block_size)\n'
    '            assert new_spec.page_size_bytes == lcm_page_size\n'
    '            new_kv_cache_spec[layer_name] = new_spec\n'
    '    return new_kv_cache_spec'
)
assert old in content, 'Patch target not found in kv_cache_utils.py!'
content = content.replace(old, new, 1)
with open(filepath, 'w') as f:
    f.write(content)
print('Patch C: LCM-based page size unification in kv_cache_utils.py')
PYEOF

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
