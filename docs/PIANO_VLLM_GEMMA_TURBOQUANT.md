# Piano di Implementazione: vLLM (Main/Nightly) + TurboQuant — Gemma 3 27B su RTX 5090

> **Aggiornato**: Maggio 2026 — basato su vLLM ufficiale v0.20.1rc1.dev126+gc3868bbbe

## 📋 Sommario Esecutivo

Guida per servire **Gemma 3 27B-IT (AWQ INT4)** tramite **vLLM ufficiale (branch main)** su una NVIDIA RTX 5090, con KV cache compressa via **TurboQuant `k8v4`**.

L'algoritmo TurboQuant è integrato nel repository ufficiale di vLLM, ma la versione corrente (Maggio 2026) presenta alcuni bug con modelli che hanno layer a **sliding window attention** misti a layer full-attention (come Gemma 3 27B). Sono necessarie 4 patch runtime applicate via Dockerfile.

---

## 🎯 Obiettivi Tecnici

| Aspetto | Target | Note |
|---------|--------|------|
| **Modello** | Gemma 3 27B-IT AWQ INT4 | `gaunernst/gemma-3-27b-it-int4-awq` |
| **Pesi** | AWQ Marlin INT4 | ~17.8 GiB VRAM |
| **KV Cache** | TurboQuant `k8v4` | `torch.uint8`, slot 196 bytes, ~97 K token cache |
| **Hardware** | NVIDIA RTX 5090 | 32 GB GDDR7, Blackwell SM120, WSL |
| **Framework** | vLLM `main` (editable install) | `/app/vllm`, enforce-eager |
| **API** | OpenAI-compatible | `/v1/chat/completions`, `/v1/completions` |

---

## 🚀 Fasi di Implementazione

### Fase 1: Setup Ambiente e Build
Clone del branch `main` ufficiale `vllm-project/vllm`, installazione in modalità editable (`pip install -e .`) in una venv Python 3.12, con PyTorch CUDA 12.8. L'install editable permette di applicare le patch direttamente ai file sorgente senza ricompilare.

### Fase 2: Download Modello
Modello `gaunernst/gemma-3-27b-it-int4-awq` scaricato in `docker/models/gemma-3-27b-it-awq/` (4 shard safetensors, ~17 GiB). vLLM lo converte automaticamente in `awq_marlin` a runtime.

### Fase 3: Patch Runtime (vedi sotto)
4 patch Python applicate nel Dockerfile, dopo l'install editable, per correggere i bug di compatibilità TurboQuant + sliding window.

### Fase 4: Avvio Server
`--kv-cache-dtype turboquant_k8v4 --enforce-eager --enable-prefix-caching`

---

## 🔧 Patch Runtime Applicate

Le patch vengono applicate nel Dockerfile tramite blocchi `RUN python3 <<'PYEOF' ... PYEOF` che modificano i file sorgente dell'install editable.

---

### Patch A — `TQSlidingWindowSpec` in `kv_cache_interface.py`

**File**: `vllm/v1/kv_cache_interface.py`  
**Problema**: Gemma 3 27B ha 62 layer totali: i layer 0,1,60,61 sono esclusi da TQ (boundary protection) e usano full-attention BF16; i layer 2–29 e 32–59 sono sliding-window TurboQuant. Quando `attention.py` costruisce la spec per un layer SWA+TQ, restituisce `SlidingWindowSpec(dtype=bfloat16)` invece di una spec uint8. Questo causa un mismatch di dimensioni quando il KV cache manager tenta di reshapare i tensori raw.  
**Fix**: Aggiunge la classe `TQSlidingWindowSpec(SlidingWindowSpec)` con campo `tq_slot_size` e `real_page_size_bytes` corretto.

```python
@dataclass(frozen=True, kw_only=True)
class TQSlidingWindowSpec(SlidingWindowSpec):
    """SlidingWindowSpec with TurboQuant-aware page size."""

    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list["TQSlidingWindowSpec"]) -> "TQSlidingWindowSpec":
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs)
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)
```

---

### Patch B — `attention.py`: restituisce `TQSlidingWindowSpec` per layer SWA+TQ

**File**: `vllm/model_executor/layers/attention/attention.py`  
**Problema**: Il metodo `get_kv_cache_spec()` restituiva `SlidingWindowSpec` (BF16) anche per layer sliding-window con TurboQuant attivo, ignorando il dtype uint8.  
**Fix**: Aggiunge un branch `if self.kv_cache_dtype.startswith("turboquant_")` nel ramo `if self.sliding_window is not None`, che istanzia `TQSlidingWindowSpec` con `tq_slot_size` calcolato da `TurboQuantConfig.slot_size_aligned`.

```python
if self.sliding_window is not None:
    assert not vllm_config.model_config.use_mla, (...)
    if self.kv_cache_dtype.startswith("turboquant_"):
        from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
        from vllm.v1.kv_cache_interface import TQSlidingWindowSpec
        tq_cfg = TurboQuantConfig.from_cache_dtype(self.kv_cache_dtype, self.head_size)
        return TQSlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            tq_slot_size=tq_cfg.slot_size_aligned,
            sliding_window=self.sliding_window,
        )
    return SlidingWindowSpec(...)  # fallback BF16 per layer non-TQ
```

---

### Patch C — LCM in `kv_cache_utils.py` (`unify_kv_cache_spec_page_size`)

**File**: `vllm/v1/core/kv_cache_utils.py`  
**Problema**: La funzione `unify_kv_cache_spec_page_size` usava `max(page_sizes)` e richiedeva che tutte le page size fossero divisori del massimo. Con Gemma 3 27B + TurboQuant, i layer BF16 (skip) hanno page size 4096 byte e i layer TQ hanno page size 3136 byte (= `block_size * 16 kv_heads * 196 slot`). Questi valori non sono pairwise divisibili → `NotImplementedError`.  
**Fix**: Sostituisce `max` con `math.lcm(*page_sizes)`, garantendo sempre divisibilità intera.

```python
# Prima (buggy):
max_page_size = max(page_sizes)
if max_page_size % layer_page_size != 0:
    raise NotImplementedError(...)

# Dopo (fix LCM):
import math as _math
lcm_page_size = _math.lcm(*page_sizes)
ratio = lcm_page_size // layer_spec.page_size_bytes
new_block_size = layer_spec.block_size * ratio
```

---

### Patch D — MRO lookup in `single_type_kv_cache_manager.py`

**File**: `vllm/v1/core/single_type_kv_cache_manager.py`  
**Problema**: La funzione `get_manager_for_kv_cache_spec` usa un dict `spec_manager_map` con chiavi `{FullAttentionSpec: ..., SlidingWindowSpec: ..., ...}`. La lookup `spec_manager_map[type(kv_cache_spec)]` falliva con `KeyError` per `TQSlidingWindowSpec` perché la subclass non era registrata esplicitamente.  
**Fix**: Sostituisce la lookup diretta con una ricerca MRO (Method Resolution Order), così qualunque sottoclasse di `SlidingWindowSpec` (o di qualsiasi altra spec registrata) eredita automaticamente il manager corretto.

```python
# Prima (buggy):
manager_class = spec_manager_map[type(kv_cache_spec)]

# Dopo (fix MRO):
spec_type = type(kv_cache_spec)
for _cls in spec_type.__mro__:
    if _cls in spec_manager_map:
        manager_class = spec_manager_map[_cls]
        break
else:
    raise KeyError(
        f"No KV cache manager registered for spec type {spec_type}. "
        f"Checked MRO: {spec_type.__mro__}"
    )
```

---

## ✅ Stato Finale (Maggio 2026)

| Patch | File | Stato |
|-------|------|-------|
| **A** — `TQSlidingWindowSpec` | `kv_cache_interface.py` | ✅ Applicata |
| **B** — `attention.py` SWA+TQ | `attention.py` | ✅ Applicata |
| **C** — LCM page size unification | `kv_cache_utils.py` | ✅ Applicata |
| **D** — MRO spec_manager_map | `single_type_kv_cache_manager.py` | ✅ Applicata |

**Risultato**: Server attivo su `http://0.0.0.0:8000`, KV cache TurboQuant funzionante, 97.673 token di cache disponibili, 2.98x concorrenza per contesto 32K.

```json
{
  "model": "/models/gemma-3-27b-it-awq",
  "usage": { "prompt_tokens": 11, "completion_tokens": 61, "total_tokens": 72 },
  "system_fingerprint": "vllm-0.20.1rc1.dev126+gc3868bbbe-ea4bbd8f"
}
```

> **Avvertenza**: Le patch sono specifiche per vLLM `v0.20.1rc1.dev126+gc3868bbbe`. Con versioni successive verificare se i bug sono stati corretti upstream prima di riapplicarle.
