# Storia dei problemi e delle patch — vLLM + Gemma 4 + TurboQuant

Questo documento registra tutti i problemi incontrati cercando di avviare
`vllm-project/vllm` (fork TurboQuant, v0.20.1rc1.dev98+gb55b26520) con
**Gemma 4-26B-AWQ** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) e
`kv_cache_dtype=turboquant_k8v4` su **NVIDIA RTX 5090** (SM120, CUDA 12.8, WSL).

---

## Contesto tecnico

| Parametro | Valore |
|---|---|
| Modello | Gemma4ForConditionalGeneration, 30 layer |
| Layer sliding-window | 25 (indici 0–4, 6–10, 12–16, 18–22, 24–28), `head_dim=256`, `nkv=8` |
| Layer full-attention | 5 (indici 5, 11, 17, 23, 29), `global_head_dim=512`, `nkv=2` |
| KV cache dtype | `turboquant_k8v4` → `torch.uint8` |
| Slot size (SWA) | `slot_size_aligned(256) = 388` byte |
| Slot size (Full) | `slot_size_aligned(512) = 772` byte |
| Page size TQ SWA | `16 × 8 × 388 = 49.664` byte |
| Page size TQ Full | `16 × 2 × 772 = 24.704` byte |

---

## Problema 0 — Backend errato per Gemma 4 con TurboQuant

**File**: `vllm/model_executor/models/config.py`

**Causa**: Gemma 4 ha `head_dim=256` (locale) e `global_head_dim=512` (globale).
vLLM rilevava le dimensioni eterongenee e forzava `TRITON_ATTN` come backend
per tutti i layer — ma `TRITON_ATTN` non supporta `kv_cache_dtype=turboquant_*`.

**Errore**: crash al caricamento del modello, backend incompatibile.

**Patch 1** (applicata poi rimossa):
```python
# Se kv_cache_dtype inizia con "turboquant_", forza TURBOQUANT invece di TRITON_ATTN
kv_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
if kv_dtype and kv_dtype.startswith("turboquant_"):
    vllm_config.attention_config.backend = AttentionBackendEnum.TURBOQUANT
else:
    vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
```

---

## Problema 1 — `supports_kv_cache_dtype` rifiuta `"auto"`

**File**: `vllm/v1/attention/backends/turboquant_attn.py`

**Causa**: il meccanismo di *boundary skip layers* assegna `kv_cache_dtype="auto"`
ai layer 0, 1, 28, 29 (BF16 puro, protezione qualità). Il metodo
`supports_kv_cache_dtype` del backend TurboQuant restituiva `False` per `"auto"`,
causando il fallback a un backend incompatibile.

**Patch 2** (applicata poi rimossa):
```python
@classmethod
def supports_kv_cache_dtype(cls, kv_cache_dtype):
    if kv_cache_dtype is None or kv_cache_dtype == "auto":
        return True          # accetta skip layers
    return kv_cache_dtype.startswith("turboquant_")

@classmethod
def supports_mm_prefix(cls) -> bool:
    return True              # Gemma 4 è multimodale
```

---

## Problema 2 — 5 crash in `TurboQuantAttentionImpl` per i skip layers

**File**: `vllm/v1/attention/backends/turboquant_attn.py`

**Causa**: i 4 layer con `kv_cache_dtype="auto"` entravano comunque nel codice TQ
(`from_cache_dtype`, reshape TQ, kernel TQ) che si aspettava solo dtype `turboquant_*`.

**Patch 3** (applicata poi rimossa) — 5 guard separati:
1. `__init__`: `tq_config = None` se dtype non è `turboquant_*`
2. `get_kv_cache_shape`: shape standard flash-attn `(2, blocks, block_size, nkv, head_size)` per skip layers
3. `_ensure_on_device`: early return se `tq_config is None`
4. `do_kv_cache_update`: scatter KV con `reshape_and_cache_flash` per skip layers
5. `forward()`: path flash-attn standard senza TQ per skip layers

---

## Problema 3 — `load_weights` non trova i pesi AWQ packed

**File**: `vllm/model_executor/models/gemma4.py`

**Causa**: il checkpoint `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` usa il formato
compressed-tensors con pesi packed (`gate_up_proj_packed`, `down_proj_packed`)
e scale (`*_scale`). Il mapping `expert_params_mapping` in `Gemma4TextModel.load_weights`
conteneva solo i nomi base (`gate_proj`, `down_proj`, `up_proj`) senza suffissi
`_packed` / `_scale`.

**Patch 4** (applicata poi rimossa):
```python
for expert_id in range(num_experts):
    expert_params_mapping.extend([
        ("experts.w2_weight_packed",  f"experts.{expert_id}.down_proj_packed",  expert_id, "w2"),
        ("experts.w13_weight_packed", f"experts.{expert_id}.gate_proj_packed",  expert_id, "w1"),
        ("experts.w13_weight_packed", f"experts.{expert_id}.up_proj_packed",    expert_id, "w3"),
        ("experts.w2_weight_scale",   f"experts.{expert_id}.down_proj_scale",   expert_id, "w2"),
        ("experts.w13_weight_scale",  f"experts.{expert_id}.gate_proj_scale",   expert_id, "w1"),
        ("experts.w13_weight_scale",  f"experts.{expert_id}.up_proj_scale",     expert_id, "w3"),
    ])
```

---

## Problema 4 — `page size not divisible by max page size`

**File**: `vllm/v1/core/kv_cache_utils.py` → `unify_kv_cache_spec_page_size`

**Causa**: la funzione usava `max(page_sizes)` come denominatore comune e richiedeva
che ogni page size fosse divisore del massimo. Con 4 page sizes diverse
(49.664 SWA TQ, 24.704 Full TQ, 131.072 BF16 skip SWA, 65.536 BF16 skip Full),
la divisibilità non era garantita.

**Errore**:
```
NotImplementedError: page size not divisible by max page size. Cannot unify by adjusting block_size.
```

**Patch 5** (applicata poi rimossa) — sostituisce `max` con `math.lcm`:
```python
import math as _math
lcm_page_size = _math.lcm(*page_sizes)
ratio = lcm_page_size // layer_spec.page_size_bytes
new_block_size = layer_spec.block_size * ratio
```

---

## Problema 5 — Shape mismatch nel reshape dei KV tensor

**File**: `vllm/model_executor/layers/attention/attention.py` → `get_kv_cache_spec`

**Causa**: la funzione controllava `sliding_window is not None` **prima** di
`kv_cache_dtype.startswith("turboquant_")`. Risultato: tutti i 25 layer SWA
ricevevano `SlidingWindowSpec(dtype=bfloat16)` invece di un spec TQ.
Dopo l'unificazione LCM, `_reshape_kv_cache_tensors` chiamava
`raw_tensor.view(bfloat16)` dimezzando il numero di elementi, poi cercava di
fare reshape con shape TQ (in unità uint8) → mismatch.

**Errore**:
```
RuntimeError: shape '[76428, 16, 8, 388]' is invalid for input of size 5008785408
```

**Patch 6** (applicata poi rimossa) — due parti:

**6/A** — aggiunge `TQSlidingWindowSpec` a `kv_cache_interface.py`:
```python
@dataclass(frozen=True, kw_only=True)
class TQSlidingWindowSpec(SlidingWindowSpec):
    tq_slot_size: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes
```

**6/B** — patch `attention.py` per usare `TQSlidingWindowSpec` nei layer SWA+TQ:
```python
if self.sliding_window is not None:
    if self.kv_cache_dtype.startswith("turboquant_"):
        return TQSlidingWindowSpec(
            tq_slot_size=tq_config.slot_size_aligned, ...
        )
    return SlidingWindowSpec(...)  # fallback non-TQ
```

**Matematica verificata dopo la patch**:
```
LCM(49664, 24704) = 9.585.152 byte
SWA: block_size=3088, shape=(19300, 16, 8, 388)  ✓
Full: block_size=6208, shape=(38800, 16, 2, 772) ✓
```

---

## Problema 6 — `125 GiB KV cache needed, only 9.34 GiB available`

**File**: `vllm/engine/arg_utils.py`

**Causa**: il meccanismo di *boundary skip layers* aggiungeva layer 0, 1, 28, 29
con `kv_cache_dtype="auto"` (BF16). Questo introduceva page sizes BF16 enormi
(131.072 e 65.536 byte) nelle 4 page sizes totali.
L'LCM di tutte e 4 = **~9,8 GB per blocco** → stima memoria KV = 125 GiB.

**Errore**:
```
ValueError: 125.69 GiB KV cache needed but only 9.34 GiB available
```

**Patch 7** (applicata poi rimossa) — disabilita skip layers per modelli con head eterogenee:
```python
_has_hetero_heads = (global_head_dim is not None
                     and local_head_dim is not None
                     and global_head_dim != local_head_dim)
if not _has_hetero_heads:
    # aggiungi skip layers normalmente
else:
    logger.info("TQ: boundary skip layers disabled for heterogeneous-head model...")
    # non aggiungere skip layers
```

**Senza skip layers**: `LCM(49664, 24704) = 9.585.152` byte → memoria stimata ~1 GB ✓

---

## Perché le patch sono state rimosse

Dopo 7 patch il server non era ancora arrivato a girare in modo stabile.
L'analisi ha mostrato che i problemi erano tutti **specifici di Gemma 4** (testa
eterogenea, formato AWQ cyankiwi, boundary skip layers) e non erano presenti
nell'uso standard di vLLM.

La decisione è stata di **passare a Gemma 3-27B-IT** con vLLM standard
(immagine ufficiale), evitando la necessità di qualsiasi patch.

---

## Stato finale

| Componente | Stato |
|---|---|
| Modello `gemma-3-27b-it` | ✅ Scaricato (52 GB, 12 safetensors) |
| Dockerfile | ✅ Pulito — zero patch |
| docker-compose.yml | ✅ Aggiornato per Gemma 3 |
| entrypoint.sh | ✅ Aggiornato |
| download_model.sh | ✅ Usa `python:3.12-slim` (no dipendenza dall'immagine vllm) |
