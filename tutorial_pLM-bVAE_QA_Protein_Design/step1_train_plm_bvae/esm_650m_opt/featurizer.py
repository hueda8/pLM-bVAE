"""Prepare sequence labels, residue selections, and batches for pLM-bVAE."""

import numpy as np
import os
import hashlib
import json
import shutil
from typing import Dict, List, Tuple, Iterable, Sequence, Union, Any, Optional

# =========================
# Amino-acid vocabulary
# =========================
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
PAD_IDX = 0
EOS_IDX = 1
UNK_IDX = 2

# 3..22 = 20 AA
AA2IDX: Dict[str, int] = {aa: (3 + i) for i, aa in enumerate(AA_LIST)}
IDX2AA: Dict[int, str] = {idx: aa for aa, idx in AA2IDX.items()}

class Encoder:
    """Map amino acid sequences to token IDs and reconstruct sequences from IDs."""

    def __init__(self, hparams):
        self.vocab = ['<pad>', '<EOS>', '<unk>'] + AA_LIST
        self.word2idx = {tok: i for i, tok in enumerate(self.vocab)}

    def encode(self, line: str) -> List[int]:
        seq = line.strip().upper().replace(" ", "")
        ids: List[int] = []
        for ch in seq:
            if ch in AA2IDX:
                ids.append(AA2IDX[ch])
            else:
                ids.append(UNK_IDX)
        return ids

    def _to_int_list(self, numbers: Any) -> List[int]:
        """
        Normalize various container types into a flat list[int].
        Accepts: list[int], list[list[int]], numpy arrays, tensors converted to lists, etc.
        """
        # Convert to numpy array of objects first to preserve nested shapes
        if isinstance(numbers, np.ndarray):
            arr = numbers
        else:
            try:
                arr = np.array(numbers, dtype=object)
            except Exception:
                # Fallback: wrap single value
                return [int(numbers)]
        # Flatten once
        arr = arr.reshape(-1)
        out: List[int] = []
        for v in arr:
            # Unwrap 0-d arrays or 1-element lists
            if isinstance(v, (list, tuple, np.ndarray)) and np.size(v) == 1:
                try:
                    v = int(np.array(v).reshape(()).item())
                except Exception:
                    # best-effort: take first element
                    try:
                        v = int(list(v)[0])
                    except Exception:
                        continue
            # Cast scalars to int
            if isinstance(v, (np.generic,)) or (not isinstance(v, (list, tuple, dict, set))):
                try:
                    out.append(int(v))
                except Exception:
                    # skip uncastable values
                    continue
            else:
                # skip nested containers that survived the above
                continue
        return out

    def decode_list(self, numbers: Any) -> List[str]:
        """
        Convert a sequence of token ids to list[str] of amino acids.
        Robust to nested lists like [[3],[4],...], numpy arrays, or floats convertible to int.
        Stops at EOS.
        """
        ids: List[int] = self._to_int_list(numbers)
        out: List[str] = []
        for n in ids:
            if n == EOS_IDX:
                break
            if n == PAD_IDX:
                continue
            if n == UNK_IDX:
                out.append('X')
            elif n in IDX2AA:
                out.append(IDX2AA[n])
            else:
                out.append('X')
        return out

    def decode(self, numbers: Any) -> str:
        return ''.join(self.decode_list(numbers))

def pad_int_sequences(arr: List[List[int]], ml: int = None) -> np.ndarray:
    max_len = ml or (max([len(seq) for seq in arr]) if arr else 0)
    out = np.full((len(arr), max_len), PAD_IDX, dtype=np.int32)
    for i, seq in enumerate(arr):
        L = len(seq)
        if L > 0:
            out[i, :L] = np.asarray(seq, dtype=np.int32)
    return out

def pad_emb_sequences(arr: List[np.ndarray], emb_dim: int, ml: int = None) -> np.ndarray:
    max_len = ml or (max([x.shape[0] for x in arr]) if arr else 0)
    out = np.zeros((len(arr), max_len, emb_dim), dtype=np.float32)
    for i, x in enumerate(arr):
        L = x.shape[0]
        if L > 0:
            out[i, :L, :] = x
    return out

def _stable_json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

def _build_trim_cache_key(hparams) -> str:
    key_obj = {
        "data_path": hparams.get("data_path", ""),
        "embeddings_path": hparams.get("embeddings_path", ""),
        "select_indices": hparams.get("select_indices", None),
        "allow_sidecar_indices": bool(hparams.get("allow_sidecar_indices", False)),
    }
    s = _stable_json_dumps(key_obj).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]

def _apply_index_selection_precomputed(
    emb: np.ndarray,
    labels: List[int],
    L: int,
    static_idx,
    allow_sidecar_idx: bool,
    idx_dir: str,
    sample_i: int,
):
    # Explicit residue indices take precedence over per-sample index files.
    if static_idx is not None and len(static_idx) > 0:
        sel = np.array(static_idx, dtype=np.int64) - 1
        sel = sel[(sel >= 0) & (sel < L)]
        if sel.size > 0:
            emb = emb[sel, :]
            labels = [labels[j] for j in sel.tolist()]
    elif allow_sidecar_idx and os.path.isdir(idx_dir):
        idx_path = os.path.join(idx_dir, f"{sample_i}.idx.npy")
        if os.path.isfile(idx_path):
            sel = np.load(idx_path).astype(np.int64).reshape(-1) - 1
            sel = sel[(sel >= 0) & (sel < L)]
            if sel.size > 0:
                emb = emb[sel, :]
                labels = [labels[j] for j in sel.tolist()]
    return emb, labels

def prepare_trimmed_cache(hparams, mode: str, force_rebuild: bool = False) -> str:
    """
    Trim precomputed pLM representations using explicit or per-sample residue indices.
    Reuse the cached arrays on subsequent calls.
    Return the embedding directory, e.g. .../trim_cache/<key>/<mode>.
    """
    assert mode in ["train", "dev", "test"]

    data_file = os.path.join(hparams["data_path"], f"{mode}.txt")
    src_emb_dir = os.path.join(hparams["embeddings_path"], mode)
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.isdir(src_emb_dir):
        raise FileNotFoundError(f"Embeddings dir not found: {src_emb_dir}")

    cache_root = hparams.get("trimmed_cache_root", os.path.join(hparams["embeddings_path"], "trim_cache"))
    cache_key = _build_trim_cache_key(hparams)
    dst_emb_dir = os.path.join(cache_root, cache_key, mode)
    done_flag = os.path.join(dst_emb_dir, ".done.json")

    if (not force_rebuild) and os.path.isfile(done_flag):
        return dst_emb_dir

    os.makedirs(dst_emb_dir, exist_ok=True)

    encoder = Encoder(hparams)
    static_idx = hparams.get("select_indices", None)
    allow_sidecar_idx = bool(hparams.get("allow_sidecar_indices", False))
    idx_dir = os.path.join(src_emb_dir, "indices")

    n_lines = 0
    for i, line in enumerate(open(data_file)):
        n_lines += 1
        labels = encoder.encode(line.strip())

        src_path = os.path.join(src_emb_dir, f"{i}.npy")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Missing source embedding: {src_path}")

        emb = np.load(src_path).astype(np.float32)
        L = min(len(labels), emb.shape[0])
        labels = labels[:L]
        emb = emb[:L, :]

        emb_trim, labels_trim = _apply_index_selection_precomputed(
            emb=emb,
            labels=labels,
            L=L,
            static_idx=static_idx,
            allow_sidecar_idx=allow_sidecar_idx,
            idx_dir=idx_dir,
            sample_i=i,
        )

        # Cache embeddings only; the batch generator applies the same selection to labels.
        out_path = os.path.join(dst_emb_dir, f"{i}.npy")
        np.save(out_path, emb_trim.astype(np.float32), allow_pickle=False)

    with open(done_flag, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "num_samples": n_lines,
                "cache_key": cache_key,
                "source_emb_dir": src_emb_dir,
                "select_indices": static_idx,
                "allow_sidecar_indices": allow_sidecar_idx,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return dst_emb_dir

def gen_batch(buf: Dict[int, Tuple[np.ndarray, List[int]]], hparams, keep_order: bool):
    """Group similar sequence lengths within the padded-token budget."""
    id0 = min(list(buf))
    batch = [buf[id0]]
    idxes = [id0]
    max_len = buf[id0][0].shape[0]
    order = sorted(list(buf)) if keep_order else sorted(list(buf), key=lambda x: abs(buf[x][0].shape[0] - buf[id0][0].shape[0]))
    for idx in order:
        if idx == id0:
            continue
        max_len = max(max_len, buf[idx][0].shape[0])
        if (len(batch) + 1) * max_len > hparams['tokens_per_batch']:
            break
        batch.append(buf[idx]); idxes.append(idx)

    emb_list = [x[0] for x in batch]
    lab_list = [x[1] for x in batch]
    out_with_eos = [lab + [EOS_IDX] for lab in lab_list]
    emb_dim = emb_list[0].shape[1] if emb_list else hparams['embedding_dim']
    batch_x = pad_emb_sequences(emb_list, emb_dim, ml=max_len)
    batch_y = pad_int_sequences(out_with_eos, ml=max_len + 1)
    for idx in idxes: del buf[idx]
    return batch_x, batch_y

def batch_generator(hparams, mode: str, start_from: int = 0, keep_order: bool = False):
    assert mode in ['train', 'test', 'dev']
    if not hparams.get('input_is_precomputed', True):
        for data in batch_generator_text(hparams, mode, start_from, keep_order):
            yield data
        return

    encoder = Encoder(hparams)
    filename = os.path.join(hparams['data_path'], mode + '.txt')
    emb_dir = os.path.join(hparams['embeddings_path'], mode)
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    # Residue selection uses explicit indices or per-sample index files.
    static_idx = hparams.get('select_indices', None)  # 1-based residue positions, e.g. [1, 3, 5].
    allow_sidecar_idx = bool(hparams.get('allow_sidecar_indices', False))  # Allow per-sample indices in i.idx.npy files.

    # Locate the original residue indices even when embeddings have been cached.
    if bool(hparams.get("selection_already_applied", False)):

        src_root = hparams.get("source_embeddings_path_for_indices", hparams["embeddings_path"])
        idx_dir = os.path.join(src_root, mode, "indices")
    else:
        idx_dir = os.path.join(emb_dir, "indices")

    buf: Dict[int, Tuple[np.ndarray, List[int]]] = {}
    for i, line in enumerate(open(filename)):
        if i < start_from: continue
        labels_full = encoder.encode(line.strip())
        emb_path = os.path.join(emb_dir, f"{i}.npy")
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(f"Missing embedding file for line {i}: {emb_path}")
        emb = np.load(emb_path).astype(np.float32)

        selection_already_applied = bool(hparams.get("selection_already_applied", False))

        if selection_already_applied:
            # Embeddings are already trimmed; select labels using full-sequence positions.
            labels = labels_full
            L_full = len(labels)
            # Explicit residue indices take precedence over per-sample index files.
            if static_idx is not None and len(static_idx) > 0:
                sel = np.array(static_idx, dtype=np.int64) - 1
                sel = sel[(sel >= 0) & (sel < L_full)]
                if sel.size > 0:
                    labels = [labels[j] for j in sel.tolist()]
            elif allow_sidecar_idx and os.path.isdir(idx_dir):
                idx_path = os.path.join(idx_dir, f"{i}.idx.npy")
                if os.path.isfile(idx_path):
                    sel = np.load(idx_path).astype(np.int64).reshape(-1) - 1
                    sel = sel[(sel >= 0) & (sel < L_full)]
                    if sel.size > 0:
                        labels = [labels[j] for j in sel.tolist()]

            # Keep labels and embeddings aligned after residue selection.
            L2 = min(len(labels), emb.shape[0])
            labels = labels[:L2]
            emb = emb[:L2, :]

        else:
            # Apply the same residue selection to embeddings and labels.
            labels = labels_full
            L = min(len(labels), emb.shape[0])
            labels = labels[:L]
            emb = emb[:L, :]

            if static_idx is not None and len(static_idx) > 0:
                sel = np.array(static_idx, dtype=np.int64) - 1 # Convert 1-based positions to 0-based indices.
                sel = sel[(sel >= 0) & (sel < L)]
                if sel.size > 0:
                    emb = emb[sel, :]
                    labels = [labels[j] for j in sel.tolist()]
            elif allow_sidecar_idx and os.path.isdir(idx_dir):
                idx_path = os.path.join(idx_dir, f"{i}.idx.npy")
                if os.path.isfile(idx_path):
                    sel = np.load(idx_path).astype(np.int64).reshape(-1) - 1 # Convert 1-based positions to 0-based indices.
                    sel = sel[(sel >= 0) & (sel < L)]
                    if sel.size > 0:
                        emb = emb[sel, :]
                        labels = [labels[j] for j in sel.tolist()]

        buf[i] = (emb, labels)
        if len(buf) >= hparams['buffer_size']:
            yield gen_batch(buf, hparams, keep_order)
    while len(buf) != 0:
        yield gen_batch(buf, hparams, keep_order)

# ============ Text generator for on-the-fly pLM ============
def _build_text_sel_cache_key(hparams) -> str:
    key_obj = {
        "data_path": hparams.get("data_path", ""),
        "embeddings_path": hparams.get("embeddings_path", ""),
        "select_indices": hparams.get("select_indices", None),
        "allow_sidecar_indices": bool(hparams.get("allow_sidecar_indices", False)),
    }
    s = _stable_json_dumps(key_obj).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:16]

def _prepare_text_selection_cache(hparams, mode: str) -> Dict[int, Optional[List[int]]]:
    """
    Cache 0-based residue indices in JSON for on-the-fly pLM extraction.
    Return a mapping from sample index to selected indices or None.
    """
    assert mode in ["train", "dev", "test"]

    data_file = os.path.join(hparams["data_path"], f"{mode}.txt")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    cache_root = hparams.get(
        "text_selection_cache_root",
        os.path.join(hparams.get("embeddings_path", "."), "text_sel_cache")
    )
    cache_key = _build_text_sel_cache_key(hparams)
    cache_dir = os.path.join(cache_root, cache_key, mode)
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, "sel_indices.json")
    done_flag = os.path.join(cache_dir, ".done.json")
    force_rebuild = bool(hparams.get("text_selection_cache_force_rebuild", False))

    if (not force_rebuild) and os.path.isfile(cache_path) and os.path.isfile(done_flag):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): (v if v is None else list(map(int, v))) for k, v in raw.items()}

    encoder = Encoder(hparams)
    static_idx = hparams.get("select_indices", None)
    allow_sidecar_idx = bool(hparams.get("allow_sidecar_indices", False))
    emb_dir = os.path.join(hparams.get("embeddings_path", ""), mode)
    idx_dir = os.path.join(emb_dir, "indices")

    out: Dict[int, Optional[List[int]]] = {}
    for i, line in enumerate(open(data_file)):
        labels = encoder.encode(line.strip())
        L = len(labels)

        sel0: Optional[List[int]] = None
        if static_idx is not None and len(static_idx) > 0:
            sel = np.array(static_idx, dtype=np.int64) - 1
            sel = sel[(sel >= 0) & (sel < L)]
            sel0 = sel.tolist()
        elif allow_sidecar_idx and os.path.isdir(idx_dir):
            idx_path = os.path.join(idx_dir, f"{i}.idx.npy")
            if os.path.isfile(idx_path):
                sel = np.load(idx_path).astype(np.int64).reshape(-1) - 1
                sel = sel[(sel >= 0) & (sel < L)]
                sel0 = sel.tolist()

        out[i] = sel0

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True)

    with open(done_flag, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "num_samples": len(out),
                "cache_key": cache_key,
                "allow_sidecar_indices": allow_sidecar_idx,
                "select_indices": static_idx,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    return out

def _gen_text_batch(buf: Dict[int, Tuple[str, List[int], List[int]]], hparams, keep_order: bool):
    """
    Build a batch from (sequence, labels, selected_indices) buffer entries.
    Selected indices are 0-based; an empty selection leaves the sequence intact.
    Return sequences, padded labels, and the per-sample residue indices.
    """
    id0 = min(list(buf))
    batch = [buf[id0]]; idxes = [id0]
    max_len = len(buf[id0][1])
    order = sorted(list(buf)) if keep_order else sorted(list(buf), key=lambda x: abs(len(buf[x][1]) - len(buf[id0][1])))
    for idx in order:
        if idx == id0: continue
        max_len = max(max_len, len(buf[idx][1]))
        if (len(batch) + 1) * max_len > hparams['tokens_per_batch']:
            break
        batch.append(buf[idx]); idxes.append(idx)

    seq_list = [x[0] for x in batch]
    lab_list = [x[1] for x in batch]
    sel_list = [x[2] for x in batch]  # 0-based indices per sample
    out_with_eos = [lab + [EOS_IDX] for lab in lab_list]
    batch_y = pad_int_sequences(out_with_eos, ml=max_len + 1)
    for idx in idxes: del buf[idx]
    # The model applies these indices to both pLM representations and labels.

    return seq_list, batch_y, sel_list

def batch_generator_text(hparams, mode: str, start_from: int = 0, keep_order: bool = False):
    """
    Generate sequence batches for on-the-fly pLM extraction.
    Read 1-based positions from select_indices or per-sample i.idx.npy files.
    Return 0-based residue indices for selection in the model.
    """
    assert mode in ['train', 'test', 'dev']
    encoder = Encoder(hparams)
    filename = os.path.join(hparams['data_path'], mode + '.txt')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Data text file not found: {filename}")

    # Use the same index-file layout as the precomputed embedding workflow.
    emb_dir = os.path.join(hparams.get('embeddings_path', ''), mode)
    idx_dir = os.path.join(emb_dir, "indices")


    static_idx = hparams.get('select_indices', None)  # 1-based indices
    allow_sidecar_idx = bool(hparams.get('allow_sidecar_indices', False))

    use_sel_cache = bool(hparams.get("use_text_selection_cache", True))
    sel_cache: Optional[Dict[int, Optional[List[int]]]] = None
    if use_sel_cache:
        sel_cache = _prepare_text_selection_cache(hparams, mode)

    buf: Dict[int, Tuple[str, List[int], Optional[List[int]]]] = {}
    for i, line in enumerate(open(filename)):
        if i < start_from: continue
        seq = line.strip()
        labels = encoder.encode(seq)
        L = len(labels)

        # Convert 1-based residue positions to 0-based indices, preferring explicit positions.
        if sel_cache is not None:
            sel0 = sel_cache.get(i, None)
        else:
            sel0: Optional[List[int]] = None
            if static_idx is not None and len(static_idx) > 0:
                sel = np.array(static_idx, dtype=np.int64) - 1  # 0-based
                sel = sel[(sel >= 0) & (sel < L)]
                sel0 = sel.tolist()
            elif allow_sidecar_idx and os.path.isdir(idx_dir):
                idx_path = os.path.join(idx_dir, f"{i}.idx.npy")
                if os.path.isfile(idx_path):
                    sel = np.load(idx_path).astype(np.int64).reshape(-1) - 1
                    sel = sel[(sel >= 0) & (sel < L)]
                    sel0 = sel.tolist()

        buf[i] = (seq, labels, sel0)

        if len(buf) >= hparams['buffer_size']:
            yield _gen_text_batch(buf, hparams, keep_order)
    while len(buf) != 0:
        yield _gen_text_batch(buf, hparams, keep_order)
