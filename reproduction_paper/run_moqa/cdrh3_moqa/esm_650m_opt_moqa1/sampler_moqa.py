"""Design sequences with pLM-bVAE, PARROT predictors, and quantum annealing."""

import numpy as np
import dimod
import os
import time
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from parrot import py_predictor as ppp

from fmqa import TorchFMBQM

# -----------------------------
# Config
# -----------------------------
# multi-objective optimization direction for predictors
# "max": larger is better, "min": smaller is better
OBJECTIVE_DIRECTION = "min"

N_LAYERS_INIT = 20
LAYER_DECAY = 1.0  # A value of 1.0 keeps the Pareto-layer count unchanged.

# Factorization-machine training settings.
FM_RANK = 8
FM_LR = 0.01
FM_WEIGHT_DECAY = 0.01
FM_EPOCHS = 1000
FM_PATIENCE = 50
FM_AUTO_SCALE = False
FM_VAL_RATIO = 0
FM_BATCH_SIZE = 64


ANNEALING_TIME = 20
NUM_READS = 10
MAX_RETRIES = 20
N_ITER = 200

DWAVE_ENDPOINT = "https://cloud.dwavesys.com/sapi"
DWAVE_SOLVER = "Advantage_system6"
DWAVE_TOKEN = "..."  # set your token

# penalty for invalid/failed prediction
INVALID_OBJ_VALUE = -1.0e6
MAX_SAME_AA_RUN = 5

# HV reference point (in RAW objective space)
HV_REF_POINT = np.array([3.501031315217148, 3.501031315217148, 3.501031315217148], dtype=np.float64) # Reference threshold in the original objective units.

# -----------------------------
# Predictors (multi objective: 3 objectives)
# -----------------------------
my_predictor_1 = ppp.Predictor(
    "./parrot_models/model_network_1.pt",
    dtype="sequence",
    encoding_file="../../zscale3_encoding.txt",
)
my_predictor_2 = ppp.Predictor(
    "./parrot_models/model_network_2.pt",
    dtype="sequence",
    encoding_file="../../zscale3_encoding.txt",
)
my_predictor_3 = ppp.Predictor(
    "./parrot_models/model_network_3.pt",
    dtype="sequence",
    encoding_file="../../zscale3_encoding.txt",
)


# -----------------------------
# QPU diagnostics helpers
# -----------------------------
def _get_max_chain_length(embedding):
    if not embedding:
        return 0
    return max(len(chain) for chain in embedding.values())


def _max_chain_length_from_anywhere(res, sampler):
    emb = getattr(getattr(sampler, "child", None), "embedding", None)
    if emb:
        return int(_get_max_chain_length(emb))

    info = getattr(res, "info", {}) or {}
    ctx = info.get("embedding_context")
    if isinstance(ctx, dict):
        emb2 = ctx.get("embedding")
        if emb2:
            return int(_get_max_chain_length(emb2))
    return 0


def _safe_get_chain_strength_from_info(res):
    info = getattr(res, "info", {}) or {}
    for key in ("chain_strength", "embedding_context", "chain_strengths"):
        if key in info:
            val = info[key]
            if isinstance(val, dict) and "chain_strength" in val:
                return val["chain_strength"]
            return val
    return None

def compute_energy_metrics(res, bqm):
    """Summarize QPU energies and chain breaks, retaining raw and normalized values."""
    offset = float(getattr(bqm, "offset", 0.0))
    max_abs_h = float(max((abs(v) for v in bqm.linear.values()), default=0.0))
    max_abs_J = float(max((abs(v) for v in bqm.quadratic.values()), default=0.0))
    scale = max(max_abs_h, max_abs_J)
    num_vars = int(len(bqm.variables))
    denom = num_vars * scale

    E = res.record.energy.astype(float) # hamiltonian_reads
    cbf_reads = res.record["chain_break_fraction"].astype(float)

    E0 = E - offset
    E_norm_reads = np.full_like(E0, np.nan, dtype=float) if denom == 0.0 else (E0 / denom)

    best_i = int(np.argmin(E))
    best_energy = float(E[best_i])
    cbf_best = float(cbf_reads[best_i])
    E_best_norm = float(E_norm_reads[best_i]) if len(E_norm_reads) > 0 else float("nan")

    return {
        "offset": offset,
        "max_abs_h": max_abs_h,
        "max_abs_J": max_abs_J,
        "E": E,
        "cbf_reads": cbf_reads,
        "E_norm_reads": E_norm_reads,
        "best_i": best_i,
        "best_energy": best_energy,
        "cbf_best": cbf_best,
        "E_best_norm": E_best_norm,
    }


def append_qpu_log_csv(csv_path, iter_idx, res, sampler, bqm, requested_chain_strength=None):
    m = compute_energy_metrics(res, bqm)

    cbf_mean = float(m["cbf_reads"].mean())
    cbf_max = float(m["cbf_reads"].max())
    max_chain_length = _max_chain_length_from_anywhere(res, sampler)

    used_chain_strength = requested_chain_strength
    if used_chain_strength is None:
        used_chain_strength = _safe_get_chain_strength_from_info(res)

    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if need_header:
            f.write(
                "iter,best_energy,E_norm,offset,max_abs_h,max_abs_J,cbf_mean,cbf_max,cbf_best,max_chain_length,chain_strength\n"
            )
        f.write(
            f"{iter_idx},{m['best_energy']:.16g},{m['E_best_norm']:.16g},"
            f"{m['offset']:.16g},{m['max_abs_h']:.16g},{m['max_abs_J']:.16g},"
            f"{cbf_mean:.16g},{cbf_max:.16g},{m['cbf_best']:.16g},{max_chain_length},"
            f"{'' if used_chain_strength is None else used_chain_strength}\n"
        )


def sample_with_more_embedding_effort(
    bqm, token, num_reads, annealing_time, solver, endpoint, max_retries, #chain_strength,
):
    last = None
    for k in range(1, max_retries + 1):
        try:
            sampler = EmbeddingComposite(
                DWaveSampler(endpoint=endpoint, token=token, solver=solver),
            )
            return sampler.sample(
                bqm,
                num_reads=num_reads,
                annealing_time=annealing_time,
                chain_break_fraction=True,
            )
        except ValueError as e:
            last = e
            print(f"[attempt {k}/{max_retries}] embedding failed: {e}")
            time.sleep(0.2)
    raise last


# -----------------------------
# Multi-objective helpers
# -----------------------------
def apply_objective_direction(values: np.ndarray, direction: str) -> np.ndarray:
    """Convert predictor scores to maximization objectives for Pareto ranking."""
    if direction == "max":
        return np.asarray(values, dtype=np.float32)
    if direction == "min":
        # convert minimization objective to maximization space
        return -np.asarray(values, dtype=np.float32)
    raise ValueError("OBJECTIVE_DIRECTION must be 'max' or 'min'")

def best_value(values: np.ndarray, direction: str) -> float:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return float("nan")
    if direction == "max":
        return float(np.max(vals))
    if direction == "min":
        return float(np.min(vals))
    raise ValueError("OBJECTIVE_DIRECTION must be 'max' or 'min'")

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return a mask for the non-dominated front when all objectives are maximized."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

def ParetoV(objective_values: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute and log the Pareto hypervolume for maximization objectives.

    objective_values: shape (n, 3)
    ref_point: shape (3,)
    """
    front_mask = is_pareto_efficient(objective_values)
    front = objective_values[front_mask]

    if front.shape[0] == 0:
        pv = 0.0
    else:
        x0 = torch.from_numpy(front.astype(np.float64))
        xref = torch.from_numpy(ref_point.astype(np.float64))
        hv = Hypervolume(xref)
        pv = float(hv.compute(x0))

    with open("./model_output/binary/hv_score.txt", "a") as f:
        f.write(f"{pv}\n")
    return pv

def NSPareto(objective_values: np.ndarray, NLayers: int) -> np.ndarray:
    """
    Return per-sample score (lower is better):
      - inside top-NLayers fronts: rank / N_LAYERS_INIT
      - outside: log(1 + rank)
    rank is 1-based Pareto front index.
    """
    n = objective_values.shape[0]
    scores = np.empty(n, dtype=np.float32)

    rem_idx = np.arange(n)
    rem_obj = objective_values.copy()
    rank = 1

    # Assign non-dominated sorting ranks to all valid samples.
    ranks = np.empty(n, dtype=np.int32)
    while rem_obj.shape[0] > 0:
        front_mask = is_pareto_efficient(rem_obj)
        front_global_idx = rem_idx[front_mask]
        ranks[front_global_idx] = rank

        rem_idx = rem_idx[~front_mask]
        rem_obj = rem_obj[~front_mask]
        rank += 1

    # Convert ranks into scalar targets for factorization-machine training.
    nl = int(max(NLayers, 1))
    denom = float(max(N_LAYERS_INIT, 1))  # Keep the normalization scale fixed across iterations.
    inside = ranks <= nl

    scores[inside] = ranks[inside].astype(np.float32) / denom
    scores[~inside] = np.log1p(ranks[~inside].astype(np.float32))

    return scores

def objectives_multi(seq_list):
    """Evaluate each sequence with the three trained PARROT predictors."""
    p1, p2, p3 = [], [], []
    for s in seq_list:
        s = (s or "").strip()
        if len(s) == 0 or "<pad>" in s or "<unk>" in s or has_run_of_same_char(s, MAX_SAME_AA_RUN):
            p1.append(INVALID_OBJ_VALUE)
            p2.append(INVALID_OBJ_VALUE)
            p3.append(INVALID_OBJ_VALUE)
            continue

        try:
            v1 = float(my_predictor_1.predict(s)[0])
            v2 = float(my_predictor_2.predict(s)[0])
            v3 = float(my_predictor_3.predict(s)[0])

            if not np.isfinite(v1):
                v1 = INVALID_OBJ_VALUE
            if not np.isfinite(v2):
                v2 = INVALID_OBJ_VALUE
            if not np.isfinite(v3):
                v3 = INVALID_OBJ_VALUE

            p1.append(v1)
            p2.append(v2)
            p3.append(v3)
        except Exception:
            p1.append(INVALID_OBJ_VALUE)
            p2.append(INVALID_OBJ_VALUE)
            p3.append(INVALID_OBJ_VALUE)

    return (
        np.asarray(p1, dtype=np.float32),
        np.asarray(p2, dtype=np.float32),
        np.asarray(p3, dtype=np.float32),
    )

def valid_objective_mask_raw(obj_mat_raw: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(obj_mat_raw), axis=1) & np.all(obj_mat_raw != INVALID_OBJ_VALUE, axis=1)

# -----------------------------
# Sequence utilities
# -----------------------------
def seq_fix():
    with open("./model_output/binary/decoded.txt", "r") as f:
        lines = f.readlines()

    seq_d = [i.replace(" ", "").rstrip() for i in lines]

    bad_pad = [s for s in seq_d if "<pad>" in s]
    bad_empty = [s for s in seq_d if len(s) == 0]
    bad_unk = [s for s in seq_d if "<unk>" in s]

    filler = "S" * 56

    for t in bad_pad:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    for t in bad_empty:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    for t in bad_unk:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    return seq_d

def has_run_of_same_char(s: str, n: int) -> bool:
    if n <= 1:
        return len(s) > 0
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
    return False


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs("./model_output/binary/fm_log", exist_ok=True)
    # Decode initial binary vector set
    os.system(
        'CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml '
        "--checkpoint_kind best "
        "--decode_strategy greedy"
    )

    c0 = [x.split(" ")[:] for x in open("./model_output/binary/vectors.txt").readlines()]

    vectors_all = []
    for k in range(len(c0)):
        vectors = []
        for i in c0[k]:
            vectors.append(int(float(i)))
        vectors_all.append(vectors)
    vectors_all = np.array(vectors_all, dtype=np.float32)

    # Initial sequence eval (multi objective)
    seq_d_all = seq_fix()
    pred1_raw_all, pred2_raw_all, pred3_raw_all = objectives_multi(seq_d_all)
    obj_raw_all = np.stack([pred1_raw_all, pred2_raw_all, pred3_raw_all], axis=1).astype(np.float32)

    # Convert objectives to maximization space for Pareto/HV/FM
    pred1_all = apply_objective_direction(pred1_raw_all, OBJECTIVE_DIRECTION)
    pred2_all = apply_objective_direction(pred2_raw_all, OBJECTIVE_DIRECTION)
    pred3_all = apply_objective_direction(pred3_raw_all, OBJECTIVE_DIRECTION)
    obj_all = np.stack([pred1_all, pred2_all, pred3_all], axis=1).astype(np.float32)

    # Calculate initial Pareto hypervolume
    os.system('rm -f ./model_output/binary/hv_score.txt')
    hv_ref_point = HV_REF_POINT.copy()
    if OBJECTIVE_DIRECTION == "min":
        hv_ref_point = -hv_ref_point
    mask_hv = valid_objective_mask_raw(obj_raw_all)
    if np.any(mask_hv):
        ParetoV(obj_all[mask_hv], hv_ref_point)
    else:
        with open("./model_output/binary/hv_score.txt", "a") as f:
            f.write("0.0\n")

    # Pareto-layer target
    mask_valid = valid_objective_mask_raw(obj_raw_all)
    scores_all = np.full(obj_all.shape[0], np.nan, dtype=np.float32)
    NLayerss = float(N_LAYERS_INIT)
    SLayers = NSPareto(obj_all[mask_valid], int(round(NLayerss)))
    scores_all[mask_valid] = SLayers

    # FM train data: only valid objective rows
    x_train = vectors_all[mask_valid]
    y_train = scores_all[mask_valid].astype(np.float32)
    if len(y_train) == 0:
        raise RuntimeError("No valid samples for initial FM training (all objectives invalid).")

    # Start a new run; the following commands remove previous outputs.
    os.system("rm -f ./model_output/binary/fm_log/fm_train_log_iter_*.csv")
    os.system("rm -f ./model_output/binary/vectors_list.txt")
    os.system("rm -f ./model_output/binary/decoded_list.txt")
    os.system("rm -f ./model_output/binary/all_points_samples.txt")
    os.system("rm -f ./model_output/binary/qpu_diag_log.csv")

    # Fit the initial surrogate to Pareto-rank targets.
    fmbqm = TorchFMBQM.from_data(
        x_train,
        y_train,
        k=FM_RANK,
        lr=FM_LR,
        weight_decay=FM_WEIGHT_DECAY,
        epochs=FM_EPOCHS,
        patience=FM_PATIENCE,
        auto_scale=FM_AUTO_SCALE,
        val_ratio=FM_VAL_RATIO,
        batch_size=FM_BATCH_SIZE,
        split_seed=42,
        log_path="./model_output/binary/fm_log/fm_train_log_iter_0000.csv",
    )
    bqm = fmbqm.to_bqm()

    # initial best log (RAW values, direction-aware)
    valid_pred1_raw = pred1_raw_all[mask_valid]
    valid_pred2_raw = pred2_raw_all[mask_valid]
    valid_pred3_raw = pred3_raw_all[mask_valid]

    # iteration-only best mean (RAW values, direction-aware)
    init_raw_mat = np.stack([pred1_raw_all, pred2_raw_all, pred3_raw_all], axis=1).astype(np.float32)
    init_valid_mask = valid_objective_mask_raw(init_raw_mat)
    if np.any(init_valid_mask):
        init_mean_raw = np.mean(init_raw_mat[init_valid_mask], axis=1)
        init_best_mean = best_value(init_mean_raw, OBJECTIVE_DIRECTION)
    else:
        init_best_mean = float("nan")

    _, _, w0 = fmbqm.model.get_parameters()
    with open("./model_output/binary/all_points_best.txt", "w") as oo:
        oo.write(
            f"{0} "
            f"{best_value(valid_pred1_raw, OBJECTIVE_DIRECTION):.16g} "
            f"{best_value(valid_pred2_raw, OBJECTIVE_DIRECTION):.16g} "
            f"{best_value(valid_pred3_raw, OBJECTIVE_DIRECTION):.16g} "
            f"{init_best_mean:.16g} "
            f"{NLayerss:.16g} "
            f"{w0:.16g}\n"
        )
    print(
        f"[iter 0] best_pred1={best_value(valid_pred1_raw, OBJECTIVE_DIRECTION):.16g} "
        f"best_pred2={best_value(valid_pred2_raw, OBJECTIVE_DIRECTION):.16g} "
        f"best_pred3={best_value(valid_pred3_raw, OBJECTIVE_DIRECTION):.16g} "
        f"iter_best_pred_mean={init_best_mean:.16g}",
        flush=True,
    )

    # Sample the binary quadratic model with the D-Wave quantum annealer.
    sampler = EmbeddingComposite(
        DWaveSampler(
            endpoint=DWAVE_ENDPOINT,
            token=DWAVE_TOKEN,
            solver=DWAVE_SOLVER,
        )
    )

    for iter_idx in range(N_ITER):


        # Sample the binary quadratic model with the D-Wave quantum annealer.
        try:
            res = sampler.sample(
                bqm,
                annealing_time=ANNEALING_TIME,
                num_reads=NUM_READS,
                chain_break_fraction=True,
            )
        except ValueError as e:
            print("[DWave embedding] primary sampler failed, retrying with more effort:", e, flush=True)
            res = sample_with_more_embedding_effort(
                bqm,
                endpoint=DWAVE_ENDPOINT,
                token=DWAVE_TOKEN,
                solver=DWAVE_SOLVER,
                annealing_time=ANNEALING_TIME,
                num_reads=NUM_READS,
                max_retries=MAX_RETRIES,
            )

        append_qpu_log_csv("./model_output/binary/qpu_diag_log.csv", iter_idx + 1, res, sampler, bqm,
                          )

        # append sampled binary vectors
        vectors_all = np.r_[vectors_all, res.record["sample"]].astype(np.float32)
        vectors_sample = np.r_[res.record["sample"]]

        vectors_sample_out = [" ".join(str(int(v)) for v in row) for row in vectors_sample]

        with open("./model_output/binary/vectors.txt", "w") as f:
            for s in vectors_sample_out:
                f.write(s + "\n")

        with open("./model_output/binary/vectors_list.txt", "a") as f:
            for s in vectors_sample_out:
                f.write(s + "\n")

        # decode sampled vectors
        with open("./model_output/binary/decoded.txt", "w"):
            os.system(
                'CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml '
                "--checkpoint_kind best "
                "--decode_strategy greedy"
            )

        with open("./model_output/binary/decoded_list.txt", "a") as f_out:
            for line in open("./model_output/binary/decoded.txt").readlines():
                f_out.write(line)

        seq_d = seq_fix()

        # evaluate sampled sequences (raw objectives)
        p1_raw_s, p2_raw_s, p3_raw_s = objectives_multi(seq_d)
        # convert sampled objectives to maximization space
        p1_s = apply_objective_direction(p1_raw_s, OBJECTIVE_DIRECTION)
        p2_s = apply_objective_direction(p2_raw_s, OBJECTIVE_DIRECTION)
        p3_s = apply_objective_direction(p3_raw_s, OBJECTIVE_DIRECTION)

        # accumulate both raw (for reporting) and transformed (for optimization)
        pred1_raw_all = np.r_[pred1_raw_all, p1_raw_s].astype(np.float32)
        pred2_raw_all = np.r_[pred2_raw_all, p2_raw_s].astype(np.float32)
        pred3_raw_all = np.r_[pred3_raw_all, p3_raw_s].astype(np.float32)
        obj_raw_all = np.stack([pred1_raw_all, pred2_raw_all, pred3_raw_all], axis=1).astype(np.float32)

        pred1_all = np.r_[pred1_all, p1_s].astype(np.float32)
        pred2_all = np.r_[pred2_all, p2_s].astype(np.float32)
        pred3_all = np.r_[pred3_all, p3_s].astype(np.float32)
        obj_all = np.stack([pred1_all, pred2_all, pred3_all], axis=1).astype(np.float32)

        # Calculate Pareto hypervolume for the updated population
        mask_hv = valid_objective_mask_raw(obj_raw_all)
        if np.any(mask_hv):
            ParetoV(obj_all[mask_hv], hv_ref_point)
        else:
            with open("./model_output/binary/hv_score.txt", "a") as f:
                f.write("0.0\n")

        # update pareto layer count
        mask_valid_all = valid_objective_mask_raw(obj_raw_all)
        if iter_idx > 0:
            NLayerss = LAYER_DECAY * NLayerss
        NLayers = int(round(NLayerss))
        NLayers = max(NLayers, 1)

        SLayers = NSPareto(obj_all[mask_valid_all], NLayers)
        scores_all = np.full(obj_all.shape[0], np.nan, dtype=np.float32)
        scores_all[mask_valid_all] = SLayers

        # energy diagnostics (shared)
        metrics = compute_energy_metrics(res, bqm)
        E = metrics["E"]  # hamiltonian_reads
        cbf_reads = metrics["cbf_reads"]
        E_norm_reads = metrics["E_norm_reads"]
        E_best_norm = metrics["E_best_norm"]
        cbf_best = metrics["cbf_best"]

        # FM predictions (for log)
        fm_preds_sample = fmbqm.predict(vectors_sample).astype(np.float32)

        # best-so-far log (RAW values, direction-aware)
        valid1_raw = pred1_raw_all[mask_valid_all]
        valid2_raw = pred2_raw_all[mask_valid_all]
        valid3_raw = pred3_raw_all[mask_valid_all]

        # iteration-only best mean (RAW values, direction-aware)
        sample_raw_mat = np.stack([p1_raw_s, p2_raw_s, p3_raw_s], axis=1).astype(np.float32)
        sample_valid_mask = valid_objective_mask_raw(sample_raw_mat)
        if np.any(sample_valid_mask):
            iter_mean_raw = np.mean(sample_raw_mat[sample_valid_mask], axis=1)
            iter_best_mean = best_value(iter_mean_raw, OBJECTIVE_DIRECTION)
        else:
            iter_best_mean = float("nan")

        _, _, w0 = fmbqm.model.get_parameters()
        with open("./model_output/binary/all_points_best.txt", "a") as oo:
            oo.write(
                f"{iter_idx+1} "
                f"{best_value(valid1_raw, OBJECTIVE_DIRECTION):.16g} "
                f"{best_value(valid2_raw, OBJECTIVE_DIRECTION):.16g} "
                f"{best_value(valid3_raw, OBJECTIVE_DIRECTION):.16g} "
                f"{iter_best_mean:.16g} "
                f"{E_best_norm:.16g} {cbf_best:.16g} "
                f"{NLayerss:.16g} "
                f"{w0:.16g}\n"
            )

        # log
        print(
            f"[iter {iter_idx+1}] "
            f"best_pred1={best_value(valid1_raw, OBJECTIVE_DIRECTION):.16g} "
            f"best_pred2={best_value(valid2_raw, OBJECTIVE_DIRECTION):.16g} "
            f"best_pred3={best_value(valid3_raw, OBJECTIVE_DIRECTION):.16g} "
            f"iter_best_pred_mean={iter_best_mean:.16g} "
            f"(direction={OBJECTIVE_DIRECTION}) "
            f"E_best_norm={E_best_norm:.16g} cbf_best={cbf_best:.16g}",
            flush=True,
        )

        # per-read sample log
        with open("./model_output/binary/all_points_samples.txt", "a") as oo:
            n_reads = len(p1_raw_s)
            scores_sample = scores_all[-n_reads:]
            for i in range(len(p1_raw_s)):
                oo.write(
                    f"{iter_idx+1} "
                    f"{p1_raw_s[i]:.16g} {p2_raw_s[i]:.16g} {p3_raw_s[i]:.16g} "
                    f"{scores_sample[i]:.16g} "
                    f"{E_norm_reads[i]:.16g} {cbf_reads[i]:.16g} {E[i]:.16g} "
                    f"{float(fm_preds_sample[i]):.16g}\n"
                )

        # FM retraining on valid rows
        mask_train = valid_objective_mask_raw(obj_raw_all)
        x_train_all = vectors_all[mask_train]
        y_train_all = scores_all[mask_train].astype(np.float32)

        if len(y_train_all) == 0:
            print(f"[iter {iter_idx+1}] skip FM training: no valid samples", flush=True)
        else:
            iter_log_path = f"./model_output/binary/fm_log/fm_train_log_iter_{iter_idx+1:04d}.csv"
            fmbqm.train(
                x_train_all,
                y_train_all,
                lr=FM_LR,
                weight_decay=FM_WEIGHT_DECAY,
                epochs=FM_EPOCHS,
                patience=FM_PATIENCE,
                val_ratio=FM_VAL_RATIO,
                batch_size=FM_BATCH_SIZE,
                split_seed=42 + iter_idx, # Use a reproducible split that varies between iterations.
                log_path=iter_log_path,
            )
            bqm = fmbqm.to_bqm()


if __name__ == "__main__":
    main()
