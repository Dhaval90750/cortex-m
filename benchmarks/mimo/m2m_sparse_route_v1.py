# M2M-SparseRoute v1 / M2M-35D
# Standalone reproduction of the current champion on the original N=8 benchmark.
#
# Architecture:
#   role-aware context attention
#   dynamic router
#   Sparsemax-K competitive sparse routing
#   role-conditioned synaptic memory
#
# No explicit Q is supplied to the model.
# Q exists only in the generated dataset as a hidden-mode diagnostic/target.
#
# Expected research reference:
#   35D 5-seed mean exact graph  ~= 96.02%
#   35D 10-seed mean exact graph ~= 96.20%
#
# NOTE:
# Exact numbers can vary with PyTorch/version/hardware/randomness.
# This code is the research implementation, not a guarantee of bitwise reproduction.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from pathlib import Path

torch.set_num_threads(2)

# ============================================================
# CONFIG
# ============================================================

N_IN = 8
N_OUT = 4
ROLE_DIM = 8
HIDDEN = 32
K = 2

TR, TE = 4000, 1200
EPOCHS = 500
BS = 128
LR = 0.0012

TAU_START = 2.0
TAU_DECAY = 0.985
TAU_MIN = 0.08

SEEDS = [101, 202, 303, 404, 505,
         606, 707, 808, 909, 1001]

RULES = [
    [(0, 1), (3, 7), (4, 5), (1, 6)],
    [(2, 7), (0, 6), (1, 4), (3, 5)]
]

# ============================================================
# DATA
# ============================================================

def make_data(n, seed):
    g = torch.Generator().manual_seed(seed)

    X, R, Q, Y, T = [], [], [], [], []

    for _ in range(n):

        # Latent semantic vector.
        s = torch.randn(
            N_IN,
            generator=g
        )

        # Hidden computational mode.
        qid = 0 if s[:4].mean() > 0 else 1

        # Random physical permutation.
        p = torch.randperm(
            N_IN,
            generator=g
        )

        x = s[p]

        # Role information:
        # physical position -> semantic role.
        r = F.one_hot(
            p,
            ROLE_DIM
        ).float()

        q = F.one_hot(
            torch.tensor(qid),
            2
        ).float()

        y = []

        # Ground-truth graph.
        target = torch.zeros(
            N_OUT,
            N_IN
        )

        for j, (a, b) in enumerate(RULES[qid]):

            y.append(
                s[a] + s[b]
            )

            # Find physical locations of semantic roles a,b.
            pa = (
                r[:, a] == 1
            ).nonzero(
                as_tuple=True
            )[0].item()

            pb = (
                r[:, b] == 1
            ).nonzero(
                as_tuple=True
            )[0].item()

            target[j, pa] = 1
            target[j, pb] = 1

        X.append(x)
        R.append(r)
        Q.append(q)
        Y.append(torch.stack(y))
        T.append(target)

    return tuple(
        map(
            torch.stack,
            (X, R, Q, Y, T)
        )
    )


XTR, RTR, QTR, YTR, TTR = make_data(
    TR,
    35001
)

XTE, RTE, QTE, YTE, TTE = make_data(
    TE,
    36001
)

# ============================================================
# SPARSEMAX
# ============================================================

def sparsemax(z, dim=-1):
    """
    Sparsemax projection onto the probability simplex.

    Unlike softmax, sparsemax can produce exact zeros.
    """
    z = z - z.max(
        dim=dim,
        keepdim=True
    ).values

    zs = torch.sort(
        z,
        dim=dim,
        descending=True
    ).values

    cssv = torch.cumsum(
        zs,
        dim=dim
    ) - 1

    r = torch.arange(
        1,
        z.size(dim) + 1,
        device=z.device,
        dtype=z.dtype
    )

    shape = [1] * z.dim()
    shape[dim] = -1
    r = r.reshape(shape)

    cond = zs * r > cssv

    k_z = cond.sum(
        dim=dim,
        keepdim=True
    ).clamp(min=1)

    tau = cssv.gather(
        dim,
        k_z - 1
    ) / k_z

    return torch.clamp(
        z - tau,
        min=0
    )

# ============================================================
# MODEL
# ============================================================

class M2MSparseRoute(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # 1. NODE ENCODER
        # ----------------------------------------------------

        self.node = nn.Sequential(
            nn.Linear(
                1 + ROLE_DIM,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # 2. ROLE-AWARE CONTEXT ATTENTION
        # ----------------------------------------------------

        self.context_key = nn.Sequential(
            nn.Linear(
                HIDDEN + ROLE_DIM,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            )
        )

        self.context_query = nn.Sequential(
            nn.Linear(
                ROLE_DIM,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            )
        )

        self.context_value = nn.Sequential(
            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            )
        )

        self.context_pool = nn.Sequential(
            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU()
        )

        # Diagnostic only.
        # It is NOT used in the training objective.
        self.context_classifier = nn.Linear(
            HIDDEN,
            2
        )

        # ----------------------------------------------------
        # 3. OUTPUT ROLE ENCODER
        # ----------------------------------------------------

        self.out_role = nn.Sequential(
            nn.Linear(
                ROLE_DIM,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # 4. DYNAMIC ROUTER
        # ----------------------------------------------------

        self.router = nn.Sequential(
            nn.Linear(
                5 * HIDDEN,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                1
            )
        )

        # ----------------------------------------------------
        # 5. ROLE-CONDITIONED SYNAPTIC MEMORY
        # ----------------------------------------------------

        self.synapse = nn.Sequential(
            nn.Linear(
                2 * ROLE_DIM,
                HIDDEN
            ),
            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                1
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(N_OUT)
        )

    def forward(
        self,
        x,
        r,
        tau=TAU_MIN
    ):

        B = x.size(0)

        # ----------------------------------------------------
        # NODE REPRESENTATIONS
        # ----------------------------------------------------

        h = self.node(
            torch.cat(
                [
                    x.unsqueeze(-1),
                    r
                ],
                dim=-1
            )
        )

        # ----------------------------------------------------
        # ROLE-AWARE CONTEXT
        # ----------------------------------------------------

        key = self.context_key(
            torch.cat(
                [h, r],
                dim=-1
            )
        )

        # The query is derived from the set of role descriptors.
        # No hidden Q is supplied.
        global_role = r.mean(
            dim=1
        )

        query = self.context_query(
            global_role
        )

        value = self.context_value(
            h
        )

        attention_scores = (
            key * query[:, None]
        ).sum(
            dim=-1
        ) / math.sqrt(HIDDEN)

        attention_weights = F.softmax(
            attention_scores,
            dim=1
        )

        context_state = (
            attention_weights[:, :, None]
            * value
        ).sum(
            dim=1
        )

        c = self.context_pool(
            context_state
        )

        context_logits = self.context_classifier(
            c
        )

        # ----------------------------------------------------
        # ROUTER
        # ----------------------------------------------------

        out_roles = self.out_role(
            torch.eye(
                N_OUT,
                ROLE_DIM,
                device=x.device
            )
        )

        hi = h[:, None].expand(
            B,
            N_OUT,
            N_IN,
            HIDDEN
        )

        ho = out_roles[
            None, :, None
        ].expand(
            B,
            N_OUT,
            N_IN,
            HIDDEN
        )

        hc = c[
            :, None, None
        ].expand(
            B,
            N_OUT,
            N_IN,
            HIDDEN
        )

        pair = hi * ho

        z = torch.cat(
            [
                hi,
                ho,
                pair,
                hc,
                pair * hc
            ],
            dim=-1
        )

        scores = self.router(
            z
        ).squeeze(-1)

        # ----------------------------------------------------
        # COMPETITIVE SPARSE MICRO-ROUTING
        # ----------------------------------------------------

        P = sparsemax(
            scores / tau,
            dim=2
        )

        # Fixed K routing budget.
        G = K * P

        # ----------------------------------------------------
        # ROLE-CONDITIONED SYNAPTIC MEMORY
        # ----------------------------------------------------

        rin = r[
            :, None
        ].expand(
            B,
            N_OUT,
            N_IN,
            ROLE_DIM
        )

        rout = torch.eye(
            N_OUT,
            ROLE_DIM,
            device=x.device
        )[
            None, :, None
        ].expand(
            B,
            N_OUT,
            N_IN,
            ROLE_DIM
        )

        W = self.synapse(
            torch.cat(
                [rin, rout],
                dim=-1
            )
        ).squeeze(-1)

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        y = (
            G
            * W
            * x[:, None]
        ).sum(
            dim=2
        ) + self.bias

        return (
            y,
            G,
            W,
            context_logits,
            c,
            attention_weights
        )

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model):

    model.eval()

    with torch.no_grad():

        y, G, W, clog, c, att = model(
            XTE,
            RTE,
            TAU_MIN
        )

    # For graph scoring, take K strongest gates.
    # This measures topology separately from continuous values.
    idx = G.topk(
        K,
        dim=2
    ).indices

    selected = torch.zeros_like(
        G
    )

    selected.scatter_(
        2,
        idx,
        1.
    )

    target = TTE > 0.5

    exact_output = (
        selected == target
    ).all(
        dim=2
    ).float()

    exact_graph = (
        exact_output.all(
            dim=1
        ).float()
    )

    correct = (
        selected.bool()
        & target
    ).sum(
        (1, 2)
    ).float()

    precision = (
        correct /
        (
            selected.sum(
                (1, 2)
            ).float()
            + 1e-8
        )
    ).mean()

    recall = (
        correct /
        target.sum(
            (1, 2)
        ).float()
    ).mean()

    context_accuracy = (
        clog.argmax(dim=1)
        == QTE.argmax(dim=1)
    ).float().mean()

    zero_rate = (
        G.abs() < 1e-8
    ).float().mean()

    attention_entropy = (
        -(
            att
            * (att + 1e-8).log()
        ).sum(
            dim=1
        )
    ).mean()

    return {
        "MSE": F.mse_loss(
            y,
            YTE
        ).item(),

        "Exact graph": exact_graph.mean().item(),

        "Exact output": exact_output.mean().item(),

        "Precision": precision.item(),

        "Recall": recall.item(),

        "Context accuracy": context_accuracy.item(),

        "Gate zero rate": zero_rate.item(),

        "Attention entropy": attention_entropy.item()
    }

# ============================================================
# TRAIN
# ============================================================

results = []

q0_idx = torch.where(
    QTR[:, 0] == 1
)[0]

q1_idx = torch.where(
    QTR[:, 1] == 1
)[0]

for seed in SEEDS:

    print(
        f"\nM2M-SparseRoute | seed={seed}"
    )

    torch.manual_seed(
        seed
    )

    model = M2MSparseRoute()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # Preserve the research convention:
    # ordinary global torch RNG for training batches.
    torch.manual_seed(
        seed
    )

    for epoch in range(
        EPOCHS
    ):

        tau = max(
            TAU_MIN,
            TAU_START
            * (
                TAU_DECAY ** epoch
            )
        )

        i0 = q0_idx[
            torch.randperm(
                len(q0_idx)
            )[:BS]
        ]

        i1 = q1_idx[
            torch.randperm(
                len(q1_idx)
            )[:BS]
        ]

        idx = torch.cat(
            [i0, i1]
        )

        model.train()

        y, G, W, clog, c, att = model(
            XTR[idx],
            RTR[idx],
            tau
        )

        # IMPORTANT:
        # Output-only objective.
        # No Q loss.
        # No topology loss.
        loss = F.mse_loss(
            y,
            YTR[idx]
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

    metrics = evaluate(
        model
    )

    metrics["Seed"] = seed

    results.append(
        metrics
    )

    # Preserve the trained model for every seed so the benchmark can be
    # independently inspected and used by the downstream specialist adapter.
    _checkpoint_dir = Path(os.environ.get("MIMO_RESULTS_DIR", "results/mimo-35d-full")) / "checkpoints"
    _checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": seed,
            "state_dict": model.state_dict(),
            "architecture": "M2MSparseRoute",
            "config": {"N_IN": N_IN, "N_OUT": N_OUT, "ROLE_DIM": ROLE_DIM,
                       "HIDDEN": HIDDEN, "K": K, "TAU_MIN": TAU_MIN},
            "metrics": metrics,
        },
        _checkpoint_dir / f"seed-{seed}.pt",
    )

    print(
        f"graph={100*metrics['Exact graph']:.2f}% | "
        f"output={100*metrics['Exact output']:.2f}% | "
        f"MSE={metrics['MSE']:.6f} | "
        f"zero={100*metrics['Gate zero rate']:.2f}%"
    )

# ============================================================
# REPORT
# ============================================================

df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 120
)

print(
    "M2M-SparseRoute v1 — 10-SEED RESULT"
)

print(
    "=" * 120
)

print(
    df.to_string(
        index=False
    )
)

print("\nSUMMARY")

print(
    "Mean exact graph:",
    f"{100*df['Exact graph'].mean():.2f}%"
)

print(
    "Median exact graph:",
    f"{100*df['Exact graph'].median():.2f}%"
)

print(
    "Mean exact output:",
    f"{100*df['Exact output'].mean():.2f}%"
)

print(
    "Mean MSE:",
    f"{df['MSE'].mean():.6f}"
)

print(
    "Mean gate zero rate:",
    f"{100*df['Gate zero rate'].mean():.2f}%"
)

print(
    "Seeds >95% graph:",
    f"{int((df['Exact graph'] > .95).sum())}/{len(df)}"
)

# Save machine-readable evidence without changing the benchmark recipe above.
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime, timezone

_output = Path(os.environ.get("MIMO_RESULTS_DIR", "results/mimo-35d-full"))
_output.mkdir(parents=True, exist_ok=True)
_csv = _output / "seed_metrics.csv"
_json = _output / "summary.json"
df.to_csv(_csv, index=False)
_summary = {
    "benchmark": "M2M-SparseRoute v1 / M2M-35D archived N=8 task",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "device": "cpu" if not torch.cuda.is_available() else "cuda",
    "torch_version": torch.__version__,
    "pandas_version": pd.__version__,
    "config": {"N_IN": N_IN, "N_OUT": N_OUT, "ROLE_DIM": ROLE_DIM, "HIDDEN": HIDDEN, "K": K,
               "TR": TR, "TE": TE, "EPOCHS": EPOCHS, "BS": BS, "LR": LR,
               "TAU_START": TAU_START, "TAU_DECAY": TAU_DECAY, "TAU_MIN": TAU_MIN,
               "seeds": SEEDS},
    "metrics_mean": {k: float(df[k].mean()) for k in df.columns if k != "Seed"},
    "metrics_median": {k: float(df[k].median()) for k in df.columns if k != "Seed"},
    "seeds_over_95pct_exact_graph": int((df["Exact graph"] > .95).sum()),
    "seed_count": len(df),
    "source_script_sha256": "6BD4DDF174E9BC6B1D0B4BAD396C404991A89759295675C8EDCF0C7DFE26BF65",
    "saved_csv_sha256": hashlib.sha256(_csv.read_bytes()).hexdigest(),
    "limitations": ["Synthetic fixed rule-family benchmark only; does not establish universal language or coding ability.",
                    "Q is withheld from the model but the dataset generator is a known fixed benchmark.",
                    "Reported graph metric is top-K exact edge selection, evaluated on the fixed 1200-example test set."],
}
_json.write_text(json.dumps(_summary, indent=2) + "\n", encoding="utf-8")
print(f"Saved: {_csv.resolve()}")
print(f"Saved: {_json.resolve()}")
