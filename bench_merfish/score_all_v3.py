"""Score every finished result dir for one dataset and print the comparison table.

Guards against the two failure modes that have already produced false "success"
in this benchmark:

  * a glob that matches nothing and reports an empty table as a pass -- here the
    count of scored dirs is printed and a dir missing any of the four slots is
    listed as SKIPPED with the reason, not silently dropped;
  * OpenBLAS aborting on this 256-core box ("tried to allocate too many memory
    regions"). The thread caps must be set BEFORE numpy is imported, so they are
    set here at the top of the module rather than in the shell.

    python score_all_v3.py --data data/simv3_d1_s0.h5ad
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402

import pandas as pd  # noqa: E402

from score_2x2 import score  # noqa: E402

SLOTS = ("background", "spatial", "salient", "salient_spatial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results_root", default="results_v3")
    ap.add_argument("--rep_mode", choices=["own", "aggregated"], default="own")
    ap.add_argument("--cluster_method", choices=["gmm", "leiden"], default="gmm")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on the result-dir name")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = os.path.basename(args.data).replace(".h5ad", "")
    root = os.path.join(args.results_root, tag)
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if args.only:
        dirs = [d for d in dirs if any(s in os.path.basename(d) for s in args.only)]
    if not dirs:
        raise SystemExit(f"no result dirs under {root} -- nothing to score")

    rows, skipped = [], []
    for d in dirs:
        name = os.path.basename(d)
        missing = [s for s in SLOTS if not os.path.exists(os.path.join(d, f"{s}.npy"))]
        if missing:
            skipped.append((name, f"missing {','.join(missing)}"))
            continue
        try:
            print(f"[score] {name}", flush=True)
            rows.append(score(args.data, d, name, 0, args.rep_mode,
                              args.cluster_method, 0.6))
        except Exception as e:                       # noqa: BLE001
            skipped.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print(f"\nscored {len(rows)} dirs, skipped {len(skipped)}")
    for n, why in skipped:
        print(f"  SKIPPED {n}: {why}")
    if not rows:
        raise SystemExit("nothing scored")

    df = pd.DataFrame(rows)
    out = args.out or os.path.join(root, f"metrics_{args.rep_mode}_{args.cluster_method}.csv")
    df.to_csv(out, index=False)

    cols = {
        "DEP__background__gt_arch": "bg>ctype",
        "DEP__spatial__gt_niche": "sp>bgDOM",
        "DEP__salient__gt_sal_intrinsic": "sal>state",
        "DEP__salient_spatial__gt_sal_domain": "salsp>salDOM",
        "diag_mean": "diag",
        "offdiag_mean": "LEAK",
        "oracle_offdiag": "floor",
        "excess_offdiag": "EXCESS",
        "assignment_margin": "margin",
    }
    have = [c for c in cols if c in df.columns]
    view = df[["method"] + have].rename(columns=cols)
    view = view.sort_values("margin", ascending=False)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print("\n" + view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
