#!/usr/bin/env python3
"""Replication test for the salient-mean mask.

A prior agent reported mode fraction 0/16 at mask weight 1e3 and 15/16 at 1e4, then was
stopped before verifying and persisted no data. A 0->15 jump across a single decade is
sharp enough to be a bug rather than an effect, so this re-runs it on SEEDS NOT USED THEN
(100+) and maps the weight response, to separate a real threshold from a fluke.
"""
import argparse, json, os, sys

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Haruka")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scvi.external.contrastivevi._module import ContrastiveVAE
import run_salient_eval as ev

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="1e3,3e3,1e4,3e4")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed0", type=int, default=100)
    p.add_argument("--out", default="outputs/eval_mask/results.jsonl")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    ev.VARIANTS["C_mask"] = dict(n_spatial_latent=16, spatial_attach="both",
                                 spatial_invariance_penalty=0.5,
                                 mask_salient_mean_on_background=True)
    rows = []
    with open(a.out, "a") as fh:
        for w in [float(x) for x in a.weights.split(",")]:
            ContrastiveVAE._SALIENT_MEAN_MASK_WEIGHT = w
            ok = 0
            for s in range(a.seed0, a.seed0 + a.seeds):
                try:
                    r = ev.run_one("C_mask", s)
                except Exception as e:
                    print(f"  w={w:g} seed={s} FAILED {type(e).__name__}: {e}", flush=True); continue
                r["mask_weight"] = w
                clean = r["bg_ari_layer"] > 0.8 and r["bg_ari_lesion"] < 0.15
                ok += clean
                rows.append(r); fh.write(json.dumps(r) + "\n"); fh.flush()
                print(f"  w={w:g} seed={s}  bg_layer {r['bg_ari_layer']:.3f}  "
                      f"bg_lesion {r['bg_ari_lesion']:.3f}  niche {r.get('sp_probe_niche', float('nan')):.3f}"
                      f"  {'CLEAN' if clean else ''}", flush=True)
            print(f"WEIGHT {w:g}: mode fraction {ok}/{a.seeds}\n", flush=True)

if __name__ == "__main__":
    main()
