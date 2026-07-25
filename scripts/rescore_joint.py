#!/usr/bin/env python3
"""Re-score an existing eval under the JOINT success criterion.

The previous headline metric scored only the background latent, which silently made
background stability the objective and ignored niche recovery -- the capability this work
set out to add. This re-scores the same runs under a criterion that keeps both in view,
and sweeps the niche threshold so the ranking's sensitivity to it is visible rather than
buried in a single arbitrary cut.
"""
import argparse, collections, json

import numpy as np


def load(path):
    by = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        by[r["variant"]].append(r)
    return by


def med_iqr(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return "n/a"
    return f"{np.median(v):.3f} [{np.percentile(v,25):.3f},{np.percentile(v,75):.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/eval_2x2/results.jsonl")
    p.add_argument("--taus", default="0.55,0.60,0.70,0.80")
    a = p.parse_args()
    by = load(a.results)
    taus = [float(t) for t in a.taus.split(",")]

    print("Per-latent medians [IQR]\n" + "=" * 92)
    print(f"{'variant':16s}{'bg×layer':>22s}{'bg×lesion':>22s}{'sal×lesion':>16s}{'sp×niche':>16s}")
    for v, rs in by.items():
        print(f"{v:16s}"
              f"{med_iqr([r['bg_ari_layer'] for r in rs]):>22s}"
              f"{med_iqr([r['bg_ari_lesion'] for r in rs]):>22s}"
              f"{med_iqr([r['sal_ari_lesion'] for r in rs]):>16s}"
              f"{med_iqr([r.get('sp_probe_niche') for r in rs]):>16s}")

    print("\nBackground-only criterion (the OLD objective)\n" + "=" * 92)
    for v, rs in by.items():
        ok = sum(r["bg_ari_layer"] > 0.8 and r["bg_ari_lesion"] < 0.15 for r in rs)
        print(f"  {v:16s} {ok:2d}/{len(rs)}")

    print("\nJOINT criterion: bg×layer>0.8 AND bg×lesion<0.15 AND sal×lesion>0.8 AND sp×niche>tau")
    print("(variants with no spatial latent cannot satisfy the niche clause -- marked n/a)")
    print("=" * 92)
    print(f"{'variant':16s}" + "".join(f"{'tau='+str(t):>12s}" for t in taus))
    for v, rs in by.items():
        cells = []
        for t in taus:
            if rs[0].get("sp_probe_niche") is None:
                cells.append("n/a")
                continue
            ok = sum(
                r["bg_ari_layer"] > 0.8 and r["bg_ari_lesion"] < 0.15
                and r["sal_ari_lesion"] > 0.8 and (r.get("sp_probe_niche") or 0) > t
                for r in rs
            )
            cells.append(f"{ok}/{len(rs)}")
        print(f"{v:16s}" + "".join(f"{c:>12s}" for c in cells))

    print("\nNiche clause ALONE (how often each variant clears tau on niche)\n" + "=" * 92)
    print(f"{'variant':16s}" + "".join(f"{'tau='+str(t):>12s}" for t in taus))
    for v, rs in by.items():
        if rs[0].get("sp_probe_niche") is None:
            print(f"{v:16s}" + "".join(f"{'n/a':>12s}" for _ in taus)); continue
        print(f"{v:16s}" + "".join(
            f"{sum((r.get('sp_probe_niche') or 0) > t for r in rs)}/{len(rs)}".rjust(12) for t in taus))


if __name__ == "__main__":
    main()
