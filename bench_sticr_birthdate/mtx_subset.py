"""Stream a gene-sorted (genes x cells) .mtx.gz, keep only a subset of cells, parse in 8 workers.
Output: cells(subset, in given order) x genes CSR .npz + .genes.tsv + .cells.npy
Run: python -I mtx_subset.py <mtx.gz> <features.tsv.gz> <subset_cellids.npy (0-based column ids)> <out.npz>
"""
import sys, subprocess, io
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import get_context

mtx, feat, sub_p, out = sys.argv[1:5]
genes = pd.read_csv(feat, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
sub = np.sort(np.load(sub_p).astype(np.int64))
proc = subprocess.Popen(["pigz", "-dc", "-p", "4", mtx], stdout=subprocess.PIPE, bufsize=1 << 24)
while True:
    l = proc.stdout.readline()
    if not l.startswith(b"%"):
        break
nr, nc, nnz = map(int, l.split())
LUT = -np.ones(nc, dtype=np.int64)
LUT[sub] = np.arange(len(sub))


def parse(buf):
    df = pd.read_csv(io.BytesIO(buf), sep=" ", header=None, engine="c",
                     dtype={0: np.int32, 1: np.int64, 2: np.float32})
    cell = LUT[df[1].to_numpy() - 1]
    m = cell >= 0
    return len(df), cell[m].astype(np.int32), (df[0].to_numpy()[m] - 1).astype(np.int16), df[2].to_numpy()[m]


def chunks(sz=1 << 28):
    rest = b""
    while True:
        b = proc.stdout.read(sz)
        if not b:
            if rest:
                yield rest
            return
        b = rest + b
        k = b.rfind(b"\n") + 1
        rest = b[k:]
        yield b[:k]


cells, gs, vs = [], [], []
seen = 0
with get_context("fork").Pool(8) as pool:
    for n, c, g, v in pool.imap(parse, chunks(), chunksize=1):
        seen += n; cells.append(c); gs.append(g); vs.append(v)
        if len(cells) % 10 == 0:
            print("lines", seen, flush=True)
assert seen == nnz, (seen, nnz)
cell = np.concatenate(cells); gene = np.concatenate(gs).astype(np.int64); val = np.concatenate(vs)
assert np.all(np.diff(gene) >= 0)
indptr = np.concatenate([[0], np.cumsum(np.bincount(gene, minlength=nr))])
X = sp.csr_matrix((val, cell, indptr), shape=(nr, len(sub))).T.tocsr()
sp.save_npz(out, X, compressed=False)
np.save(out + ".cells.npy", sub)
pd.Series(genes).to_csv(out + ".genes.tsv", sep="\t", index=False, header=False)
print("done", X.shape, X.nnz, flush=True)
