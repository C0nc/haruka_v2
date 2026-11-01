"""Parallel MatrixMarket parser: pigz-decompress into RAM, split at newlines, parse chunks in 8 forked workers
with pandas C parser, assemble cells x genes CSR. Run: python -I mtx_to_npz_par.py <mtx.gz> <features.tsv.gz> <out.npz>
"""
import sys, subprocess, io
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import get_context
mtx, feat, out = sys.argv[1:4]
genes = pd.read_csv(feat, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
RAW = subprocess.run(["pigz", "-dc", "-p", "8", mtx], stdout=subprocess.PIPE, check=True).stdout
pos = 0
while RAW[pos:pos + 1] == b"%":
    pos = RAW.index(b"\n", pos) + 1
end = RAW.index(b"\n", pos)
nr, nc, nnz = map(int, RAW[pos:end].split()); pos = end + 1
NCH = 256
bounds = [pos]
step = (len(RAW) - pos) // NCH
for k in range(1, NCH):
    bounds.append(RAW.index(b"\n", pos + k * step) + 1)
bounds.append(len(RAW))

def parse(k):
    a = np.loadtxt  # unused; keep pandas
    df = pd.read_csv(io.BytesIO(RAW[bounds[k]:bounds[k + 1]]), sep=" ", header=None, engine="c",
                     dtype={0: np.int16, 1: np.int32, 2: np.float32})
    return df[1].to_numpy() - 1, df[0].to_numpy().astype(np.int32) - 1, df[2].to_numpy()

print("bounds ready", flush=True)
with get_context("fork").Pool(8) as pool:
    parts = pool.map(parse, range(NCH))
print("parsed", flush=True)
del RAW
cell = np.concatenate([p[0] for p in parts]); gene = np.concatenate([p[1] for p in parts]); v = np.concatenate([p[2] for p in parts])
del parts
assert len(cell) == nnz, (len(cell), nnz)
# file is sorted by gene (row of the mtx): build genes x cells CSR without sorting, then transpose linearly
assert np.all(np.diff(gene) >= 0)
indptr = np.concatenate([[0], np.cumsum(np.bincount(gene, minlength=nr))]).astype(np.int64)
Xg = sp.csr_matrix((v, cell.astype(np.int32), indptr), shape=(nr, nc))
del cell, gene, v
print("gene-major built", flush=True)
X = Xg.T.tocsr()
del Xg
print("transposed", flush=True)
sp.save_npz(out, X, compressed=False)
pd.Series(genes).to_csv(out + ".genes.tsv", sep="\t", index=False, header=False)
print("done", X.shape, X.nnz, flush=True)
