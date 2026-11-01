"""Stream a (genes x cells) MatrixMarket .mtx.gz into a cells x genes CSR .npz.
Optionally keep only genes listed in a features file (by name).
Run: python -I mtx_to_npz.py <mtx.gz> <features.tsv.gz> <out.npz> [keep_genes.tsv.gz]
"""
import sys, gzip, io
import numpy as np, pandas as pd, scipy.sparse as sp

mtx, feat, out = sys.argv[1:4]
keep = sys.argv[4] if len(sys.argv) > 4 else None
genes = pd.read_csv(feat, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
with gzip.open(mtx, "rt") as fh:
    line = fh.readline()
    while line.startswith("%"):
        line = fh.readline()
    nr, nc, nnz = map(int, line.split())
print("dims", nr, nc, nnz, flush=True)
assert nr == len(genes)
if keep:
    kg = pd.read_csv(keep, sep="\t", header=None).iloc[:, 0].astype(str)
    kidx = np.where(np.isin(genes, kg.to_numpy()))[0]
else:
    kidx = np.arange(nr)
remap = -np.ones(nr + 1, dtype=np.int64)
remap[kidx + 1] = np.arange(len(kidx))
# fast path: pigz -> plain text pipe -> pyarrow multithreaded CSV reader
import subprocess, pyarrow as pa, pyarrow.csv as pc
pa.set_cpu_count(8)
rows, cols, vals = [], [], []
proc = subprocess.Popen(["pigz", "-dc", "-p", "4", mtx], stdout=subprocess.PIPE, bufsize=1 << 24)
hdr = 0
while True:  # skip header lines
    l = proc.stdout.readline()
    if not l.startswith(b"%"):
        break  # dims line consumed
rd = pc.open_csv(proc.stdout, read_options=pc.ReadOptions(column_names=["r", "c", "v"], block_size=1 << 28,
                                                         use_threads=True),
                 parse_options=pc.ParseOptions(delimiter=" "),
                 convert_options=pc.ConvertOptions(column_types={"r": pa.int32(), "c": pa.int32(), "v": pa.float32()}))
seen = 0
for b in rd:
    gr = b.column(0).to_numpy(); cc = b.column(1).to_numpy(); vv = b.column(2).to_numpy()
    g = remap[gr]; m = g >= 0
    rows.append(cc[m] - 1); cols.append(g[m].astype(np.int32)); vals.append(vv[m])
    seen += len(gr)
    if len(rows) % 20 == 0:
        print("read", seen, flush=True)
assert seen == nnz, (seen, nnz)
r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
X = sp.csr_matrix((v, (r, c)), shape=(nc, len(kidx)))
sp.save_npz(out, X, compressed=False)
pd.Series(genes[kidx]).to_csv(out + ".genes.tsv", sep="\t", index=False, header=False)
print("done", X.shape, X.nnz)
