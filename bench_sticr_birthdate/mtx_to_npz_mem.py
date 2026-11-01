"""Like mtx_to_npz.py, but decompresses the whole .mtx.gz into RAM and parses with multithreaded pyarrow.
Run: python -I mtx_to_npz_mem.py <mtx.gz> <features.tsv.gz> <out.npz>
"""
import sys, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
import pyarrow as pa, pyarrow.csv as pc
pa.set_cpu_count(8)
mtx, feat, out = sys.argv[1:4]
genes = pd.read_csv(feat, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
raw = subprocess.run(["pigz", "-dc", "-p", "8", mtx], stdout=subprocess.PIPE, check=True).stdout
print("decompressed bytes", len(raw), flush=True)
pos = 0
while raw[pos:pos + 1] == b"%":
    pos = raw.index(b"\n", pos) + 1
end = raw.index(b"\n", pos)
nr, nc, nnz = map(int, raw[pos:end].split()); pos = end + 1
assert nr == len(genes)
tab = pc.read_csv(pa.BufferReader(memoryview(raw)[pos:]),
                  read_options=pc.ReadOptions(column_names=["r", "c", "v"], block_size=1 << 26, use_threads=True),
                  parse_options=pc.ParseOptions(delimiter=" "),
                  convert_options=pc.ConvertOptions(column_types={"r": pa.int16(), "c": pa.int32(), "v": pa.float32()}))
del raw
assert tab.num_rows == nnz, (tab.num_rows, nnz)
r = tab.column("c").to_numpy() - 1
c = tab.column("r").to_numpy().astype(np.int32) - 1
v = tab.column("v").to_numpy()
del tab
X = sp.csr_matrix((v, (r, c)), shape=(nc, nr))
sp.save_npz(out, X, compressed=False)
pd.Series(genes).to_csv(out + ".genes.tsv", sep="\t", index=False, header=False)
print("done", X.shape, X.nnz, flush=True)
