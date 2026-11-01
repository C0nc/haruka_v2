"""Load every lineage segmentation (IBA1/GFAP/NeuN/Olig2) with the harmonised panel."""
import json, ast
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks')
DONORS = {'2997': 'AD-2', '3155': 'AD-1', '3196': 'AD-3', '3280': 'AD-4', '1796': 'Ctl-3',
          '3628': 'Ctl-2', '3026': 'Ctl-1', '1873': 'Ctl-4'}
LINEAGE = {'iba': 'Myeloid', 'gfap': 'Astrocyte', 'neun': 'Neuron', 'olig2': 'Oligodendrocyte'}
RENAME = {'Claudin-5': 'Claudin5', 'GalMac3': 'Gal3', 'PSD-95': 'PSD95', 'a-Beta': 'A-Beta'}
SHARED = ['AQP4', 'ATG7', 'ApoE', 'CD11b', 'CD11c', 'CD14', 'CD163', 'CD31', 'CD34', 'CD3e', 'CD4',
          'CD40', 'CD44', 'CD45', 'CD68', 'CD74', 'CD8', 'CollagenIV', 'GFAP', 'H2A.x', 'HLA-A', 'HLA-DR',
          'IBA1', 'MAP2', 'NeuN', 'PCNA', 'S100B', 'SMA', 'Synaptophysin', 'TMEM119', 'Vimentin',
          'Claudin5', 'Gal3', 'PSD95']                      # 34 model inputs
READOUT = ['A-Beta']                                          # pathology readout, NEVER a model input


def file_map():
    maps = {}
    for cell in json.loads((SRC / 'generate_metadata.ipynb').read_text())['cells']:
        if cell['cell_type'] != 'code': continue
        try: tree = ast.parse(''.join(cell['source']))
        except SyntaxError: continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ('iba_sample_num_dict', 'gfap_sample_num_dict',
                                                             'olig2_sample_num_dict', 'neun_sample_num_dict'):
                        maps[t.id.split('_')[0]] = ast.literal_eval(node.value)
    return maps


def load(exclude=('3628',)):
    frames = []
    for lin, files in file_map().items():
        for f, spec in files.items():
            if spec in exclude: continue                   # Ctl-2 = AML control, excluded as before
            d = pd.read_csv(SRC / 'marker_intensities' / f).rename(columns=RENAME)
            missing = [m for m in SHARED + READOUT if m not in d.columns]
            assert not missing, (f, missing)
            out = d[['cell', 'x', 'y', 'area µm²'] + SHARED + READOUT].copy()
            out.insert(0, 'lineage_mask', LINEAGE[lin]); out.insert(0, 'donor_id', DONORS[spec])
            out.insert(0, 'specimen', spec); out['source_file'] = f
            frames.append(out)
    return pd.concat(frames, ignore_index=True)
