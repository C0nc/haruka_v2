#!/usr/bin/env python3
"""Reconstruct segmentation-supported CODEX lineages; audit sample composition.

These labels are broad segmentation identities, not de novo molecular subtypes.
Missing segmentation modalities are never treated as absent biological cells.
"""
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from shapely.geometry import shape

ROOT = Path('/home/yancui/haruka_v2')
SOURCE = ROOT / 'bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks'
OUT = ROOT / 'final_results_nb/codex_ad/biology/broad_celltype_annotation'
LABELS = {'GFAP': 'Astrocyte', 'IBA1': 'Myeloid_microglia_macrophage',
          'Olig2': 'Oligodendrocyte_OPC', 'NeuN': 'Neuron'}
DONORS = {'2997': 'AD-2', '3155': 'AD-1', '3196': 'AD-3', '3280': 'AD-4',
          '1796': 'Ctl-3', '3628': 'Ctl-2', '3026': 'Ctl-1', '1873': 'Ctl-4'}
COLORS = {'Astrocyte': '#e78ac3', 'Myeloid_microglia_macrophage': '#66c2a5',
          'Oligodendrocyte_OPC': '#8da0cb', 'Neuron': '#fc8d62'}


def inside(geometry, xy):
    """Vectorized polygon inclusion, including holes and multipart regions."""
    if geometry.geom_type == 'MultiPolygon':
        return np.logical_or.reduce([inside(g, xy) for g in geometry.geoms])
    mask = MplPath(np.asarray(geometry.exterior.coords)).contains_points(xy)
    for ring in geometry.interiors:
        mask &= ~MplPath(np.asarray(ring.coords)).contains_points(xy)
    return mask


def plot_composition(table, path, title, categories):
    regions = ['All tissue', 'Grey Matter', 'White Matter']
    donors = sorted(table.donor_id.unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    for ax, region in zip(axes, regions):
        sub = table[table.region.eq(region)]
        mat = sub.pivot(index='donor_id', columns='cell_type', values='proportion').reindex(
            index=donors, columns=categories, fill_value=0).fillna(0)
        bottom = np.zeros(len(donors))
        for ct in categories:
            ax.bar(np.arange(len(donors)), mat[ct], bottom=bottom, color=COLORS[ct], label=ct)
            bottom += mat[ct].to_numpy()
        ax.set_xticks(np.arange(len(donors)), donors, rotation=40, ha='right')
        ax.set_title(region); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Proportion among included segmentation classes')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = SOURCE / 'generate_metadata.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    mappings = {}
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in [
                        'iba_sample_num_dict', 'gfap_sample_num_dict',
                        'olig2_sample_num_dict', 'neun_sample_num_dict']:
                        mappings[target.id] = ast.literal_eval(node.value)
    assert len(mappings) == 4, mappings.keys()
    frames, inventory = [], []
    for key, files in mappings.items():
        marker = {'iba': 'IBA1', 'gfap': 'GFAP', 'olig2': 'Olig2', 'neun': 'NeuN'}[key.split('_')[0]]
        for filename, donor in files.items():
            if donor == '3628':  # AML control; preserve previous scientific cohort.
                continue
            path = SOURCE / 'marker_intensities' / filename
            columns = pd.read_csv(path, nrows=0).columns
            marker_column = marker if marker in columns else {'Olig2': 'Olig-2'}.get(marker, marker)
            usecols = ['cell', 'x', 'y', 'area µm²', marker_column]
            frame = pd.read_csv(path, usecols=usecols)
            frame = frame.rename(columns={'cell': 'segmentation_cell_id', 'x': 'spatial_x',
                'y': 'spatial_y', 'area µm²': 'area_um2', marker_column: 'lineage_marker_intensity'})
            frame['cell_id'] = donor + '_' + marker + '_' + frame.segmentation_cell_id.astype(str)
            frame['specimen_id'] = donor; frame['donor_id'] = DONORS[donor]
            frame['condition'] = 'disease' if DONORS[donor].startswith('AD') else 'control'
            frame['cell_type'] = LABELS[marker]; frame['segmentation_marker'] = marker
            frame['annotation_source'] = 'original_marker_specific_segmentation'
            frame['source_file'] = filename
            frames.append(frame)
            inventory.append({'donor_id': DONORS[donor], 'specimen_id': donor,
                'cell_type': LABELS[marker], 'n_cells': len(frame), 'source_file': filename,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(DONORS[donor], marker, len(frame), flush=True)
    data = pd.concat(frames, ignore_index=True)
    assert data.cell_id.is_unique
    data['region'] = 'Unassigned'
    for donor, indexes in data.groupby('specimen_id').groups.items():
        features = json.loads((SOURCE / 'GM_WM_Annotations_QP' /
            f'{donor}_GM-WM Annotations.geojson').read_text())['features']
        xy = data.loc[indexes, ['spatial_x', 'spatial_y']].to_numpy()
        regions = np.full(len(indexes), 'Unassigned', dtype=object)
        for f in features:
            name = f['properties'].get('classification', {}).get('name', 'Unassigned')
            geom = shape(f['geometry'])
            if not geom.is_valid:
                geom = geom.buffer(0)
            mask = inside(geom, xy) & (regions == 'Unassigned')
            regions[mask] = name
        data.loc[indexes, 'region'] = regions
    data.to_csv(OUT / 'codex_broad_celltype_annotations.csv.gz', index=False)
    inventory = pd.DataFrame(inventory)
    coverage = inventory.pivot(index='donor_id', columns='cell_type', values='n_cells')
    coverage.to_csv(OUT / 'segmentation_coverage_missing_is_NA.csv')
    inventory.to_csv(OUT / 'source_inventory.csv', index=False)
    common = [LABELS[m] for m in ['GFAP', 'IBA1', 'NeuN']]
    complete_donors = coverage.dropna().index.tolist()
    tables = []
    for name, donors, types in [('common_three_lineages', coverage.index.tolist(), common),
            ('complete_four_lineages', complete_donors, list(LABELS.values()))]:
        sub = data[data.donor_id.isin(donors) & data.cell_type.isin(types)]
        by_region = sub.groupby(['donor_id', 'condition', 'region', 'cell_type']).size().rename('n_cells').reset_index()
        whole = sub.groupby(['donor_id', 'condition', 'cell_type']).size().rename('n_cells').reset_index()
        whole['region'] = 'All tissue'
        table = pd.concat([by_region, whole], ignore_index=True)
        table['proportion'] = table.n_cells / table.groupby(['donor_id', 'region']).n_cells.transform('sum')
        table['comparison'] = name
        table.to_csv(OUT / f'{name}_sample_region_proportions.csv', index=False)
        plot_composition(table, OUT / f'{name}_sample_region_stacked.png',
            f'CODEX broad-cell-type composition: {name.replace("_", " ")}\nSegmentation-based; Ctl-2 excluded', types)
        tables.append(table)
    # Standardize GM/WM sampling coverage with equal weights, not cell-count weights.
    all_tables = pd.concat(tables, ignore_index=True)
    balanced = all_tables[all_tables.region.isin(['Grey Matter', 'White Matter'])].groupby(
        ['comparison', 'donor_id', 'condition', 'cell_type']).proportion.mean().reset_index()
    balanced.to_csv(OUT / 'equal_GM_WM_weight_proportions.csv', index=False)
    (OUT / 'analysis_metadata.json').write_text(json.dumps({
        'seed': 0, 'method': 'original marker-specific segmentation identity; no argmax protein labeling',
        'n_cells': len(data), 'complete_four_lineage_donors': complete_donors,
        'excluded': ['Ctl-2 (AML control)'],
        'limitations': ['IBA1 does not distinguish microglia from macrophages.',
            'OLIG2 does not distinguish mature oligodendrocytes from OPCs.',
            'GFAP segmentation may miss GFAP-low astrocytes.',
            'Absent OLIG2 files are missing measurements, not zero proportions.',
            'Counts are segmentation instances; cross-mask overlap is not independently deduplicated.',
            'No new Haruka fit or astro/oligo CODEX DPT performed.',
            'Sample differences are descriptive; condition and donor are confounded.'],
        'source_notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest()
    }, indent=2))
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
