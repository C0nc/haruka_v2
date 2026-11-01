#!/usr/bin/env python3
"""Compare Hotspot gene programmes across three graphs: real space, z-CC, s-CC."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from scipy.stats import spearmanr

D = Path('/home/yancui/haruka_v2/bench_kras_xenium/hotspot_arms')
OUT = Path('/home/yancui/haruka_v2/bench_kras_xenium/figures')
SPACE, ZCC, SCC = '#0f6e7a', '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8a86', '#e3e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
  'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,
  'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
COL = {'space':SPACE,'z-CC':ZCC,'s-CC':SCC}
ARMS = ['space','z-CC','s-CC']

A = pd.read_csv(D/'autocorrelation.csv.gz')
S = json.loads((D/'summary.json').read_text())
M = pd.read_csv(D/'modules.csv') if (D/'modules.csv').exists() else None
secs = sorted(A.section.unique())

fig = plt.figure(figsize=(13.2,8.2))
gs = gridspec.GridSpec(2,3,wspace=0.3,hspace=0.62,left=0.06,right=0.985,top=0.9,bottom=0.12)

ax=fig.add_subplot(gs[0,0]); w=.26; xs=np.arange(len(secs))
for i,arm in enumerate(ARMS):
    v=[ (A[(A.section==s)&(A.arm==arm)].FDR<0.05).sum() for s in secs]
    ax.bar(xs+(i-1)*w,v,w,color=COL[arm],label=arm)
    for j,q in enumerate(v): ax.text(j+(i-1)*w,q+30,f'{q}',ha='center',fontsize=6.4)
ax.set_xticks(xs); ax.set_xticklabels([s.replace('JB_','') for s in secs])
ax.set_ylabel('genes, FDR < 0.05'); ax.set_title('A  How many genes look spatially structured',
    loc='left',fontsize=9.5,color=INK)
ax.legend(frameon=False,fontsize=7.4,ncol=3,loc='lower left',bbox_to_anchor=(0,-0.36))
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[0,1]); pairs=[('z-CC','space'),('s-CC','space'),('z-CC','s-CC')]
cols=[ZCC,SCC,'#7a6ea8']; w=.26
for i,(p1,p2) in enumerate(pairs):
    v=[]
    for s in secs:
        t=A[A.section==s].pivot_table(index='gene',columns='arm',values='Z').dropna()
        v.append(spearmanr(t[p1],t[p2]).statistic)
    ax.bar(xs+(i-1)*w,v,w,color=cols[i],label=f'{p1} vs {p2}')
    for j,q in enumerate(v): ax.text(j+(i-1)*w,q+.015,f'{q:.2f}',ha='center',fontsize=6.4)
ax.set_xticks(xs); ax.set_xticklabels([s.replace('JB_','') for s in secs]); ax.set_ylim(0,1.05)
ax.set_ylabel('ρ of per-gene autocorrelation Z')
ax.set_title('B  Do the arms agree on WHICH genes?',loc='left',fontsize=9.5,color=INK)
ax.legend(frameon=False,fontsize=7,ncol=3,loc='lower left',bbox_to_anchor=(0,-0.36))
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[0,2]); w=.26
for i,arm in enumerate(ARMS):
    v=[]
    for s in secs:
        t=A[A.section==s].pivot_table(index='gene',columns='arm',values='Z').dropna()
        top={k:set(t[k].nlargest(300).index) for k in ARMS}
        ref=top['space']; v.append(len(top[arm]&ref)/len(top[arm]|ref))
    ax.bar(xs+(i-1)*w,v,w,color=COL[arm])
    for j,q in enumerate(v): ax.text(j+(i-1)*w,q+.012,f'{q:.2f}',ha='center',fontsize=6.4)
ax.set_xticks(xs); ax.set_xticklabels([s.replace('JB_','') for s in secs]); ax.set_ylim(0,1.08)
ax.set_ylabel('Jaccard vs real-space top 300')
ax.set_title('C  Overlap of the top programmes',loc='left',fontsize=9.5,color=INK)
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,0])
rows=[]
for k,v in S.items():
    sec,arm=k.split('|')
    for m,d in (v.get('modules') or {}).items():
        rows.append(dict(section=sec,arm=arm,module=m,n=d['n_genes'],
                         rho_tumor=d['rho_local_tumor'],rho_mapk=d['rho_MAPK']))
R=pd.DataFrame(rows)
if len(R):
    for arm in ARMS:
        r=R[R.arm==arm]
        ax.scatter(np.abs(r.rho_tumor),np.abs(r.rho_mapk),s=np.clip(r.n,15,160),
                   color=COL[arm],alpha=.72,linewidths=.6,edgecolors='white',label=arm)
    ax.axhline(0.1,color=MUTED,ls='--',lw=1); ax.text(0.015,0.108,'|ρ| = 0.1',fontsize=7,color=MUTED)
ax.set_xlabel('|ρ| module score × local tumour fraction')
ax.set_ylabel('|ρ| module score × MAPK output')
ax.set_title('D  What each module tracks',loc='left',fontsize=9.5,color=INK)
ax.legend(frameon=False,fontsize=7.4,ncol=3,loc='lower left',bbox_to_anchor=(0,-0.36))
ax.grid(color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,1]); w=.26
for i,arm in enumerate(ARMS):
    v=[S.get(f'{s}|{arm}',{}).get('n_modules',0) for s in secs]
    ax.bar(xs+(i-1)*w,v,w,color=COL[arm])
    for j,q in enumerate(v): ax.text(j+(i-1)*w,q+.12,f'{q}',ha='center',fontsize=6.8)
ax.set_xticks(xs); ax.set_xticklabels([s.replace('JB_','') for s in secs])
ax.set_ylabel('modules recovered'); ax.set_title('E  Module count',loc='left',fontsize=9.5,color=INK)
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,2]); ax.axis('off')
lines=['Largest module per arm (section JB_M1)','']
for arm in ARMS:
    d=(S.get(f'JB_M1|{arm}',{}) or {}).get('modules') or {}
    if not d: lines += [f'{arm}: none','']; continue
    big=max(d.items(),key=lambda kv: kv[1]['n_genes'])
    lines.append(f'{arm}  ({big[1]["n_genes"]} genes)')
    lines.append('   '+', '.join(big[1]['top'][:8]))
    lines.append(f'   ρ tumour {big[1]["rho_local_tumor"]:+.2f}   ρ MAPK {big[1]["rho_MAPK"]:+.2f}')
    lines.append('')
ax.text(0,1,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=8,color=INK,
        linespacing=1.5,family='DejaVu Sans')
fig.suptitle('Hotspot gene programmes: real space vs background vs salient embedding',
             x=0.06,ha='left',fontsize=12,color=INK,fontweight='bold')
p=OUT/'hotspot_arms.png'
fig.savefig(p,dpi=210,bbox_inches='tight'); fig.savefig(str(p).replace('.png','.pdf'),bbox_inches='tight')
print('wrote',p)
