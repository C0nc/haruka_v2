#!/usr/bin/env python3
"""CODEX AD story figure: batch mechanism, anatomy in the salient latent, plaque deflation."""
from __future__ import annotations
import numpy as np, pandas as pd, anndata as ad, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score as nmi
from scipy.stats import spearmanr

SEED=0; BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
INK,INK2,MUTED,GRID='#0b0b0b','#52514e','#8a8a86','#e3e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
    'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
P='/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/'
R='/home/yancui/haruka_v2/results_codex_channelnorm_ab/'
CC='haruka_z_s_cellcharter_cc3_condition_only/'
ARMS={'baseline':'codex_cns_myeloid_primary_no_aml_no_combat',
      'channelnorm':'codex_cns_myeloid_primary_no_aml_channelnorm'}

def load(stem, h5):
    a=ad.read_h5ad(P+h5+'.h5ad'); src=np.load(R+stem+'/'+CC+'source_rows.npy'); o=a.obs.iloc[src]
    X={k:np.load(R+stem+'/'+CC+v) for k,v in
       [('z','background.npy'),('s','salient.npy'),('zcc','spatial.npy'),('scc','salient_spatial.npy')]}
    return o, X, a.obsm['spatial'][src].astype(float)

def knn_ex(x,lab):
    xs=StandardScaler().fit_transform(x)
    nb=NearestNeighbors(n_neighbors=31).fit(xs).kneighbors(xs,return_distance=False)[:,1:]
    e=float((pd.Series(lab).value_counts(normalize=True)**2).sum())
    return ((lab[nb]==lab[:,None]).mean()-e)/(1-e)
def km(x,lab):
    return nmi(lab,MiniBatchKMeans(n_clusters=len(np.unique(lab)),n_init=20,batch_size=4096,
        random_state=SEED).fit_predict(StandardScaler().fit_transform(x)))
def pc1(x): return PCA(n_components=1,random_state=SEED).fit_transform(StandardScaler().fit_transform(x))[:,0]
def agg(x,xy,donor,k=20,hops=3,random=False,rng=None):
    out,cur=[],x.copy()
    for _ in range(hops):
        nx=np.zeros_like(x)
        for d in np.unique(donor):
            i=np.where(donor==d)[0]
            nb=(rng.integers(0,len(i),size=(len(i),k)) if random else
                NearestNeighbors(n_neighbors=k+1).fit(xy[i]).kneighbors(xy[i],return_distance=False)[:,1:])
            nx[i]=cur[i][nb].mean(1)
        out.append(nx); cur=nx
    return np.hstack(out)
def cw(x,lab):
    o=x.copy()
    for g in np.unique(lab): m=lab==g; o[m]-=o[m].mean(0)
    return o

D={}
for arm,stem in ARMS.items():
    o,X,xy=load(stem,stem); donor=o['donor_id'].astype(str).to_numpy(); reg=o['region'].astype(str).to_numpy()
    D[arm]=dict(o=o,X=X,xy=xy,donor=donor,reg=reg)
plq=pd.read_csv('/home/yancui/haruka_v2/final_results_nb/codex_ad/biology/plaque_distance/'
                'codex_plaque_distance_cells.csv.gz'); plq['cell_id']=plq['cell_id'].astype(str)

fig=plt.figure(figsize=(12.8,7.6))
gs=gridspec.GridSpec(2,3,wspace=0.34,hspace=0.62,left=0.06,right=0.985,top=0.92,bottom=0.12)

ax=fig.add_subplot(gs[0,0]); views=['z','s','zcc','scc']; lbl=['z','s','z-CC','s-CC']
w=0.38; xs=np.arange(4)
b=[knn_ex(D['baseline']['X'][v],D['baseline']['donor']) for v in views]
c=[knn_ex(D['channelnorm']['X'][v],D['channelnorm']['donor']) for v in views]
ax.bar(xs-w/2,b,w,color=ORANGE); ax.bar(xs+w/2,c,w,color=BLUE)
for i,(u,v) in enumerate(zip(b,c)):
    ax.text(i-w/2,u+.02,f'{u:.2f}',ha='center',fontsize=7); ax.text(i+w/2,v+.02,f'{v:.2f}',ha='center',fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(lbl); ax.set_ylim(0,1.15); ax.set_ylabel('same-donor kNN excess')
ax.set_title('A  Batch: the honest metric',loc='left',fontsize=9.5,color=INK)
ax.legend(handles=[Patch(facecolor=ORANGE,label='baseline norm'),Patch(facecolor=BLUE,label='channel norm')],
          frameon=False,fontsize=7.5,loc='lower left',bbox_to_anchor=(0,-0.42),ncol=2)
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[0,1]); d=D['channelnorm']; rng=np.random.default_rng(SEED); z=d['X']['z']
vals=[km(z,d['donor']),km(agg(z,d['xy'],d['donor']),d['donor']),
      km(agg(cw(z,d['donor']),d['xy'],d['donor']),d['donor']),
      km(agg(z,d['xy'],d['donor'],random=True,rng=rng),d['donor'])]
names=['z intrinsic','+ real graph','+ donor-centred','+ RANDOM graph']
ax.barh(range(4),vals,color=[BLUE,ORANGE,BLUE,ORANGE],height=.6)
for i,v in enumerate(vals): ax.text(v+.02,i,f'{v:.3f}',va='center',fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(names,fontsize=8); ax.set_xlim(0,1.18)
ax.set_xlabel('donor NMI'); ax.set_title('B  Aggregation unmasks a hidden offset',loc='left',fontsize=9.5,color=INK)
ax.grid(axis='x',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[0,2]); d=D['channelnorm']; don=np.unique(d['donor'])
for tag,key,col in [('z-CC','zcc',BLUE),('s-CC','scc',ORANGE)]:
    p=pc1(d['X'][key])
    v=[max(a_,1-a_) for a_ in [roc_auc_score((d['reg'][d['donor']==dd]=='WM').astype(int),p[d['donor']==dd]) for dd in don]]
    ax.plot(range(4),v,'-o',color=col,lw=2,ms=6,label=tag)
    for i,q in enumerate(v): ax.text(i,q+.02,f'{q:.2f}',ha='center',fontsize=7,color=INK)
ax.axhline(.5,color=MUTED,ls='--',lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(don)
ax.set_ylim(.4,1.05); ax.set_ylabel('grey/white AUROC (PC1)')
ax.set_title('C  Anatomy sits in the SALIENT latent',loc='left',fontsize=9.5,color=INK)
ax.legend(frameon=False,fontsize=7.6,loc='lower right'); ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,0]); d=D['channelnorm']
dist=pd.Series(d['o'].index.astype(str)).map(plq.set_index('cell_id')['distance_to_dense_abeta_centroid_um']).to_numpy()
p=pc1(d['X']['scc']); pool,gm,wm=[],[],[]
for dd in don:
    m=d['donor']==dd
    pool.append(abs(spearmanr(p[m],dist[m]).statistic))
    gm.append(abs(spearmanr(p[m&(d['reg']=='GM')],dist[m&(d['reg']=='GM')]).statistic))
    wm.append(abs(spearmanr(p[m&(d['reg']=='WM')],dist[m&(d['reg']=='WM')]).statistic))
xs=np.arange(4); w=.26
for off,v,c,l in [(-w,pool,ORANGE,'pooled'),(0,gm,BLUE,'grey matter only'),(w,wm,AQUA,'white matter only')]:
    ax.bar(xs+off,v,w,color=c,label=l)
    for i,q in enumerate(v): ax.text(i+off,q+.01,f'{q:.2f}',ha='center',fontsize=6.6)
ax.set_xticks(xs); ax.set_xticklabels(don); ax.set_ylabel('|ρ| s-CC PC1 × Aβ distance'); ax.set_ylim(0,.82)
ax.set_title('D  The plaque signal is largely anatomy',loc='left',fontsize=9.5,color=INK)
ax.legend(frameon=False,fontsize=7.3,ncol=3,loc='lower left',bbox_to_anchor=(0,-0.42))
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,1])
res={}
for arm in ARMS:
    d=D[arm]; y=d['o']['plaque_within_30um'].to_numpy().astype(int); ok=y>=0
    for tag,key in [('z-CC','zcc'),('s-CC','scc')]:
        sc=[]
        for dd in np.unique(d['donor']):
            tr=ok&(d['donor']!=dd); te=ok&(d['donor']==dd)
            if len(np.unique(y[te]))<2: continue
            ss=StandardScaler().fit(d['X'][key][tr])
            clf=LogisticRegression(max_iter=3000,random_state=SEED,class_weight='balanced').fit(ss.transform(d['X'][key][tr]),y[tr])
            sc.append(roc_auc_score(y[te],clf.predict_proba(ss.transform(d['X'][key][te]))[:,1]))
        res[(arm,tag)]=np.mean(sc)
xs=np.arange(2); w=.38
bb=[res[('baseline','z-CC')],res[('baseline','s-CC')]]; cc=[res[('channelnorm','z-CC')],res[('channelnorm','s-CC')]]
ax.bar(xs-w/2,bb,w,color=ORANGE); ax.bar(xs+w/2,cc,w,color=BLUE)
for i,(u,v) in enumerate(zip(bb,cc)):
    ax.text(i-w/2,u+.004,f'{u:.3f}',ha='center',fontsize=7); ax.text(i+w/2,v+.004,f'{v:.3f}',ha='center',fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(['z-CC','s-CC']); ax.set_ylim(.6,.85)
ax.set_ylabel('plaque ≤30 µm AUROC'); ax.set_title('E  Normalization improved the readout',loc='left',fontsize=9.5,color=INK)
ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,2]); ax.axis('off')
txt=('What the CODEX audit established\n\n'
     '1  The intrinsic latent was never batch-free: donor η² = 0.075\n'
     '    hidden under 12× more cell-level variance.\n\n'
     '2  Aggregation is a low-pass filter. It keeps the donor mean\n'
     '    exactly (1.00×) and cuts within-donor variance to 0.37×,\n'
     '    so the offset surfaces. A RANDOM graph gives NMI 1.000.\n\n'
     '3  Channel normalization fixed the input, not the mechanism:\n'
     '    batch down, plaque AUROC up, but s-CC PC1 is STILL a\n'
     '    grey/white-matter axis in every donor.\n\n'
     '4  So the anatomy-in-salient behaviour is architectural.\n'
     '    The plaque claim must be reported within region.')
ax.text(0,1,txt,transform=ax.transAxes,va='top',fontsize=8.3,color=INK,linespacing=1.55)
fig.suptitle('CODEX Alzheimer: what the salient spatial embedding encodes',x=0.06,ha='left',
             fontsize=12,color=INK,fontweight='bold')
out='/home/yancui/haruka_v2/bench_codex_cns_2026/codex_story.png'
fig.savefig(out,dpi=210,bbox_inches='tight'); fig.savefig(out.replace('.png','.pdf'),bbox_inches='tight')
print('wrote',out)
