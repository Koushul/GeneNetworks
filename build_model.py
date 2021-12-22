#!/bin/python

import os
from pathlib import Path
import datetime
import math
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys
import shutil
import pandas as pd
import networkx as nx
import time

tic = time.perf_counter()

sys.path.append("../EM-sCGGM/")
from em_scggm import em_scggm


cancer = 'HNSC'
root_dir = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/{cancer}')
scripts = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts')

Z = np.loadtxt(root_dir / 'traits.txt', delimiter='\t')
Y = np.loadtxt(root_dir / 'expression.txt', delimiter='\t')
X = np.loadtxt(root_dir / 'genotype.txt', delimiter='\t')


lambdaLambda_z = 0
lambdaTheta_yz = 0.01

lambdaLambda_y = 0.65
lambdaTheta_xy = 0.0787


print(lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy)

(estLambda_z, estTheta_yz, estLambda_y, estTheta_xy, estStats) = em_scggm(
    Z, Y, X, 
    lambdaLambda_z, 
    lambdaTheta_yz,
    lambdaLambda_y, 
    lambdaTheta_xy, 
    max_em_iters = 5, 
    threads = int(sys.argv[1]))


ssp.save_npz(root_dir / 'Lambda_z.npz', estLambda_z)
ssp.save_npz(root_dir / 'Theta_yz.npz', estTheta_yz)
ssp.save_npz(root_dir / 'Lambda_y.npz', estLambda_y)
ssp.save_npz(root_dir / 'Theta_xy.npz', estTheta_xy)

toc = time.perf_counter()
t = (toc - tic)/60.0

with open(root_dir / 'params.json', 'w') as f:
    f.write(json.dumps({'lambdaLambda_z': lambdaLambda_z, 
                        'lambdaTheta_yz': lambdaTheta_yz, 
                        'lambdaLambda_y': lambdaLambda_y, 
                        'lambdaTheta_xy': lambdaTheta_xy,
                        'time_elapsed': round(t, 3),
                        'threads': int(sys.argv[1]),
                        'timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}))

os.chdir(root_dir)

# plt.figure()
# plt.spy(estTheta_xy)
# plt.title("Mutations-to-Expression")
# plt.savefig('theta_xy.png', dpi=300)

# plt.figure()
# plt.spy(estTheta_yz)
# plt.title("Expression-to-Traits")
# plt.savefig('theta_yz.png', dpi=300)

# plt.figure()
# plt.spy(estLambda_y)
# plt.title("Expression Network")
# plt.savefig('lambda_y.png', dpi=300)

# plt.figure()
# plt.spy(estLambda_z)
# plt.title("Traits Network")
# plt.savefig('lambda_z.png', dpi=300)


shutil.copy(scripts / f'{cancer}.log', root_dir)

# ph.push_note('CRC', f'EM-sCGGM completed for {cancer} in {round(t, 2)} minutes.')


from scipy.sparse.linalg import inv
import warnings
from scipy import sparse

warnings.filterwarnings('ignore')
outputdir = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/{cancer}')
scripts = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts')


Theta_xy = ssp.load_npz(outputdir / 'Theta_xy.npz') #Mutations-to-Expression
Lambda_y = ssp.load_npz(outputdir / 'Lambda_y.npz') #Expression Network
Lambda_z = ssp.load_npz(outputdir / 'Lambda_z.npz') #Traits Network
Theta_yz = ssp.load_npz(outputdir / 'Theta_yz.npz') #Expression-to-Traits



traits = pd.read_csv(outputdir / 'traits_data.csv').set_index('Unnamed: 0')
mutations = pd.read_csv(outputdir / 'mutations_data.csv').set_index('Unnamed: 0')
rnaseq = pd.read_csv(outputdir / 'rnaseq_data.csv').set_index('Unnamed: 0')
traits.index.name = None
mutations.index.name = None
rnaseq.index.name = None
mutations = mutations.T

special_genes = """
RELA
PBX1
SPI1
HIVEP1
MXI1
TBX19
DNMT1
SMARCC1
ZNF410
HMBOX1
TFEB
HINFP
NFYA
BPTF
CREB1
AR
STAT6
TERT
""".split()


df = pd.DataFrame(Theta_yz.tocsr().todense())
hpv_connected_genes = rnaseq.reset_index().loc[df[df[0] != 0][0].index]['index'].values

print(f'HPV connected genes: {len(hpv_connected_genes)}')

G = nx.from_scipy_sparse_matrix(Lambda_y)
G = nx.relabel_nodes(G, dict(zip(range(0, len(rnaseq)), rnaseq.index)))

# neighbor_count = [len(list(nx.neighbors(G, gene))) for gene in rnaseq.index]
# neighbors = pd.DataFrame(neighbor_count, index=rnaseq.index, columns = ['neighbors'])

# cull = []
# for i in neighbors[neighbors.neighbors < 2].index:
#     if i not in hpv_connected_genes:
#         cull.append(i)
# G.remove_nodes_from(cull)

G.remove_edges_from(nx.selfloop_edges(G))
G.remove_nodes_from(set(nx.isolates(G)))

print()
print([i for i in special_genes if i in G.nodes()])
print()

all_paths = []
for gene in [i for i in special_genes if i in G.nodes()]:
    for source in set(G.nodes()) & set(hpv_connected_genes):
        for path in nx.all_simple_paths(G, source=source, target=gene, cutoff=4):
            all_paths.append(path)

if len(all_paths) > 0:
    paths_df = pd.DataFrame(all_paths)
    print(pd.DataFrame([[paths_df[paths_df[i]==x].__len__() for i in range(4)] for x in [i for i in special_genes if i in G.nodes()]], 
                index=[i for i in special_genes if i in G.nodes()]))

print()
print(f'\nGene expression network has {len(G.nodes())} nodes\n')


Sigma_z = inv(Lambda_z)
Sigma_y = sparse.csr_matrix(np.linalg.inv(Lambda_y.tocsr().todense()))
B_xy = -Theta_xy * Sigma_y # Indirect mutation perturbation effects on gene expression levels
B_yz = -Theta_yz * Sigma_z # Indirect effects of gene expression levels on clinical phenotypes
B_xz = B_xy * B_yz # mutation effects on clinical phenotypes

if len(traits.columns) > 1:
    impact = pd.DataFrame(B_xz.todense(), columns=traits.columns, index=mutations.index)
else:
    impact = pd.DataFrame(B_xz, columns=traits.columns, index=mutations.index)
impact = impact.abs()
impact['overall'] = impact.sum(1)

# print(impact)

print(impact[impact.overall > 0].sort_values(by='overall', ascending=False)[:20].index)
check_impact = ['TP53', 'TTN', 'FAT1', 'NOTCH1', 'PIK3CA', 'CASP8', 'NSD1']

print()
Theta_df = pd.DataFrame(abs(Theta_xy.tocsr()).sum(1), mutations.index, columns=['weight'])
Theta_df['frequency'] = Theta_df.index.map(lambda x: mutations.loc[x].sum())
print(Theta_df.sort_values(by='weight', ascending=False)[:20].index)

print()
check_impact = [i in impact[impact.overall > 0].sort_values(by='overall', ascending=False)[:25].index for i in ['TP53', 'TTN', 'FAT1', 'NOTCH1', 'PIK3CA', 'CASP8', 'NSD1']]
check_theta = [i in Theta_df.sort_values(by='weight', ascending=False)[:10].index for i in ['TP53', 'TTN', 'FAT1', 'NOTCH1', 'PIK3CA', 'CASP8', 'NSD1']]
print(pd.DataFrame([check_impact, check_theta], columns = ['TP53', 'TTN', 'FAT1', 'NOTCH1', 'PIK3CA', 'CASP8', 'NSD1'], index=['in_impact', 'in_theta']).T)