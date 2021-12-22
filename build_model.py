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

# cancer = sys.argv[1]
cancer = 'MESO'
# cancer = 'HNSC2'

# root_dir = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/{cancer}')
root_dir = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/{cancer}')
scripts = Path(f'/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts')

Z = np.loadtxt(root_dir / 'traits.txt', delimiter='\t')
Y = np.loadtxt(root_dir / 'expression.txt', delimiter='\t')
X = np.loadtxt(root_dir / 'genotype.txt', delimiter='\t')


lambdaLambda_z = 0.8
lambdaTheta_yz = 0.02

lambdaLambda_y = 0.7 
lambdaTheta_xy = 0.02


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


traits = pd.read_pickle(root_dir / 'traits_data.pkl')
mutations = pd.read_pickle(root_dir / 'mutations_data.pkl')
rnaseq = pd.read_pickle(root_dir / 'rnaseq_data.pkl')

traits = pd.read_csv(root_dir / 'traits_data.csv').set_index('Unnamed: 0')
mutations = pd.read_csv(root_dir / 'mutations_data.csv').set_index('Unnamed: 0')
rnaseq = pd.read_csv(root_dir / 'rnaseq_data.csv').set_index('Unnamed: 0')

G = nx.from_scipy_sparse_matrix(estLambda_y)
G = nx.relabel_nodes(G, dict(zip(range(0, len(rnaseq)), rnaseq.index)))
neighbor_count = [len(list(nx.neighbors(G, gene))) for gene in rnaseq.index]
neighbors = pd.DataFrame(neighbor_count, index=rnaseq.index, columns = ['neighbors'])
G.remove_nodes_from([i for i in neighbors[neighbors.neighbors < 2].index])

print(f'\nGene expression network has {len(G.nodes())} nodes\n')

from scipy.sparse.linalg import inv
import warnings
from scipy import sparse
import matplotlib.pyplot as plt
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
Black = '#000000'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet, Black]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.style.use('ggplot')

Sigma_z = inv(estLambda_z)
Sigma_y = sparse.csr_matrix(np.linalg.inv(estLambda_y.tocsr().todense()))
B_xy = -estTheta_xy * Sigma_y # Indirect mutation perturbation effects on gene expression levels
B_yz = -estTheta_yz * Sigma_z # Indirect effects of gene expression levels on clinical phenotypes
B_xz = B_xy * B_yz # mutation effects on clinical phenotypes


impact = pd.DataFrame(B_xz.todense(), columns=traits.columns, index=mutations.columns)
impact['overall'] = impact.sum(1)
impact['frequency'] = impact.index.map(lambda x: mutations[x].sum())

impact.sort_values(by='Survival', ascending=True)[:50][::-1].drop(['overall', 'frequency'], axis=1).plot(kind='barh', stacked=True, figsize=(13, 9))
plt.grid(False)
plt.tight_layout()
plt.savefig('top_mutations.png', dpi = 120)
plt.show()


shutil.copy(scripts / f'{cancer}.log', root_dir)
os.system('zip meso.zip MESO.log traits_data.csv mutations_data.csv rnaseq_data.csv Lambda_z.npz Theta_yz.npz Lambda_y.npz Theta_xy.npz traits_data.pkl mutations_data.pkl rnaseq_data.pkl params.json')