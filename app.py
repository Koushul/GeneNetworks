import scipy.sparse as ssp
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pyvis
import random
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import read_table
from pylab import rcParams
import seaborn as sns
from collections import Counter
from matplotlib.pyplot import figure
import itertools
from scipy.sparse.linalg import inv
from scipy import sparse
import streamlit.components.v1 as components


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import altair as alt

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.express as px

HNSC_path = Path('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/HNSC/')
outputdir = Path(HNSC_path / '2021-10-01-11:49:08.060841')

traits = pd.read_csv(outputdir / 'traits_data.csv')
traits.index = traits[traits.columns[0]]
traits = traits.drop(traits.columns[0], axis=1)
traits.index.name = None

rnaseq = pd.read_csv(outputdir / 'rnaseq_data.csv')
rnaseq.index = rnaseq[rnaseq.columns[0]]
rnaseq = rnaseq.drop(rnaseq.columns[0], axis=1)
rnaseq.index.name = None

mutations = pd.read_csv(outputdir / 'mutations_data.csv')
mutations.index = mutations[mutations.columns[0]]
mutations = mutations.drop(mutations.columns[0], axis=1)
mutations.index.name = None

# "Expression"
# rnaseq

# "Phenotype"
# traits

# "Genotype"
# mutations


Theta_xy = ssp.load_npz(outputdir / 'Theta_xy.npz') #Mutations-to-Expression
Lambda_y = ssp.load_npz(outputdir / 'Lambda_y.npz') #Expression Network
Lambda_z = ssp.load_npz(outputdir / 'Lambda_z.npz') #Traits Network
Theta_yz = ssp.load_npz(outputdir / 'Theta_yz.npz') #Expression-to-Traits


with open(HNSC_path / 'metis_parts.json') as f:
    metis_parts = json.load(f)

st.title("PerturbNet on TCGA Data")


Sigma_y = inv(Lambda_y)
Sigma_z = inv(Lambda_z)

B_xy = -Theta_xy * Sigma_y # Indirect mutation perturbation effects on gene expression levels
B_yz = -Theta_yz * Sigma_z # Indirect effects of gene expression levels on clinical phenotypes
B_xz = B_xy * B_yz # mutation effects on clinical phenotypes

Sigma_z_given_x = Sigma_z + Sigma_z * Theta_yz.getH() * Sigma_y * Theta_yz * Sigma_z

## Joint distribution of p(z,y|x)
Lambda_y_given_xz = Lambda_y + Theta_yz * Sigma_z * Theta_yz.getH()
Lambda_zy_given_x = np.array([[Lambda_z, Theta_yz.getH()], [Theta_yz, Lambda_y_given_xz]])
Theta_zy_given_x = np.array([sparse.csr_matrix(np.zeros((len(mutations), len(traits)))), Theta_xy])

#Posterior gene network after seeing phenotype data
Lambda_y_given_xz = Lambda_y + Theta_yz * Sigma_z * Theta_yz.getH()

def module_M_on_traits_direct(M):
    return sum(abs(Theta_yz.tocsr()[np.where(np.array(parts) == M)[0], :])).todense()

def module_M_on_traits_indirect(M):
    return sum(abs(B_yz.tocsr()[np.where(np.array(parts) == M)[0], :])).todense()

# The effects of mutation i on traits
def mutation_i_on_traits(i, trait_ixs):
    return abs(B_xz.tocsr()[i,trait_ixs])

# The effects of mutation i on traits, mediated by module M
def mutation_i_on_traits_mediated_by_M(mu, trait_ix, M):
    return sum(abs(B_xy.tocsr()[i, np.where(np.array(parts) == M)[0]] *
                   B_yz.tocsr()[np.where(np.array(parts) == M)[0], trait_ix])).todense()

mutations_traits = np.zeros((len(mutations.columns), len(traits.columns)))
for i in list(itertools.product(range(len(mutations.columns)), range(len(traits.columns)))):
    mutations_traits[i[0], i[1]] = mutation_i_on_traits(*i)
                        

num_modules = st.sidebar.slider("Number of gene modules:", 
    min_value=2, max_value=100, 
    value=20, 
    step=1)

parts = metis_parts[str(num_modules)]

modules = []
for i in range(len(Counter(parts).keys())):
    modules.append(np.where(np.array(parts)==i)[0])

module_effect = pd.DataFrame([ 
    [module_M_on_traits_direct(i).sum() for i in range(num_modules)],
    [module_M_on_traits_indirect(i).sum() for i in range(num_modules)],
    ], index = ['Direct Effects', 'Indirect Effects'], columns = range(num_modules)).T


impact = []
for i in range(len(mutations.columns)):
    impact.append(([mutation_i_on_traits(i, t) for t in range(len(traits.columns))]))
impact = pd.DataFrame(B_xz.todense(), columns=traits.columns, index=mutations.columns)
impact = impact.abs()
impact['overall'] = impact.sum(1)


heatmap = px.imshow(impact[impact.overall > 0.08].sort_index(),
        width = 600, height = 1000,
        labels = {'x': 'Traits', 'y': 'Genotype'},
        color_continuous_scale = 'Greens')



single = alt.selection_single()

line_plot = alt.Chart(module_effect.reset_index().melt('index')).mark_line(point=True).encode(
    x = alt.Y('index' + ':Q', title='Gene Module'),
    y = alt.Y('value' + ':Q', title='Overall Effect on Traits'),
    color = 'variable',
    tooltip = ['value']
).configure_axis(
    grid = False,
).configure_view(
    strokeOpacity=0
).properties(
    width=850,
    height=400
)

st.header('Gene Module Impact on Traits')
st.altair_chart(line_plot)

st.header('Genotype Impact on Traits')
st.plotly_chart(heatmap)


modules = []
for i in range(len(Counter(parts).keys())):
    modules.append(np.where(np.array(parts)==i)[0])

M= nx.Graph()
G = nx.from_scipy_sparse_matrix(Lambda_y)
for i in range(len(Counter(parts).keys())):
    M.add_node(i)

for m1, m2 in itertools.product(range(len(M.nodes)), range(len(M.nodes))):
    w = 0
    for a, b in itertools.product(modules[m1], modules[m2]):
        weight = G.get_edge_data(m1, m2)
        if weight:
            w += weight.get('weight')
    if w > 0:
        M.add_edge(m1, m2, weight=w)

for idx, trait in enumerate(traits.columns):
    M.add_node(trait)
    for node in [parts[i] for i in np.where(Theta_yz.todense()[:,idx] > 0)[0]]:
        M.add_edge(node, trait)
        
for idx, mutation in enumerate(mutations.columns):
    M.add_node(mutation)
    for node in [parts[i] for i in np.where(Theta_xy.todense()[:,idx] > 0)[0]]:
        M.add_edge(node, mutation)  
        
T = nx.from_scipy_sparse_matrix(Lambda_z)
T.remove_edges_from(nx.selfloop_edges(T))

for i in T.edges:
    M.add_edge(traits.columns[i[0]], traits.columns[i[1]])

M.remove_edges_from(nx.selfloop_edges(M))

from pyvis.network import Network


net = Network(width="900px", height="900px")
net.from_nx(M)

st.header('Network')
show_all = st.sidebar.checkbox("Show All Nodes")

if show_all:
    net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.01,
                       damping=0.95)

to_remove = []
has_edges = set([e['to'] for e in net.edges]) | set([e['from'] for e in net.edges])

for i in net.nodes:
    i['physics'] = show_all
    if i['id'] in traits.columns:
        i['y'] = random.randint(-400, -200)
        i['x'] = random.randint(-300, 300)
        i['color'] = 'green'

    elif i['id'] in mutations.columns:
        i['color'] = 'red'
        idx = list(mutations.columns).index(i['id'])      
        i['size'] = 200*sum([mutation_i_on_traits(idx, t) for t in range(len(traits.columns))])
        i['y'] = random.randint(200, 400)
        i['x'] = random.randint(-300, 300)

        if i['size'] < 18 and i['label'] not in has_edges:
            to_remove.append(i)
        
    else:
        i['x'] = random.randint(-300, 300)
#         i['y'] = random.randint(-400, 400)

if not show_all:
    for i in to_remove:
        net.nodes.remove(i)   

neighbor_map = net.get_adj_list()
for node in net.nodes:
    node['title'] = ' Neighbors:<br>' + '<br>'.join(map(str, neighbor_map[node['id']]))
    node['value'] = len(neighbor_map[node['id']])


net.save_graph('graph.html')

a_file = open('graph.html', "r")
lines = a_file.readlines()
a_file.close()

del lines[17]

new_file = open('graph2.html', "w+")

for line in lines:
    new_file.write(line)

new_file.close()


HtmlFile = open('graph2.html', 'r', encoding='utf-8')


components.html(HtmlFile.read(), height=2000, width = 2000)

