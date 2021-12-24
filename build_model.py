#!/bin/python

#####
# PerturbNet model is based on https://github.com/Koushul/PerturbNet
#####

from pathlib import Path
import shutil
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from models import GraphicalModel
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

## lower values decreases sparsity i.e number of edges increases
lambdaLambda_z = 0.31 
lambdaTheta_yz = 0.015 
lambdaLambda_y = 0.48 
lambdaTheta_xy = 0.03 


model = GraphicalModel(traits_csv='output/traits_data.csv', 
            mutations_csv='output/mutations_data.csv', 
            rnaseq_csv='output/rnaseq_data.csv')

model.fit(lambdaLambda_z, 
    lambdaTheta_yz,
    lambdaLambda_y, 
    lambdaTheta_xy, 
    threads= int(sys.argv[1]),
    verbose = True)

print()
model.create_network()
print()
print(f"HPV Connected Genes: {len(model.hpv_connected_genes)}")

model.save(path='models')

shutil.copy('HNSC.log', f'models/{model.name}')


traits = model.traits.copy()
traits['predicted from expr'] = [model.predict_from_expression(patient=p) for p in model.patients]
traits['predicted from mutations'] = [model.predict_from_mutations(patient=p) for p in model.patients]

perf = {}
for metric, name in zip([accuracy_score, recall_score, precision_score, f1_score, roc_auc_score], ['Accuracy', 'Recall', 'Precision', 'F1_Score', 'AUC']):
    perf[name] = [metric(traits.hpv, traits['predicted from mutations']), metric(traits.hpv, traits['predicted from expr'])]

print()
print(pd.DataFrame(perf, index=['From Mutations', 'From HPV Genes']).T)
print()

df = pd.DataFrame(model.Theta_yz.tocsr().todense())
df.index = model.rnaseq.index
df = df.loc[set(model.hpv_connected_genes) & set(model.Network.nodes())]
df.sort_values(by=0).plot(kind='bar', figsize=(30, 12), legend=None)
plt.xlabel('Network Edge Weight')
plt.ylabel('Gene')
plt.savefig('hpv_genes.png', dpi=120)

print()
impact = pd.DataFrame(model.B_xz.todense(), columns=model.traits.columns, index=model.mutations.index)
impact = impact.abs()
print(impact[impact.hpv > 0].sort_values(by='hpv', ascending=False)[:20].index)
key_mutations = ['TP53', 'TTN', 'FAT1', 'NOTCH1', 'PIK3CA', 'CASP8', 'NSD1']

print()
Theta_df = pd.DataFrame(abs(model.Theta_xy.tocsr()).sum(1), model.mutations.index, columns=['weight'])
Theta_df['frequency'] = Theta_df.index.map(lambda x: model.mutations.loc[x].sum())
print(Theta_df.sort_values(by='weight', ascending=False)[:20].index)
print()
check_impact = [i in impact[impact.hpv > 0].sort_values(by='hpv', ascending=False)[:25].index for i in key_mutations]
check_theta = [i in Theta_df.sort_values(by='weight', ascending=False)[:10].index for i in key_mutations]
print(pd.DataFrame([check_impact, check_theta], columns = key_mutations, index=['in_impact', 'in_theta']).T)

