#!/bin/python

#####
# PerturbNet model is based on https://github.com/Koushul/PerturbNet
#####

from pathlib import Path
import shutil
import sys
from sklearn.metrics import recall_score
from models import GraphicalModel


lambdaLambda_z = 0.0
lambdaTheta_yz = 0.04

lambdaLambda_y = 0.6
lambdaTheta_xy = 0.01


model = GraphicalModel(traits_csv='output/traits_data.csv', 
            mutations_csv='output/mutations_data.csv', 
            rnaseq_csv='output/rnaseq_data.csv')

model.fit(lambdaLambda_z, 
    lambdaTheta_yz,
    lambdaLambda_y, 
    lambdaTheta_xy, 
    threads= int(sys.argv[1]))

model.save(Path('models'))

shutil.copy('HNSC.log', 'models')

import warnings
warnings.filterwarnings('ignore')

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


predictions = [model.predict_from_mutations(patient=p) for p in model.patients]
score = recall_score(model.traits.hpv, predictions)

print(score)
