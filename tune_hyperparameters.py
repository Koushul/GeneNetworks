from optuna import create_study
from optuna.samplers import TPESampler
from models import GraphicalModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json

def objective(trial):

    params = {
        "lambdaLambda_z": trial.suggest_float("z", 0, 1, step=0.3), 
        "lambdaTheta_yz": trial.suggest_float("yz", 0, 0.1, step=0.01), 
        "lambdaLambda_y": trial.suggest_float("y", 0, 0.9, step=0.01), 
        "lambdaTheta_xy": trial.suggest_float("xy", 0, 0.01, step=0.01), 
        "max_em_iters" : 6, 
        "threads" : 46
    }

    model = GraphicalModel(traits_csv='output/traits_data.csv', 
                mutations_csv='output/mutations_data.csv', 
                rnaseq_csv='output/rnaseq_data.csv')
    
    model.fit(**params)

    predictions = [model.predict_from_mutations(patient=p) for p in model.patients]
    score = recall_score(model.traits.hpv, predictions)

    return score


sampler = TPESampler(multivariate=True)
study = create_study(direction="maximize", sampler=sampler)
study.optimize(lambda trial: objective(trial), n_trials=30)

with open('optuna_params.json', 'w') as f:
    f.write(json.dumps(study.best_params))