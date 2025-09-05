import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import joblib

FEATURES = ['hour','dow','dom','month','delta_min']

def train_classifier(df: pd.DataFrame, out_model: str):
    df = df.copy()
    df['is_fail'] = (df['status'] == 'FAILED').astype(int)
    X = df[FEATURES]
    y = df['is_fail']

    preproc = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder='drop')

    def objective(params):
        clf = Pipeline([
            ("prep", preproc),
            ("rf", RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                random_state=42))
        ])
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        clf.fit(Xtr,ytr)
        return {'loss':1-clf.score(Xte,yte),'status':STATUS_OK,'model':clf}

    space = {
        'n_estimators': hp.quniform('n_estimators',80,220,20),
        'max_depth': hp.quniform('max_depth',4,14,1),
        'min_samples_split': hp.quniform('min_samples_split',2,10,1)
    }
    trials = Trials()
    fmin(objective,space,algo=tpe.suggest,max_evals=25,trials=trials)
    best_model = trials.best_trial['result']['model']
    joblib.dump(best_model,out_model)
    return best_model