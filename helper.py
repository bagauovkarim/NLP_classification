import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

def evaluate_models_cv(models, X, y, preprocessor, cv, scoring):
    results = {}
    for name, model in models:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        results[name] = {m: scores[f'test_{m}'].mean() for m in scoring}
        results[name]['fit_time'] = scores['fit_time'].mean()
    return pd.DataFrame(results).T