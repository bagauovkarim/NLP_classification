import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def plots_importances(pipeline, name, top_n=20):

    vec = pipeline.named_steps["preprocessor"].named_steps["vectorizer"]
    feature_names = vec.get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        importances = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f'{name} нет фичей')
        return
    s = pd.Series(np.abs(importances), index=feature_names).sort_values().tail(top_n)
    s.plot(kind="barh", figsize=(8, 6))
    plt.title(f"{name} - топ-{top_n} важных фич")
    plt.show()

def plot_metrics_with_time(df):
    metric_cols = ["accuracy", "f1", "roc_auc"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    sns.heatmap(df[metric_cols], annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Метрики")

    sns.heatmap(df[["fit_time"]], annot=True, fmt=".3f", cmap="Reds", ax=axes[1])
    axes[1].set_title("Время (с)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()