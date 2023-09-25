import pandas as pd
import numpy as np
import os
import datetime
import traceback
from sklearn.metrics import roc_auc_score,  accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

"""
Improvements:
    1. Use cross-validation to find the best hyperparameters.
    2. Use grid search to find the best hyperparameters.
    3. Use Pipeline to combine multiple steps concisely.
    4. Feature importance.
"""
try:
    from src import config
    from src import helpers
except ImportError:
    import config
    import helpers

def performance_report(y_true, y_pred, y_prob):
    """
    Generate performance report for a model.
    :param y_true: np.array
    :param y_pred: np.array
    :param y_prob: np.array
    :return: dict
    """
    report = dict()
    report["dataset size"] = y_true.shape[0]
    report["positive rate"] = y_true.sum()/y_true.shape[0]
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["f1"] = f1_score(y_true, y_pred)
    report["precision"] = precision_score(y_true, y_pred)
    report["recall"] = recall_score(y_true, y_pred)
    try:
        report["auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(e)    
    return report

def select_model(df:pd.DataFrame, metric:str=config.MODEL_PERFORMANCE_METRIC, model_names:list=["rf", "gb"], performance_thresh:float=config.MODEL_PERFORMANCE_THRESHOLD, degradation_thresh:float=config.MODEL_DEGRADATION_THRESHOLD)->str:
    """
    Select the best model based on their performance reports.
        - metric >= performance_thresh where metric can be auc, recall, precision, f1_score, ... and performance_thresh is any value between 0.0 and 1.0
        - abs(<metric>_train - <metric>_test) <= degradation_thresh
    :param df: pd.DataFrame, performance report
    :param metric: str, metric to select the best model.
    :param model_names: list, model names to select from.
    :param performance_thresh: float, threshold for the performance.
    :return: str, model name.
    """
    degradation_performance = []
    for model in model_names:
        if df.loc[metric, f"{model}_train"] < performance_thresh:
            continue
        degradation = df.loc[metric, f"{model}_train"] - df.loc[metric, f"{model}_test"]
        if degradation < degradation_thresh:
            degradation_performance.append((model, degradation))
    if len(degradation_performance) == 0:
        raise(Exception("No model selected: all models have performance below the threshold. Possible overfitting."))
    return min(degradation_performance, key=lambda x: x[1])[0]

def pick_model_and_deploy(job_id, models, df, metric="auc", predictors=config.PREDICTORS, target=config.TARGET):
    """
    Among all `models`, select the model that performs best on df and mark it for deployment.
    :param job_id: str, job id.
    :param models: list of key-value items {"job_id": <str>, "purpose_to_int: <str>, "missing_values": <str>, "prediction_model": <>, "train_report": <str>}
    :param df: pd.DataFrame, test dataset
    :param metric: str, metric used to select the best model.
    :param predictors: list, predictors to use.
    :param target: str, target to use.
    :return: None
    """
    assert len(models) > 0, "`models` cannot be empty"
    if len(models)==1:
        model_name = models[0]["model_name"]
        helpers.persist_deploy_report(job_id, model_name)
        return model_name
    cols = set(predictors).difference(set(df.columns))
    assert cols == set(), f"{cols} not in {df.columns}"
    score = 0
    m_idx = 0
    for i, m in enumerate(models):
        y_true = df[target]
        y_pred = m["model"].predict(df[predictors])
        y_prob = m["model"].predict_proba(df[predictors])[:, 1]
        r = performance_report(y_true, y_pred, y_prob)
        if r[metric] > score:
            score = r[metric]
            m_idx = i
    helpers.persist_deploy_report(job_id, models[m_idx]["model_name"])
    return models[m_idx]["model_name"]
    
def train(train_dataset_filename:str=None, test_dataset_filename:str=None, job_id="", rescale=False):
    """
    Train a model on the train dataset loaded from `train_dataset_filename` 
    and test dataset loaded from `test_dataset_filename`.
    :param train_dataset_filename: str
    :param test_dataset_filename: str
    :param job_id: str
    :param rescale: bool, if true, scaled numerical variables used
    :return: None
    """
    if train_dataset_filename==None:
        train_dataset_filename = os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_training.csv")
    if test_dataset_filename==None:
        test_dataset_filename = os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_inference.csv")
    tdf = helpers.load_dataset(train_dataset_filename)
    vdf = helpers.load_dataset(test_dataset_filename)
    helpers.check_dataset_sanity(tdf)
    helpers.check_dataset_sanity(vdf)
    
    predictors = config.PREDICTORS
    target = config.TARGET.lower()
    if rescale:
        for col in predictors:
            if f"{config.RESCALE_METHOD}_{col}" in tdf.columns:
                tdf[col] = tdf[f"{config.RESCALE_METHOD}_{col}"]
            if f"{config.RESCALE_METHOD}_{col}" in vdf.columns:
                vdf[col] = vdf[f"{config.RESCALE_METHOD}_{col}"]
        
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    X, Y = tdf[predictors], tdf[target]
    report = dict()
    models = dict()
    for cl, name in [(rf, "rf"), (gb, "gb")]:
        print("[INFO] Training model:", name)
        cl.fit(X, Y)
        t_pred = cl.predict(X)
        v_pred = cl.predict(vdf[predictors])
        t_prob = cl.predict_proba(X)[:, 1]
        v_prob = cl.predict_proba(vdf[predictors])[:, 1]
        report[f"{name}_train"] = performance_report(Y, t_pred, t_prob)
        report[f"{name}_test"] = performance_report(vdf[target], v_pred, v_prob)
        models[name] = cl
        
    model_name = select_model(pd.DataFrame(report), metric=config.MODEL_PERFORMANCE_METRIC, model_names=list(models.keys()))
    report["final_model"] = model_name
    helpers.save_model_as_pickle(models[model_name], f"{job_id}_{model_name}")
    helpers.save_model_as_json(report, f"{job_id}_train_report")
    return report

if __name__=="__main__":
    train()