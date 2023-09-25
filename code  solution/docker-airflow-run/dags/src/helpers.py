import pandas as pd
import numpy as np
import os
import json
import traceback
import datetime
import pickle
import uuid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaseEnsemble
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from glob import glob

try:
    from src import config
    from src import queries
except ImportError:
    import config
    import queries

credentials = json.load(open(config.PATH_TO_CREDENTIALS, 'r'))
engine = create_engine(f"postgresql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['database']}")
print(f"[INFO] Connection to `{credentials['host']}:{credentials['database']}` initiated!")

### data handlers ###
def check_dataset_sanity(df:pd.DataFrame) -> bool:
    nulls = df.isnull().sum()
    nulls = nulls[nulls>0].index.tolist()
    if len(nulls)==0:
        return True
    else:
        raise(Exception(f"There are null values in the training dataset: {nulls}"))

### file handlers ###
def locate_raw_data_filename(job_id:str) -> str:
    """
    Locate the raw data file.
    :param job_id: str
    :return: str
    """

    files = glob(os.path.join(config.PATH_DIR_DATA, "raw", f"{job_id}_*.csv"))
    if len(files) == 0:
        print(f"[WARNING] No raw data file found for job_id : {job_id}.")
        return None
    return files[0]

def locate_preprocessed_filenames(job_id:str) -> dict:
    """
    Locate the preprocessed data files.
    :param job_id: str
    :return: dict
    """
    files = sorted(glob(os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_*.csv")))
    if len(files) == 0:
        raise(Exception(f"No preprocessed data file found for job_id : {job_id}."))
    elif len(files) > 2:
        raise(Exception(f"More than one preprocessed data file found for job_id : {job_id} ->\n{files}"))
    elif len(files) == 1:
        training_filename = None
        inference_filename = list(filter(lambda x: "inference" in x, files))[0]
        return training_filename, inference_filename
    else:
        training_filename = list(filter(lambda x: "training" in x, files))[0]
        inference_filename = list(filter(lambda x: "inference" in x, files))[0]
        return training_filename, inference_filename

def save_dataset(df:pd.DataFrame, path:str):
    """
    Save data set.
    :param df: DataFrame
    :param path: str
    :return: None
    """
    df.to_csv(path, index=False)
    print(f"[INFO] Dataset saved to {path}")

def load_dataset(path:str) -> pd.DataFrame:
    """
    Load data set.
    :param path: str
    :return: DataFrame
    """
    return pd.read_csv(path)

def save_model_as_pickle(model, model_name, directory=None):
    """
    Save a model as a pickle file.
    :param model: AnyType
    :param model_name: str
    :param directory: str
    :return: None
    """
    if directory:
        filename = os.path.join(directory, model_name+".pkl")
    else:
        filename = os.path.join(config.PATH_DIR_MODELS, model_name+".pkl")
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("[INFO] Model saved as pickle file:", filename)

def load_model_from_pickle(model_name: str):
    """
    Load a pickle model.
    :param model_name: str
    :return: AnyType
    """
    with open(os.path.join(config.PATH_DIR_MODELS, model_name+".pkl"), "rb") as f:
        return pickle.load(f)

def save_model_as_json(model:dict, model_name:str, directory:str=None):
    """
    Save a model as a json file.
    :param model: dict
    :param model_name: str
    :param directory: str
    :return: None
    """
    if directory:
        filename = os.path.join(directory, model_name+".json")
    else:
        filename = os.path.join(config.PATH_DIR_MODELS, model_name+".json")
    with open(filename, "w") as f:
        json.dump(model, f)
    print("[INFO] Model saved as json file:", filename)

def load_model_from_json(model_name: str) -> dict:
    """
    Load a json model.
    :param model_name: str
    :return: dict
    """
    with open(os.path.join(config.PATH_DIR_MODELS, model_name+".json"), "r") as f:
        return json.load(f)

def load_deploy_report() -> pd.DataFrame:
    """
    Load a deployment report.
    :param job_id: str
    :return: pd.DataFrame
    """
    return load_model_from_json("deploy_report")
### db handlers ###
def create_table_ml_job():
    """
    Create a table in the database.
    :return: None
    """
    engine.execute(text(queries.CREATE_TABLE_ML_JOB).execution_options(autocommit=True))
    print(f"[INFO] Table {credentials['database']}.mljob ready!")

def create_table_mlreport():
    raise(NotImplementedError)

### job handlers ###
def generate_uuid() -> str:
    """
    Generate a random UUID.
    :return: str
    """
    return str(uuid.uuid4()).replace("-", "")

def log_activity(job_id:str, job_type:str, stage:str, status:str, message:str, job_date:datetime.date=None):
    """
    Logs the activity of a job.
    :param job_id: str
    :param job_type: str
    :param stage: str
    :param status: str
    :param message: str
    :param job_date: datetime.date
    :return: None
    """
    assert stage in config.STAGES, f"[ERROR] Stage `{stage}` is not valid! Choose from {config.STAGES}"
    assert status in config.STATUS, f"[ERROR] Status `{status}` is not valid! Choose from {config.STATUS}"
    assert job_type in config.JOB_TYPES, f"[ERROR] Job type `{job_type}` is not valid! Choose from {config.JOB_TYPES}"
    message = message.replace("'", "\\")
    engine.execute(text(queries.LOG_ACTIVITY.format(job_id=str(job_id), job_type=job_type, stage=str(stage), status=str(status), message=message, job_date=job_date)).execution_options(autocommit=True))
    print(f"[INFO] Job {job_id} logged as {job_type}::{stage}::{status}::{message}")

def get_job_status(job_id:str) -> dict:
    """
    Get the status of a job.
    :param job_id: str
    :return: str
    """
    query = text(queries.GET_JOB_STATUS.format(job_id=job_id))
    try:
        return dict(zip(('job_id', 'job_date', 'stage', 'status', 'message', 'created_at'), engine.execute(query).fetchone()))
    except Exception as e:
        traceback.print_exc()
        return None

def get_job_date(job_id:str) -> datetime.date:
    """
    Get the date of a job.
    :param job_id: str
    :return: datetime.date
    """
    query = text(queries.GET_JOB_DATE.format(job_id=job_id))
    r = engine.execute(query)
    if r is None:
        return None
    return r.fetchone()[0]

def get_job_logs(job_id:str) -> list:
    """
    Get the logs of a job.
    :param job_id: str
    :return: list
    """
    s = ('job_id', 'job_date', 'stage', 'status', 'message', 'created_at')
    query = text(queries.GET_JOB_LOGS.format(job_id=job_id))
    return list(map(lambda x: dict(zip(s, x)), engine.execute(query).fetchall()))

def get_latest_deployed_job_id(status:str="pass") -> str:
    """
    Get the latest deployed job id by looking for the latest of all jobs with stage `deploy` and the specified status.
    :param status: str
    :return: str
    """
    try:
        return json.load(open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"))).get("job_id")
    except Exception as e:
        assert status in config.STATUS, f"[ERROR] Status `{status}` is not valid! Choose from {config.STATUS}"
        query = text(queries.GET_LATEST_DEPLOYED_JOB_ID.format(status=status))
        r = pd.read_sql(query, engine)
        if r.shape[0] == 0:
            return None
        return str(r['job_id'].values[0])

def get_latest_job_id(job_type:str=None, stage:str=None, status:str="pass") -> str:
    """
    Get the latest job id by looking for the latest of all jobs with the specified status.
    :param status: str
    :param job_type: str
    :param stage: str
    :return: str
    """
    assert status in config.STATUS, f"[ERROR] Status `{status}` is not valid! Choose from {config.STATUS}"
    assert job_type in config.JOB_TYPES, f"[ERROR] Job type `{job_type}` is not valid! Choose from {config.JOB_TYPES}"
    assert stage in config.STAGES, f"[ERROR] Stage `{stage}` is not valid! Choose from {config.STAGES}"

    query = text(queries.GET_LATEST_JOB_ID.format(status=status, job_type=job_type, stage=stage))
    r = pd.read_sql(query, engine)
    if r.shape[0] == 0:
        return None
    return str(r['job_id'].values[0])

### model handlers ###
def get_model_type(job_id:str) -> str:
    """
    Get the type of a model.
    :param job_id: str
    :return: str
    """
    report_filename = os.path.join(config.PATH_DIR_MODELS, f"{job_id}_train_report.json")
    return json.load(open(report_filename, "r"))["final_model"]

def persist_deploy_report(job_id:str, model_name:str):
    """
    Persist the deploy report of a job.
    :param job_id: str
    :return: None
    """
    report = {
        "job_id": job_id,
        "purpose_to_int": f"{job_id}_purpose_to_int_model.json",
        "missing_values": f"{job_id}_missing_values_model.pkl",
        "prediction_model": f"{model_name}.pkl",
        "train_report": f"{job_id}_train_report.json",
    }
    json.dump(report, open(os.path.join(config.PATH_DIR_MODELS, f"deploy_report.json"), "w"))
    print(f'[INFO] Deployment report saved as {os.path.join(config.PATH_DIR_MODELS, f"deploy_report.json")}')

if __name__=='__main__':
    j = get_latest_deployed_job_id()
    s = get_job_status(j)
    print(s)

    
