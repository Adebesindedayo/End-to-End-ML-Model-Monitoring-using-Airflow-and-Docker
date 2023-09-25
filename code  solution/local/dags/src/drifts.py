from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.ensemble import BaseEnsemble
from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import WholeDatasetDrift, DataDuplicates 
from deepchecks.tabular.checks import NewLabelTrainTest, TrainTestFeatureDrift, TrainTestLabelDrift
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureLabelCorrelationChange, ConflictingLabels, OutlierSampleDetection 
from deepchecks.tabular.checks import WeakSegmentsPerformance, PerformanceReport, RocReport, ConfusionMatrixReport, TrainTestPredictionDrift, CalibrationScore, BoostingOverfit

try:
    from src import config
    from src import helpers
    from src import etl
    from src import preprocess
except:
    import config
    import helpers
    import etl
    import preprocess

def check_data_quality(df:pd.DataFrame, predictors:list, target:str, job_id:str):
    """
    checks for data quality.
    A report will be saved in the results directory.
    :param df: dataframe to check
    :param predictors: predictors to check for drifts
    :param target: target variable to check for drifts
    :param job_id: job ID
    :return: boolean
    """
    features = [col for col in predictors if col in df.columns]
    cat_features = [col for col in config.CAT_VARS if col in df.columns]
    dataset = Dataset(df, label=target, features=features, cat_features=cat_features, datetime_name=config.DATETIME_VARS[0])
    retrain_suite = Suite("data quality",
        DataDuplicates().add_condition_ratio_less_or_equal(0.3), #Checks for duplicate samples in the dataset
        ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(0), #Find samples which have the exact same features' values but different labels
        FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9), #Return the PPS (Predictive Power Score) of all features in relation to the label
        OutlierSampleDetection(outlier_score_threshold=0.7).add_condition_outlier_ratio_less_or_equal(0.1), #Detects outliers in a dataset using the LoOP algorithm
    )
    r = retrain_suite.run(dataset)
    try:
        r.save_as_html(f"{config.PATH_DIR_RESULTS}/{job_id}_data_quality_report.html")
        print("[INFO] Data quality report saved as {}".format(f"{config.PATH_DIR_RESULTS}/{job_id}_data_quality_report.html"))
    except Exception as e:
        print(f"[WARNING][DRIFTS.SKIP_TRAIN] {traceback.format_exc()}")
    return {"report": r, "retrain": r.passed()}

def check_data_drift(ref_df:pd.DataFrame, cur_df:pd.DataFrame, predictors:list, target:str, job_id:str):
    """
    Check for data drifts between two datasets and decide whether to retrain the model. 
    A report will be saved in the results directory.
    :param ref_df: Reference dataset
    :param cur_df: Current dataset
    :param predictors: Predictors to check for drifts
    :param target: Target variable to check for drifts
    :param job_id: Job ID
    :return: boolean
    """
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in config.CAT_VARS if col in ref_df.columns]
    cur_cat_features = [col for col in config.CAT_VARS if col in cur_df.columns]
    ref_dataset = Dataset(ref_df, label=target, features=ref_features, cat_features=ref_cat_features, datetime_name=config.DATETIME_VARS[0])
    cur_dataset = Dataset(cur_df, label=target, features=cur_features, cat_features=cur_cat_features, datetime_name=config.DATETIME_VARS[0])
    
    suite = Suite("data drift",
        NewLabelTrainTest(),
        WholeDatasetDrift().add_condition_overall_drift_value_less_than(0.01), #0.2
        FeatureLabelCorrelationChange().add_condition_feature_pps_difference_less_than(0.05), #0.2
        TrainTestFeatureDrift().add_condition_drift_score_less_than(0.01), #0.1
        TrainTestLabelDrift().add_condition_drift_score_less_than(0.01) #0.1
    )
    r = suite.run(ref_dataset, cur_dataset)
    retrain = (len(r.get_not_ran_checks())>0) or (len(r.get_not_passed_checks())>0)
    
    try:
        r.save_as_html(f"{config.PATH_DIR_RESULTS}/{job_id}_data_drift_report.html")
        print("[INFO] Data drift report saved as {}".format(f"{config.PATH_DIR_RESULTS}/{job_id}_data_drift_report.html"))
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_DATA_DRIFT] {traceback.format_exc()}")
    return {"report": r, "retrain": retrain}

def check_model_drift(ref_df:pd.DataFrame, cur_df:pd.DataFrame, model:BaseEnsemble, predictors:list, target:str, job_id:str):
    """
    Using the same pre-trained model, compare drifts in predictions between two datasets and decides whether to retrain the model. A report will be saved in the results directory.
    :param ref_df: Reference dataset
    :param cur_df: Current dataset
    :param model: Pre-trained model. Only scikit-learn and xgboost models are supported.
    :param predictors: Predictors to check for drifts
    :param target: Target variable to check for drifts
    :param job_id: Job ID
    :return: boolean
    """
    ref_features = [col for col in predictors if col in ref_df.columns]
    cur_features = [col for col in predictors if col in cur_df.columns]
    ref_cat_features = [col for col in config.CAT_VARS if col in ref_df.columns]
    cur_cat_features = [col for col in config.CAT_VARS if col in cur_df.columns]
    ref_dataset = Dataset(ref_df, label=target, features=ref_features, cat_features=ref_cat_features, datetime_name=config.DATETIME_VARS[0])
    cur_dataset = Dataset(cur_df, label=target, features=cur_features, cat_features=cur_cat_features, datetime_name=config.DATETIME_VARS[0])
    
    suite = Suite("model drift",
        #For each class plots the ROC curve, calculate AUC score and displays the optimal threshold cutoff point.
        RocReport().add_condition_auc_greater_than(0.7), 
        #Calculate prediction drift between train dataset and test dataset, Cramer's V for categorical output and Earth Movers Distance for numerical output.
        TrainTestPredictionDrift().add_condition_drift_score_less_than(max_allowed_categorical_score=0.1) 
        )
    r = suite.run(ref_dataset, cur_dataset, model)
    retrain = (len(r.get_not_ran_checks())>0) or (len(r.get_not_passed_checks())>0)
    try:
        r.save_as_html(f"{config.PATH_DIR_RESULTS}/{job_id}_model_drift_report.html")
        print("[INFO] Model drift report saved as {}".format(f"{config.PATH_DIR_RESULTS}/{job_id}_model_drift_report.html"))
    except Exception as e:
        print(f"[WARNING][DRIFTS.check_MODEL_DRIFT] {traceback.format_exc()}")
    
    return {"report": r, "retrain": retrain}


if __name__=='__main__':
    start_date1 = datetime.date(2015, 1, 1)
    end_date1 = datetime.date(2016, 5, 31)
    start_date2 = datetime.date(2016, 6, 1)
    end_date2 = datetime.date(2016, 12, 31)

    target = config.TARGET
    predictors = config.PREDICTORS

    ref_job_id = "94173e40bbbd40f69dd8824ab2cfa6bb" #helpers.generate_uuid()
    # etl.collect_data(start_date=start_date1, end_date=end_date1, job_id=job_id)
    ref_df = helpers.load_dataset(helpers.locate_raw_data_filename(ref_job_id))
    
    cur_job_id = "ee48bcf0b07742bfb616d4c02a267d71" #helpers.generate_uuid()
    # etl.collect_data(start_date=start_date2, end_date=end_date2, job_id=job_id)
    cur_df = helpers.load_dataset(helpers.locate_raw_data_filename(cur_job_id))

    r = skip_train(cur_df, predictors, target, cur_job_id)
    print("skip training due to data quality: ", r)

    r = check_data_drift(ref_df, cur_df, predictors, target, cur_job_id)
    print("retrain due to raw data drifts?:", r)

    preprocess.preprocess_data(df=ref_df, mode="training", job_id=ref_job_id)
    preprocess.preprocess_data(df=cur_df, mode="inference", job_id=cur_job_id, ref_job_id=ref_job_id)

    _, ref_filename = helpers.locate_preprocessed_filenames(ref_job_id)
    _, cur_filename = helpers.locate_preprocessed_filenames(cur_job_id)
    model_type = helpers.get_model_type("dcb5974cab9f48fba46e95407dc2ea97")
    model = helpers.load_model_from_pickle(f"dcb5974cab9f48fba46e95407dc2ea97_{model_type}")
    r = check_model_drift(
        ref_df=helpers.load_dataset(ref_filename), 
        cur_df=helpers.load_dataset(cur_filename), 
        model=model, 
        predictors=predictors, 
        target=target, 
        job_id=cur_job_id
        )
    print("retrain due to model drifts?:", r)
    
