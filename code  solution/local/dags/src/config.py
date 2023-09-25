RUN_LOCAL = False
if RUN_LOCAL:
    PATH_DIR_DATA = "../dags/data"
    PATH_DIR_MODELS = "../dags/models"
    PATH_DIR_RESULTS = "../dags/results"
    PATH_TO_CREDENTIALS = "../dags/Creds.json"
    PATH_TO_APP_SHELL = "../dags/app.sh"
else:
    PATH_DIR_DATA = "/opt/airflow/dags/data"
    PATH_DIR_MODELS = "/opt/airflow/dags/models"
    PATH_DIR_RESULTS = "/opt/airflow/dags/results"
    PATH_TO_CREDENTIALS = "/opt/airflow/dags/Creds.json"
    PATH_TO_APP_SHELL = "/opt/airflow/dags/app.sh"

RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.3
PROB_THRESHOLD = 0.5
SPLIT_METHOD = "time based"

# lowest acceptable difference between the performances of the same model on two different datasets
MODEL_DEGRADATION_THRESHOLD = 0.1
ASSOCIATION_DEGRADATION_THRESHOLD = 0.3

# lowest acceptable performance of either accuracy, precision, recall, f1 or auc depending on the classification usecase
MODEL_PERFORMANCE_THRESHOLD = 0.7 
MODEL_PERFORMANCE_METRIC = "auc"

IDENTIFIERS = ['loan_id', 'customer_id']
TARGET = 'loan_status'
DATETIME_VARS = ['application_time']
EXC_VARIABLES = [
    'application_time'
    ]
PURPOSE_ENCODING_METHOD = "weighted ranking" # choose from (ranking, one-hot, weighted ranking, relative ranking)
RESCALE_METHOD = "standardize" # choose from (standardize, minmax, None)
CAT_VARS = [
    'term', 
    'home_ownership', 
    'purpose',
    'years_in_current_job', 
    ]
NUM_VARS = [
    'current_loan_amount', 
    'credit_score', 
    'monthly_debt',
    'annual_income',
    'years_of_credit_history', 
    'months_since_last_delinquent', 
    'no_of_open_accounts',
    'current_credit_balance',
    'max_open_credit',
    'bankruptcies',
    'tax_liens', 
    'no_of_properties', 
    'no_of_cars',
    'no_of_children', 
    'no_of_credit_problems', 
    ]

PREDICTORS = [
    "current_loan_amount",
    "term",
    "credit_score",
    "years_in_current_job",
    "home_ownership",
    "annual_income",
    "purpose",
    "monthly_debt",
    "years_of_credit_history",
    "months_since_last_delinquent",
    "no_of_open_accounts",
    "no_of_credit_problems",
    "current_credit_balance",
    "max_open_credit",
    "bankruptcies",
    "tax_liens",
    'no_of_properties', 
    'no_of_cars',
    'no_of_children',
    "application_year",
    "application_month",
    "application_week",
    "application_day",
    "application_season",
    "current_credit_balance_ratio",
]

STAGES = [
    "etl", "preprocess", "training", "testing", "inference", "postprocess", "preprocess-training", "preprocess-inference", "report", "driftcheck",
    "etl_report", "raw_data_drift-report", "deploy"
    ]
STATUS = ["pass", "fail", "skipped", "started"]
JOB_TYPES = ["training", "inference", None]


