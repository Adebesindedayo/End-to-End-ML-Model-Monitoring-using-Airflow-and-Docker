import datetime
import os
import pandas as pd
from sqlalchemy.sql import text

try:
    from src import config
    from src import helpers
    from src import queries
except ImportError:
    import config
    import helpers
    import queries

# helpers.create_table_ml_job()

def extract_data(start_date:datetime.date, end_date:datetime.date=datetime.date.today()) -> pd.DataFrame:
    """
    Extracts data from the database and returns it as a pandas dataframe.
    Queries are to be defined in the `queries.py` file.
    :param start_date: start date of the data to be extracted
    :param end_date: end date of the data to be extracted
    :return: pandas dataframe
    """
    assert start_date <= end_date, "start_date must be less than end_date"
    print("[INFO] Extracting data from the database since {0} to {1} ...".format(start_date, end_date))
    helpers.engine.execute(text("""drop table if exists customer;""").execution_options(autocommit=True))
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_CUSTOMER).execution_options(autocommit=True))
    helpers.engine.execute(text("""drop table if exists loan;""").execution_options(autocommit=True))
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_LOAN.format(start_date=start_date, end_date=end_date)).execution_options(autocommit=True))
    helpers.engine.execute(text("""drop table if exists credit;""").execution_options(autocommit=True))
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_CREDIT).execution_options(autocommit=True))
    df = pd.read_sql(text(queries.GET_DATA), helpers.engine)
    return df

def collect_data(start_date:datetime.date, end_date:datetime.date=datetime.date.today(), job_id:str=None):
    """
    Collects data from the database and dump them in the directory of raw data `config.PATH_DIR_DATA`.
    :param start_date: start date of the data to be extracted
    :param end_date: end date of the data to be extracted
    :param job_id: job id of the data to be extracted
    """
    assert isinstance(start_date, datetime.date)
    assert isinstance(end_date, datetime.date)
    assert isinstance(job_id, str)
    assert start_date <= end_date
    size = 0

    df = extract_data(start_date, end_date)
    size = df.shape[0]
    filename = os.path.join(config.PATH_DIR_DATA, "raw", f"{job_id}_"+start_date.strftime("%Y-%m-%d")+"_"+end_date.strftime("%Y-%m-%d")+".csv")
    helpers.save_dataset(df, filename)
    return filename

if __name__=="__main__":
    job_id = helpers.generate_uuid()
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date.today()
    collect_data(start_date, end_date, job_id)
