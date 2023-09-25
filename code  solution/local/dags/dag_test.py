from airflow import DAG
from airflow.operators.dummy import DummyOperator

from airflow.hooks.base_hook import BaseHook
from airflow.operators.slack_operator import SlackAPIPostOperator
import datetime


DAG_ID = "slack_test_workflow"
def create_dag(dag_id):
    with DAG(
        dag_id=dag_id,
        schedule_interval="@daily",
        default_args={
            "owner": "airflow",
            "retries": 0,
            "retry_delay": datetime.timedelta(minutes=1),
            "depends_on_past": False,
            "start_date": datetime.datetime.now() - datetime.timedelta(days=1)
        },
        catchup=False
        
    ) as dag:
        slack_channel = BaseHook.get_connection("slack").login
        slack_token = BaseHook.get_connection("slack").password
        start = DummyOperator(task_id="start")
        task = SlackAPIPostOperator(
            task_id=f'_slack_message_',
            token=slack_token, 
            text="""Hello World! This is a test message from Airflow!""",
            channel=slack_channel, 
            username='airflow'
        )
        end = DummyOperator(task_id="end")
        start >> task >> end
        
    return dag

globals()[DAG_ID] = create_dag(DAG_ID)