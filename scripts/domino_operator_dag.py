from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from domino.airflow import DominoOperator

# Parameters to DAG object
default_args = {
    'owner': 'domino',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate a DAG
dag = DAG(
    'domino_operator_demo',
    description='Execute Airflow DAG with DominoOperator',
    default_args=default_args,
    schedule=timedelta(days=1),
    catchup=False,
)

start_task = EmptyOperator(
    task_id='start',
    dag=dag,
)

# Using DominoOperator for cleaner syntax
data_processing = DominoOperator(
    task_id='data_processing',
    project='your-username/your-project-name',
    command=['python', 'scripts/process_data.py'],
    title='Data Processing Job',
    tier='Small',
    poll_freq=10,  # Check job status every 10 seconds
    max_poll_time=3600,  # Max wait time: 1 hour
    dag=dag,
)

model_training = DominoOperator(
    task_id='model_training',
    project='your-username/your-project-name', 
    command=['python', 'scripts/train_model.py'],
    title='Model Training Job',
    tier='Medium',
    poll_freq=10,
    max_poll_time=7200,  # Max wait time: 2 hours
    dag=dag,
)

generate_report = DominoOperator(
    task_id='generate_report',
    project='your-username/your-project-name',
    command=['python', 'scripts/generate_report.py'], 
    title='Generate Report Job',
    tier='Small',
    poll_freq=10,
    max_poll_time=1800,  # Max wait time: 30 minutes
    dag=dag,
)

end_task = EmptyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start_task >> data_processing >> model_training >> generate_report >> end_task