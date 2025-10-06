from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.model_development import load_data, data_preprocessing, build_model, load_model

default_args = {
    'owner': 'Shivani Sharma',
    'start_date': datetime(2025, 10, 6),
    'retries': 0
}

dag = DAG(
    'wine_quality_dag',
    default_args=default_args,
    description='DAG to train and evaluate a RandomForest model on the UCI Wine dataset',
    schedule=None,
    catchup=False,
    tags=['ml', 'wine']
)

load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag
)

build_model_task = PythonOperator(
    task_id='build_model_task',
    python_callable=build_model,
    op_args=[data_preprocessing_task.output, "rf_model.sav"],
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=load_model,
    op_args=[data_preprocessing_task.output, "rf_model.sav"],
    dag=dag
)

load_data_task >> data_preprocessing_task >> build_model_task >> evaluate_model_task
