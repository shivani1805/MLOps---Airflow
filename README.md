# Airflow DAG: Wine Quality Model Training (UCI Dataset)

This project demonstrates a simple **Apache Airflow DAG** that automates a Machine Learning workflow using the **Wine Quality dataset** from the UCI Machine Learning Repository.
The pipeline builds and evaluates a **Random Forest Classifier** model on the dataset.

---

## Structure

### `model_development.py`

Contains the ML pipeline logic:

1. **load_data()** – Loads the UCI Wine Quality dataset (CSV file)
2. **data_preprocessing()** – Splits data into training and testing sets
3. **build_model()** – Trains a RandomForest model and saves it as a `.sav` file
4. **load_model()** – Loads the saved model and prints its accuracy on the test set

### `wine_dag.py`

Defines the Airflow DAG with the following tasks:

1. **Load Data**
2. **Preprocess Data**
3. **Train and Save Model**
4. **Evaluate Model**

Each task uses a `PythonOperator` to call the corresponding Python function.

---

## Running the Project

### Initialize Airflow

1. Create a virtual environment outside the **`dags`** folder:
```bash
python3 -m venv airflow_new_venv 
source airflow_new_venv/bin/activate
```
2. Install dependencies - 

```bash
pip install apache-airflow scikit-learn pandas numpy
```
3. Set the Airflow home:
```bash
cd dags 
export AIRFLOW_HOME=/absolute/path/to/airflow_wine_project/config
```
4. In config/airflow.cfg, ensure dags_folder points to the absolute path of your dags folder.

---


### Start Airflow services

```bash
airflow standalone
```
Log in using:

* Username: admin

* Password: Check the file simple_auth_manager_passwords.json.generated in the config folder.

Then open the Airflow UI at:

[http://localhost:8080](http://localhost:8080)

---
### Enable and run the DAG

In the Airflow UI:

* Locate the DAG **wine_dag**
* Toggle it **on**
* Click **Trigger DAG** to run it manually

---

### Check the results

After a successful run:

* The trained RandomForest model will be saved in:

  ```
  model/rf_model.sav
  ```
* The "evaluate_model_task" logs will show the model’s test accuracy.

---

## Author

Developed by **Shivani Sharma**
For **MLOps Lab Assignment – Airflow**
