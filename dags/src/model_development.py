import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data():
    """Load the wine quality dataset and pickle it"""
    dag_folder = os.path.dirname(__file__)
    data_path = os.path.join(dag_folder, "../data/wine.csv")
    print("Loading CSV from:", os.path.abspath(data_path))
    df = pd.read_csv(data_path, sep=";")
    
    pickle_path = os.path.join(dag_folder, "../data/wine.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)
    
    return pickle_path 

def data_preprocessing(pickle_path):
    """Load pickled data, split into train/test and pickle the result"""
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    train_test_path = os.path.join(os.path.dirname(__file__), "../data/train_test.pkl")
    with open(train_test_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    
    return train_test_path

def build_model(train_test_pickle_path, filename="rf_model.sav"):
    """Train RandomForest and save the model"""
    with open(train_test_pickle_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    
    return model_path

def load_model(train_test_pickle_path, filename="rf_model.sav"):
    """Load model pickle, evaluate on test data"""
    with open(train_test_pickle_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    acc = model.score(X_test, y_test)
    print(f"RandomForest accuracy on test data: {acc}")
    return acc

if __name__ == "__main__":
    df_pickle = load_data()
    train_test_pickle = data_preprocessing(df_pickle)
    model_path = build_model(train_test_pickle)
    load_model(train_test_pickle)
