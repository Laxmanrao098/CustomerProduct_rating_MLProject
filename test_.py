import pickle
from src.model_dev import LinearRegressionModel
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df

def train_stream():
    data_path = "/Users/laxma/OneDrive/Desktop/My_project_ML/data/olist_customers_dataset.csv"
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = LinearRegressionModel()
    trained_model = model.train(X_train, y_train)
    return trained_model
regressor = train_stream()
data = {'model': regressor}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)
