from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset, Workspace
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_data(data):
    df = data.to_pandas_dataframe()[["Category", "Message"]]

    y_df = df["Category"]
    
    vectorizer = TfidfVectorizer()
    x_df = vectorizer.fit_transform(df["Message"])

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # Get Dataset 
    workspace = Run.get_context().experiment.workspace
    training_dataset = Dataset.get_by_name(workspace, name='capstone-spam-dataset')    

    x, y = clean_data(training_dataset)
    print(x.shape, y.shape)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()