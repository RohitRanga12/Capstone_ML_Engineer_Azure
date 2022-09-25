import json
import joblib
import os
import pandas as pd

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best-automl-model.pkl')
    model = joblib.load(model_path)


def run(data):
    test = json.loads(data)
    # test = test.get("Inputs").get("data").get("message")
#    input_data = test["Inputs"]["data"][0]
    df = pd.DataFrame.from_dict(test["Inputs"]["data"])
    return model.predict(df)