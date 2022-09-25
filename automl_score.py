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
    try: 
        test = json.loads(data)
        df = pd.DataFrame.from_dict(test['Inputs']['data'])
        return json.dumps({"result":model.predict(df).tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})

