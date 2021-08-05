
import io
import pandas as pd
import numpy as np
import json
import os
import pickle
import logging 

model_name = 'model.pkl'
scaler_name = 'scaler.pkl'

"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""

model = None
scaler = None

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger_pred = logging.getLogger('model-prediction')
logger_pred.setLevel(logging.INFO)
logger_feat = logging.getLogger('features')
logger_feat.setLevel(logging.INFO)

def load_model():
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    global model, scaler
    
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    
    if model_name in contents:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name), "rb") as file:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), scaler_name), "rb") as sfile:
                model = pickle.load(file)
                scaler = pickle.load(sfile)
                
                assert model != None
                assert scaler != None
                
                logger_pred.info("Loaded model and scaler...")
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_name, model_dir))
    
    return model

# added for data scaling
def preprocess_data(x):
    
    global scaler
    
    logger_pred.info("Scaling features...")
    
    x = scaler.transform(x)
    
    return x

def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Panda DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction':output from model.predict method}

    """
    
    logger_pred.info("In predict...")
    
    # some check
    assert model is not None, "Model is not loaded"
    
    x = pd.read_json(io.StringIO(data)).values
    
    logger_feat.info("Logging features before scaling")
    logger_feat.info(x)
    logger_feat.info("...")
    
    # apply scaling
    x = preprocess_data(x)
    
    logger_feat.info("Logging features after scaling")
    logger_feat.info(x)
    logger_feat.info("...")
    
    logger_pred.info("Invoking model......")
    
    preds = model.predict_proba(x)
    
    preds = np.round(preds[:, 1], 4)
    preds = preds.tolist()
    
    logger_pred.info("Logging predictions")
    logger_pred.info(preds)
    
    return { 'prediction': preds }
