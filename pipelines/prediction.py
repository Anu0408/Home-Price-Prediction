

import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import pickle

from src.constants import *
from src.utils import *
import configparser
config = configparser.RawConfigParser()


config.read(path.abspath(path.join(__file__ ,"../../src/config/config.ini")))

config_data=config["DATA"]
MODEL_DIR = config.get('DATA', 'model_dir')

class PredictionPipeline:
    def __init__(self):
        # load model
        with open(MODEL_DIR+'reg_model.pkl', 'rb') as f:
             self.pred_model = pickle.load(f)

    
    def predict(self, data):
        
        
        return self.pred_model.predict(data)
        
        #return 63.5