
import configparser
config = configparser.RawConfigParser()
import os.path as path
import pandas as pd
import sys
import os
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from src.components.data_cleaning import DataClean
from src.utils.common import *

from src.components.model_trainer import ModelTrainer

from src.constants import *
from src.utils import *
import configparser
config = configparser.RawConfigParser()


config.read(path.abspath(path.join(__file__ ,"../../src/config/config.ini")))

config_data=config["DATA"]
MODEL_DIR = config.get('DATA', 'model_dir')
PROCESSED_DATA_DIR=config.get('DATA', 'processed_data_dir')


print ("*************************",MODEL_DIR)


STAGE_NAME = "MODEL TRAINING"



class MachineLearningModelingPipeline:
    def __init__(self):
        self.config = config.read(CONFIG_FILE_PATH)

    def main(self):
        try:
           
            model_trainer_obj=ModelTrainer()
            data = pd.read_csv(PROCESSED_DATA_DIR) # # read the final csv data
            print(data.columns)
            # data = data.sort_values("surface").reset_index(drop=True)

            X = data.iloc[:, :-1] # # Selecting the feature matrix and target vector
            y = data["price"]

            
            
    
            #### Find BEST MODEL#################
            
            score_df=model_trainer_obj.find_best_model(X,y)
            display(score_df)
            
            scores = cross_val_score(linear_model.Lasso(), X, y, cv=5)
            print('Highest Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))
            
            rs = 118 # # Random sate for data splitting
            X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=.3, random_state=rs) 
            print("************",y_test)

            # # ## Regresiion models - MODEL BUILDING  ###
            model_reg= model_trainer_obj.final_model('linear',X,y,rs,X_train, X_test, y_train, y_test) # # run the required model 
            #model_reg.predict(96)   
            print("Regression Model Executed")
            
            
           
                       
           
        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = MachineLearningModelingPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e