
import sys
import os
import os.path as path

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from HomePrice.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from HomePrice.pipelines.stage_03_data_cleaning import DataCleaningPipeline
from HomePrice.pipelines.stage_04_data_transformation import DataTransformationPipeline
from HomePrice.pipelines.stage_05_ml_modeling import MachineLearningModelingPipeline

STAGE_NAME = "Data Ingestion"

try:
   print(">>>>>> Stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   print(">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        print((e))
        raise e
     
STAGE_NAME = "DATA CLEANING"
try:
   print(">>>>>> Stage {STAGE_NAME} started <<<<<<",STAGE_NAME) 
   data_cleaning = DataCleaningPipeline()
   data_cleaning.main()
   print(">>>>>> Stage {STAGE_NAME} completed <<<<<<",STAGE_NAME)
except Exception as e:
        print(e)
        raise e
          
STAGE_NAME = "Data Transformation"
try:
   print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
   obj = DataTransformationPipeline()
   obj.main()
   print(">>>>>> Stage completed :", STAGE_NAME)
except Exception as e:
   print(e)
   raise e

STAGE_NAME = "Model Training"
try:
   print(">>>>>> Stage started <<<<<<:",STAGE_NAME)
   obj = MachineLearningModelingPipeline()
   obj.main()
   print(">>>>>> Stage  completed <<<<<<>:", STAGE_NAME)
except Exception as e:
   print(e)
   raise e

