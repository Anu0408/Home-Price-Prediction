import os
import sys
import urllib.request as request
import zipfile
from pathlib import Path
import configparser
config = configparser.RawConfigParser()
import os.path as path
src_directory=os.path.abspath(path.join(__file__,"../../../"))
sys.path.append(src_directory)
from src.constants import *
from src.utils import *

STAGE_NAME = "Data Ingestion"

class DataIngestion:
    def __init__(self):
        self.config = config.read(CONFIG_FILE_PATH)
        
        
            
    def download_file(self):
        if not os.path.exists(config.get('DATA', 'local_data_file')):
            filename, headers = request.urlretrieve(
                url = config.get('DATA', 'source_url'),
                filename = config.get('DATA', 'local_data_file')
            )
            print(f"{filename} download! with following info: \n{headers}")
        else:
            print(f"File already exists of size:")



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
  