
import configparser
config = configparser.RawConfigParser()
import os.path as path
import pandas as pd
import sys
import os

parent_directory = os.path.abspath(path.join(__file__ ,"../../"))
sys.path.append(parent_directory)

from src.components.data_cleaning import DataClean
from src.utils.common import *

#config.read(path.abspath(path.join(__file__ ,"../../config/config.ini")))

#config_data=config["DATA"]
#DATA_DIR = config.get('DATA', 'local_data_file')
#CLEANED_DATA_DIR=config.get('DATA', 'clean_data_dir')

#print ("*************************",CLEANED_DATA_DIR)

STAGE_NAME = "Data Cleaning"

cleaning_obj=DataClean()

class DataCleaningPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            
           # dfr = read_data(DATA_DIR) # Read the initial dataset
            
            dfr = cleaning_obj.read_data()
            
            ### DATA CLEANING ###

            dfr = cleaning_obj.rename_col(dfr,"Propert Type", "Property Type" ) # reanme the data column
            df_norm = cleaning_obj.drop_val(dfr,"Property Type","shop") # drop a row an property column
            #############################################################
            
            #df_norm = cleaning_obj.normaliseProps(dfr) # Normalising the Propert Type and Property Area in Sq. Ft.
            
            
            df_norm["Property Type"] = df_norm["Property Type"].apply(cleaning_obj.splitSums)
            df_norm["Property Area in Sq. Ft."] = \
            df_norm["Property Area in Sq. Ft."]\
                .apply( lambda x : cleaning_obj.splitSums(x, False) )
            
            ##############################################################
            
            x_prt = df_norm['Property Type'] # # Checking the outliers for Property type.  ## you can change the columns and check for the outliers for different features.
            prt_up_lim = cleaning_obj.computeUpperFence(x_prt)
            df_norm[ x_prt>prt_up_lim]
            #df_norm.drop(index=86, inplace=True) # drop row 86 - after viewing the data we decide to remove rows 86 as it looks an a potential outliers
            df_norm.drop(index=df_norm[df_norm["Property Type"]==7].index, inplace=True) # dropping 7 bhk entry as they are potential ouliers that we can view with the help of scatter plot. 
             
            x_prt = df_norm['Property Type'] # # Checking the outliers for Property type.  ## you can change the columns and check for the outliers for different features.
            prt_up_lim = cleaning_obj.computeUpperFence(x_prt)
            df_norm[ x_prt>prt_up_lim]
            df_norm.drop(index=86, inplace=True) # drop row 86 - after viewing the data we decide to remove rows 86 as it looks an a potential outliers
            df_norm.drop(index=df_norm[df_norm["Property Type"]==7].index, inplace=True) # dropping 7 bhk entry as they are potential ouliers that we can view with the help of scatter plot. 
               
            
            # price selection
            # There are two target variables  - price in lakhs and price in millions; with the help of the plot we can conclude that they are the same variables; hence wew drop one;

            df_norm["Price in lakhs"] = df_norm["Price in lakhs"]\
                                .apply(lambda x: pd.to_numeric(x, errors='coerce') ) # Comparing Price in Millions with Price in lakhs
            #df_norm = cleaning_obj.drop_col(df_norm, ["Price in lakhs"] )
            df_norm.drop(columns=["Price in lakhs"], axis=1)
            
            ################################

            cleaning_obj.compute_fill_rate( df_norm ) ### Dealing with the NAN values
            df_norm[["Sub-Area", "TownShip Name/ Society Name", "Total TownShip Area in Acres" ]]\
                .sort_values("Sub-Area").reset_index(drop=True)        # Total TownShip Area in Acres
            df_norm = cleaning_obj.drop_empty_axis(df_norm, minFillRate=.5)  # Drop columns filled by less than 50%


            ################################


            ### Regularising the categorical columns ##
            binary_cols = df_norm.iloc[:,-7:].columns.to_list()
            df_norm = df_norm[df_norm["Price in Millions"]<80] # keep the target values less than 80
            binary_cols = cleaning_obj.reg_catvar(df_norm, binary_cols) # convert to binary 

            obj_cols = df_norm.select_dtypes(include="object").columns.to_list() ## Multi-categorical columns
            multiCat_cols = list(set(obj_cols)^set(binary_cols))
            multiCat_cols = cleaning_obj.reg_catvar(df_norm, multiCat_cols) # convert for multicategorical vars

            df_norm = df_norm.drop(columns=["Location"],axis=1) # drop columns
            df_norm = df_norm.drop(columns=["Price in Millions"],axis=1) # drop columns
            df_norm.columns=[ "index","sub_area", "n_bhk", "surface", "price", 
                                                "company_name", "township",
                                                "club_house", "school", "hospital", 
                                                "mall", "park", "pool", "gym"] # Renaming the columns

            #df_norm.to_csv(CLEANED_DATA_DIR, index=False)
            
            cleaning_obj.save_to_csv(df_norm) 

        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = DataCleaningPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e