
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

from src.components.data_transformation import DataTransformation


#config.read(path.abspath(path.join(__file__ ,"../../config/config.ini")))


STAGE_NAME = "Data Transformation"



class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
           
            transform_obj=DataTransformation()
                       
            ### DATA CLEANING ###
            df=transform_obj.read_data() # Read the cleaned dataset
            
            df = df.drop(columns=["index", "company_name", "township"],axis=1) # drop the columns
            df = df.drop_duplicates()

            binary_cols = df.iloc[:, 4:].columns.to_list() # convert binary columns 
            df = transform_obj.encode_categorical_variables(df, binary_cols)

            ## sub-area contribustion

            # Contribution of different sub-areas on the dataset 
            df_sa_count = df.groupby("sub_area")["price"].count().reset_index()\
                            .rename(columns={"price":"count"})\
                            .sort_values("count", ascending=False)\
                            .reset_index(drop=True)
            df_sa_count["sa_contribution"] = df_sa_count["count"]/len(df)

            df = transform_obj.fea_eng_sa(df_sa_count, "sub_area",df, 7) # feature enggineering on sub area 
            X_dummy = transform_obj.onehot_end(df,"sub_area")
            X_dummy = X_dummy.astype("int64") # Type conversion

            sa_cols_name = ["sa"+str(i+1) for i in range(X_dummy.shape[1])] # Adding the dummy columns to the dataset
            df.loc[:,sa_cols_name] = X_dummy

            df[["sub_area"]+sa_cols_name].drop_duplicates()\
                        .sort_values("sub_area").reset_index(drop=True) # Sub_area and dummy columns relationship 

            data = df.select_dtypes(exclude="object") # check only object datatype columns
            float_cols = data.select_dtypes( include="float" ).columns.to_list()

            # Price correlation with all other columns
            corr_cols_list = float_cols+binary_cols # Sorted correlations
            corr_list = transform_obj.feat_sel(data, corr_cols_list, "price", sa_cols_name)


            df_corr = pd.DataFrame( data=zip(corr_cols_list, corr_list), 
                            columns=["col_name", "corr"] )\
                        .sort_values("corr", ascending=False)\
                        .reset_index(drop=True)

            features = df_corr.loc[abs(df_corr["corr"])>.1, "col_name"].to_list() 

            lst = transform_obj.feature_sa(df, "sub_area", "price", features)
            ### Data scaling #############

            sel_data = data[features+["price"]].copy() # Selection the final dataset
            sel_data = transform_obj.data_scale(sel_data, "surface")
            transform_obj.save_to_csv(sel_data)
            

        except Exception as e:
            raise e


    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = DataTransformationPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e