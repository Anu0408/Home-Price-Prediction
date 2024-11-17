import sys
import os
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
src_directory=os.path.abspath(path.join(__file__,"../../../"))
sys.path.append(src_directory)
from src.constants import *
from src.utils import *
import configparser
config = configparser.RawConfigParser()
import pandas as pd


class DataTransformation:
    def __init__(self):
        self.config = config.read(CONFIG_FILE_PATH)
        
    
    # Function to read clean data and return dataframe
    
    def read_data(self):
        return pd.read_csv(config.get('DATA', 'clean_data_dir'))
    
    # Function for performing label encoding for the categorical variables

    def encode_categorical_variables(self,df, cat_vars):
    # Transforming the yes/no to 1/0
        laben = LabelEncoder()
        for col in cat_vars:
            df[col] = laben.fit_transform(df[col])
        return (df)
    

    # function for surface area column
    def fea_eng_sa(self,df_count, df_col, df, n):
            sa_sel_col = df_count.loc[df_count["count"]>n, df_col].to_list()
            df[df_col] = df[df_col].where(df[df_col].isin(sa_sel_col), "other")
            return df


    # function to perform one hot encoding
    def onehot_end(self,df,col_name):
        # Dummy variable conversion
        hoten = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_dummy = hoten.fit_transform(df[[col_name]] )
        return X_dummy
    
    # fucntion to perform feature selection
    def feat_sel(self,data, corr_cols_list,target,col_name):
    # Price correlation with all other columns
        corr_cols_list.remove(target)
        corr_cols_list.extend(col_name)
        corr_list = [] # to keep the , correlations with price
        for col in corr_cols_list:
            corr_list.append(round(data[target].corr(data[col]),2) )    
        return corr_list


    # function for surface area count
    def feature_sa(self,df, df_col, target,features):
        # Keeping the sub areas' name, their mean price and frequency (count)
        sa_feature_list = [sa for sa in features if "sa" in sa]
        lst = []
        for col in sa_feature_list:
            sa_triger = df[col]==1
            sa = df.loc[sa_triger, df_col].to_list()[0]
            x = df.loc[sa_triger, target]
            lst.append( (sa, np.mean(x), df[col].sum()) )
        return lst
    
    # function to scale the data
    def data_scale(self,data,df_col):
        # Standard scaling for surface
        sc = StandardScaler(with_std=True, with_mean=True)
        data[df_col] = sc.fit_transform(data[[df_col]])
        return data
    
    def save_to_csv(self,df):
        
        df.to_csv(config.get('DATA', 'processed_data_dir'), index=False) 