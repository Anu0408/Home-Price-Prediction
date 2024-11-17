
import os
import sys
import os.path as path
import configparser
config = configparser.RawConfigParser()
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
src_directory=os.path.abspath(path.join(__file__,"../../../"))
sys.path.append(src_directory)
from src.constants import *
import src.utils.homePriceUtil


class DataClean:
    def __init__(self):
         self.config = config.read(CONFIG_FILE_PATH)
         
    def read_data(self):
        return pd.read_excel(config.get('DATA', 'local_data_file'))     
        
     # rename a column
    def rename_col(self,data, column_name, new_column_name):
      data.rename(columns={"Propert Type":"Property Type"}, inplace=True)
      return data


    # to drop a specific row from a column
    def drop_val(self,data,column,row):
        data = data[ data[column]!=row]
        return data


    # data cleaning function for Property Area in Sq. Ft
    def splitSums(self, e, flag=True):
        """
        Gives the total number of bedrooms / property area
        params :
            e : string, either the number of rooms or property area
            flag : boolean, True : number of bedrooms, False : property area
        return :
            float, number of bedrooms / Property Area
        """
        try :
            e = str(e).lower()
            e = re.sub(r"[,;@#?!&$+]+\ *", " ", e)
            e = re.sub(r"[a-z]+", " ", e)
            e = re.sub(r"\s\s", "", e)

            s2list = e.strip().split()
            sumList = sum(float(e) for e in s2list)
                
            # Computing the mean value for property area that look like 
            # e.g. '1101 to 1113'
            e_norm = sumList if flag else sumList/len(s2list)
            return e_norm

        except :
            return np.nan

    
    # function to normalise the data
    def normaliseProps(self, df ) :
        """
        Extracts the number of rooms from 'Property Type' columns and mean values for 
        "Property Area in Sq. Ft."
        Params :
            data : Pandas dataframe, the input data
        Returns :
            Pandas dataframe
        """
        data = df.copy()
       # data["Property Type"] = data["Property Type"].apply(splitSums)
        #data["Property Area in Sq. Ft."] = \
        #data["Property Area in Sq. Ft."]\
        #        .apply( lambda x : splitSums(x, False) )
        
        return data
        
        
    # outliers handling
    def computeUpperFence(self, df_col, up=True ):
        """
        Computes the upper/lower fence for a given column.
        Params:
            df_col: Pandas series, dataframe column
            up: boolean, True for upper fence, False for lower fence
        Return:
            upper/lower fence value : float
        """
       
        iqr = float(df_col.quantile(.75)) - float(df_col.quantile(.25)) # inter quartile range
        if up:
            return float(df_col.quantile(.75) + iqr*1.5)
        return float(df_col.quantile(.25) - iqr*1.5 )  


    ###########################################

    ## dealing with the NAN values ### 

    def compute_fill_rate(self,df):
        """
        Computing the rate of non-NaNs for each column
        Params :
            df : Pandas dataframe, input data
        Return :
            Pandas dataframe
        """
        fr = pd.DataFrame(1-df.isnull().sum().values.reshape(1,-1)/df.shape[0], 
                            columns=df.columns)
        return fr


    def plot_fill_rate(self, df ) : 
        """
        Plot the fill rate
        df : Pandas dataframe, input data
        """
        fill_rate = pd.DataFrame(1-df.isnull().sum().values.reshape(1,-1)/df.shape[0], 
                            columns=df.columns)
        fig, ax = plt.subplots( figsize=(18,18) )
        sns.barplot(data=fill_rate, orient="h")
        ax.set_title( "Fill rate for columns", fontsize=28 )
        ax.set(xlim=(0, 1.))
        plt.show()
        

    def drop_empty_axis(self, df, minFillRate, axis=1 ) :
        """
        Drops axes that do not meet the minimum non-Nan rate
        Params :
            df : Pandas dataframe
            minFillRate : float, minimum filled fraction [0,1]
            axis : int, 1 for column, 0 for row
        Returns :
            Pandas dataframe 
        """
        i = 0 if axis==1 else 1 
        return df.dropna( axis=axis, thresh=int(df.shape[i]*minFillRate) )



    ### Regularising the categorical columns ##

    def print_uniques(self,cols, df):
        for col in cols:
            list_unique = df[col].unique()
            list_unique.sort()
            print(col, ":\n", list_unique)
            print("Number of unique categories:", len(list_unique))
            print("--------------------")



    # change values to binary and multi-categorical columns

    def reg_catvar(self,df, cols):
        for col in cols:
            df[col] = df[col].apply(lambda x: str(x).lower())
        return (cols)
    
    def save_to_csv(self,df):
        
        df.to_csv(config.get('DATA', 'clean_data_dir'), index=False) 