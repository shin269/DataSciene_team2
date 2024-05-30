import numpy as np
import pandas as pd
from typing import List
from sklearn import preprocessing

def label_encoder(df: pd.DataFrame, 
                  target: str
) -> pd.DataFrame:
    """ LabelEncoder for categorical data convert numerical data\n
    [Parmaeter]\n
    - df : Target dataframe
    - target: Target to encoding

    [Return] \n
    Label encoded pandas dataframe
    """
    
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    df[target] = lb.fit_transform(df[target])
    
    return df

def data_scaling(scaler: str, 
               df: pd.DataFrame, 
               target: str = None, 
               ignoreColumn : List[str] = None
) -> pd.DataFrame:
    """ Normalization function for feature except target column.\n
    Must to convert categorical data to numerical data before execution.\n

    [Parameter]
    1. scaler : type of scaling\n
        \t- std -> StandardScaler
        \t- minmax -> MinmaxScaler
        \t- robust -> RobustScaler
    2. df : Dataframe to scaling
    3. target : Target data for the model to learn
    4. ignoreColumn : Columns to exclude from scaling \n
    
    [Return]\n
    Scaled dataframe
    """
    if scaler == "std":
        scaler = preprocessing.StandardScaler()
    elif scaler == "minmax":
        scaler = preprocessing.MinMaxScaler()
    elif scaler == "robust":
        scaler = preprocessing.RobustScaler()
    else:
        print("Invalid scaler!")
        exit()

    # Save and drop ignore column
    if ignoreColumn != None:
        dropdata = df[ignoreColumn].copy()
        df = df.drop(columns = ignoreColumn)

    # Remove target data
    data = df.drop([target], axis=1).reset_index(drop=True)
    target_column = df[target].reset_index(drop=True)

    # Scalining data
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    
    # Add ignorecolumn to scaled data
    if ignoreColumn != None:   
        for col in ignoreColumn:
            scaled_df[col] = dropdata[col].values

    # Add target data
    scaled_df[target] = target_column

    return scaled_df

def engNum2Num(df: pd.DataFrame, 
          target : str
) -> List[float]:
    """ Converting English numeric notation to numbers """
    def split_alpha_numeric(s):
        import re
        match = re.match(r'(\d+\.\d+)(\D+)', s)
        if match:
            numbers = float(match.group(1))
            letters = match.group(2)
            if   letters == 'B': numbers *= 1000000000
            elif letters == 'M': numbers *= 1000000
            elif letters == 'K': numbers *= 1000
            else: return None, None
            return float(numbers), letters
        else:
            return None, None
    
    df = df[target]

    result = []
    for s in df:
        if pd.isna(s):
            result.append(pd.NA)
            continue

        n, _ = split_alpha_numeric(s)
        result.append(n)

    return result

def dropIQR(df: pd.DataFrame, 
        per : float = 0.25
) -> pd.DataFrame:
    """ Drop outlier use IQR \n
    df: Dataframe for drop outlier \n
    per: Percenatge of outlier range (default: 0.25)\n
    Return: Dataframe droped outlier. """
    # Set outlier range
    Q1 = df.quantile(per)
    Q3 = df.quantile(1-per)
    IQR = Q3 - Q1

    # Detect and drop outiler over IQR range
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df