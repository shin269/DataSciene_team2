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

def fillDate(data_path : str,
             start_date : str,
             end_date : str,
             save_file_name : str,
             save_file_path : str = './',
             encoding_type : str = 'cp949',
             have_index : bool = False
) -> None:
    """ Fill empty date for data """
    data_type = data_path.split('.')[-1]

    # Data Load
    try:
        if   data_type == 'csv': df = pd.read_csv(data_path)
        elif data_type == 'xlsx': df = pd.read_excel(data_path)
    except:
        print("Invalid file type!")
        exit()

    # Create data frame of start to end date
    date_range = pd.date_range(start = start_date, end = end_date)
    date_range_formatted = date_range.strftime('%Y%m%d')

    # Fill empty date
    new_df = pd.DataFrame({'날짜': date_range_formatted})
    for _, row in new_df.iterrows():
        if row['날짜'] not in df['날짜'].values: 
            # Empty data check and store
            df = df.append(row, ignore_index=True)

    df.sort_values(by='날짜',inplace=True)
    df['날짜'] = df['날짜'].str.replace('-','')

    # Save file 
    save_path = save_file_path + save_file_name
    try:
        df.to_csv(save_path, encoding = encoding_type, index = have_index)
        print(f"Suceess save file to {save_path}")
    except:
        print("Fail save file!")
        exit()

def findKBestFeature(df : pd.DataFrame, 
                     target : pd.DataFrame, 
                     select_k : int = 10
) -> None:
    from sklearn.feature_selection import SelectKBest, f_regression

    X = df
    Y = target

    selector = SelectKBest(score_func=f_regression, k = select_k)
    X_new = selector.fit_transform(X,Y)

    selected_features = X.columns[selector.get_support()]
    scores = selector.scores_

    selected_features_scores = dict(zip(X.columns, scores))
    selected_features_sorted = sorted(selected_features_scores.items(), key = lambda item:item[1], reverse= True)

    print("\n[Selected Features and their scores]")
    cnt = 1
    for feature, score in selected_features_sorted:
        if feature in selected_features:
            print(f"{cnt}. {feature}: {score:.4f}")
            cnt += 1