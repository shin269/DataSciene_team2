import pandas as pd

def dataLoad(file_path : str) -> pd.DataFrame:
    """ Data load fit filename extension """
    try:
        file_type = file_path.split('.')[-1]

        # 1. csv type            
        if file_type in "csv": 
            return pd.read_csv(file_path)
        # 2. excel type
        elif file_type in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods']:
            return pd.read_excel(file_path)
        else:
            print("Invalid data type!")
            return None
    except:
        print("Invalid file path")
        exit()