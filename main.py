import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
import utils.datapreprocessing as dp
import utils.visualization as vis
import utils.dataloader as dl

if __name__ == '__main__':
    # Data path with darty data
    file_path = './datasets/Anomalous_Data.csv'

    # Load data
    data = dl.dataLoad(file_path)

    data = pd.read_csv(file_path)
    data = data.drop(columns=['날짜'])

    # Check number of null data
    print(data.isnull().sum())

    warnings.filterwarnings('ignore') # Error ignore
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Fonts of window for korean
    plt.rcParams['font.size'] = 12  # Set defualt font size

    # Fill null data use mean
    data.fillna(data.mean(), inplace=True)

    # Drop outlier use IQR
    data = dp.dropIQR(data) 

    # Visualization that distribution of data calumns
    vis.visDistOfData(data, 3)
