import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
import utils.datapreprocessing as dp
import utils.visualization as vis
import utils.dataloader as dl

if __name__ == '__main__':
    # Load data
    file_path = './datasets/Anomalous_Data.csv'
    data = dl.dataLoad(file_path)
    
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
    # vis.visDistOfData(data, 3)


    """ Data Scaling """
    # Data Scaling with standardScaler
    std_df = dp.data_scaling("std", data, '서울')
    # Data Scaling with minmaxScaler
    mm_df = dp.data_scaling("minmax", data, '서울')
    # Data Scaling with robustScaler
    rb_df = dp.data_scaling("robust", data, '서울')

    print("Standard scaler")
    dp.findKBestFeature(std_df, data['서울'])
    print("Minmax Scaler")
    dp.findKBestFeature(mm_df, data['서울'])
    print("Robust Scaler")
    dp.findKBestFeature(rb_df, data['서울'])

    