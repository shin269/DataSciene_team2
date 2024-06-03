import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
import utils.datapreprocessing as dp
import utils.visualization as vis
import utils.dataloader as dl
import utils.evaluation as evl

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

    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    scaled_df = pd.DataFrame(np.random.rand(462, 5), columns=[f'feature_{i}' for i in range(5)])
    data = pd.DataFrame({'서울': np.random.rand(462)})

    # Define feature, label
    feature = scaled_df.reset_index(drop=True)
    label = data['서울'].reset_index(drop=True)

    # Select KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = []

    n_iter = 0
    for train_index, test_index in kfold.split(feature):  
        x_train, x_test = feature.iloc[train_index], feature.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        # Initialized model
        lr = LinearRegression()
        rf = RandomForestRegressor()

        # Learning model
        lr.fit(x_train, y_train)
        rf.fit(x_train, y_train)

        # Predict
        lr_pred = lr.predict(x_test)
        rf_pred = rf.predict(x_test)
        
        # Assessment model accuracy
        lr_mse = np.round(mean_squared_error(y_test, lr_pred), 4)
        rf_mse = np.round(mean_squared_error(y_test, rf_pred), 4)
        
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]
        
        print('\n#{0} lrCross validation MSE : {1}, Number of train data : {2}, Number of test data : {3}'
            .format(n_iter, lr_mse, train_size, test_size))
        print('#{0} lr_Cross validation index : {1}'.format(n_iter, test_index))
        cv_mse.append(lr_mse)
        
        print('\n#{0} rf Cross validation MSE : {1}, Number of train data : {2}, Number of test data : {3}'
            .format(n_iter, rf_mse, train_size, test_size))
        print('#{0} rf Cross validation index : {1}'.format(n_iter, test_index))
        cv_mse.append(rf_mse)
        
        n_iter += 1

    # Prnint MSE 
    print('\nMean lr Cross validation MSE:', round(np.mean(cv_mse[::2]),4))
    print('Mean of rf Cross validation MSE:', round(np.mean(cv_mse[1::2]),4))
        