import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import datapreprocessing as dp
import matplotlib.pyplot as plt
import seaborn as sns

# 이 파일은 결측치를 추가한 파일입니다!
file_path = 'D:\Junseo\Downloads\pp\Anomalous_Data.csv'

data = pd.read_csv(file_path)
data = data.drop(columns=['날짜'])

# Null 값 확인
print(data.isnull().sum())

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import warnings
import datapreprocessing as dataprep
import visualization as vis
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우에서 한글 폰트 설정 예시
plt.rcParams['font.size'] = 12  # 기본 폰트 크기 설정

# Null 값 처리 (예: 평균 값으로 채우기)
data.fillna(data.mean(), inplace=True)

# 이상치 처리 (예: IQR을 이용한 제거)
data = dataprep.dropIQR(data) 

# Visualization that distribution of data calumns
vis.visDistOfData(data, 3)
