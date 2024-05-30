import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import datapreprocessing as dp


# """ 기준일 전체 날짜 가져오기 """
# date_range = pd.date_range(start = '2021-01-01', end = '2023-12-31')
# date_range_formatted = date_range.strftime('%Y-%m-%d')

# # 기준일 전체 날짜를 가지고 있는 데이터 프레임 생성
# new_df = pd.DataFrame({'날짜': date_range_formatted})

# """ 기준일 전체 날짜만큼 for문으로 비교하기 """
# for index, row in new_df.iterrows():
#     # 기준일 전체 날짜중에 특정 날짜가 존재하지 않으면
#     if row['날짜'] not in df['날짜'].values:
#         # 원래 데이터에 집어넣기
#         df = df.append(row, ignore_index=True)

# # 날짜 오름차순 정렬
# df.sort_values(by='날짜',inplace=True)

# # 날짜 데이터가 yyyy-mm-dd 형식이라면 '-' 제거하기
# df['날짜'] = df['날짜'].str.replace('-','')
# print(df) # 결과 출력하기

# # 새로운 파일로 저장
# df.to_csv('./KOSPI.csv', encoding='cp949', index=False)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # 이 파일은 결측치를 추가한 파일입니다!
# file_path = 'D:\Junseo\Downloads\pp\Anomalous_Data.csv'

# data = pd.read_csv(file_path)
# data = data.drop(columns=['날짜'])

# # Null 값 확인
# print(data.isnull().sum())

# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt
# import warnings
# import datapreprocessing as dataprep
# import visualization as vis
# warnings.filterwarnings('ignore')

# plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우에서 한글 폰트 설정 예시
# plt.rcParams['font.size'] = 12  # 기본 폰트 크기 설정

# # Null 값 처리 (예: 평균 값으로 채우기)
# data.fillna(data.mean(), inplace=True)

# # 이상치 처리 (예: IQR을 이용한 제거)
# data = dataprep.dropIQR(data) 

# # Visualization that distribution of data calumns
# vis.visDistOfData(data, 3)
