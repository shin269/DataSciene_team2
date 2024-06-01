import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 데이터 불러오기
data = pd.read_excel('result.xlsx')

# 독립 변수와 종속 변수 설정 ('서울'과 '날짜' 특성 제외)
X = data.drop(['서울', '날짜'], axis=1)  # '서울', '날짜' 특성 제외
y = data['서울']  # '서울'을 타겟 변수로 사용

# 데이터 표준화
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 모든 특성에 대한 SelectKBest 적용
bestfeatures_all = SelectKBest(score_func=f_regression, k='all')
fit_all = bestfeatures_all.fit(X_standardized, y)

# 특성 이름과 점수를 데이터프레임으로 결합
dfcolumns_all = pd.DataFrame(X.columns)
dfscores_all = pd.DataFrame(fit_all.scores_)
featureScores_all = pd.concat([dfcolumns_all, dfscores_all], axis=1)
featureScores_all.columns = ['Specs', 'Score']  # 데이터프레임 컬럼 이름 지정

# 점수가 높은 순으로 정렬
featureScores_sorted = featureScores_all.sort_values(by='Score', ascending=False)

# 정렬된 데이터프레임 출력
print(featureScores_sorted)
