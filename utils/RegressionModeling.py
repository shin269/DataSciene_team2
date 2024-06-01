import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

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

# 점수가 높은 순으로 상위 10개 특성 선택
top_features = featureScores_all.nlargest(10, 'Score')['Specs']

# 선택된 특성으로 데이터 필터링
X_selected = X[top_features]

# 데이터를 훈련 세트와 테스트 세트로 분리 (70% 훈련, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# 선형 회귀 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 세트에 대한 예측
y_pred = model.predict(X_test)

# 모델 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 성능 출력
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
