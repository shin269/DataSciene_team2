import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_excel('result.xlsx')

# 독립 변수와 종속 변수 설정 ('서울'과 '날짜' 특성 제외)
X = data.drop(['서울', '날짜'], axis=1)

# 데이터 표준화
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# K-평균 클러스터링 모델 생성 및 훈련
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_standardized)

# 클러스터 결과를 원본 데이터에 추가
data['Cluster'] = clusters

# 클러스터별로 데이터 시각화
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(data[data['Cluster'] == i]['Dubai'], data[data['Cluster'] == i]['서울'], color=colors[i], label=f'Cluster {i}')

plt.xlabel('Dubai')
plt.ylabel('서울')
plt.title('Clustering of Seoul Economic Data')
plt.legend()
plt.show()
