import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False
path = os.path.dirname(os.getcwd())
sys.path.append(path)

from common.filesio import FilesIO

data = pd.read_csv(FilesIO.getHomeworkData("credit_card.csv"), 
                   index_col=0, encoding="utf-8")
data.head()

# 去除空值
data = data.dropna()
data.head()

# 对所有变量画出直方图
data.hist(figsize=(20,20))
plt.show()

# 进行k均值聚类
WSCC = []
for i in range(2, 11):
    km = KMeans(n_clusters=i, n_init='auto', random_state=0, algorithm='elkan').fit(data)
    WSCC.append(km.inertia_)
x = np.arange(2, 11)
plt.figure(figsize=(10, 6))
plt.plot(x, WSCC, 'o-')
plt.title('Searching for Elbow')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# 使用最优的K进行拟合
model = KMeans(n_clusters=6, random_state = 0, algorithm='elkan').fit(data)
model.labels_

# 查看每一类的个数
data["Clusters"] = model.labels_
plt.figure(figsize=(10,6))
pl = sns.countplot(x = data["Clusters"])
pl.set_title("Distribution of The Clusters")
plt.show()

# 绘制箱线图
sns.set_style('ticks') 
col_names = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'PURCHASES_FREQUENCY',
          'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
          'CASH_ADVANCE_FREQUENCY', 'CREDIT_LIMIT']
plt.figure(figsize=(24,18))
for i, col in enumerate(col_names):
    plt.subplot(3,3,i+1)
    sns.boxplot(x = "Clusters", y = col, data = data)
    plt.title(col)
plt.show()