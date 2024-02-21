import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from common.filesio import FilesIO

# 读取数据
data = pd.read_csv(FilesIO.getHomeworkData("Titanic.csv"))
data1 = data.dropna()

# 可视化生存/未生存人数
plt.figure(figsize=(12,8))
survived_data = data["Survived"].value_counts()
survived_data = np.array(survived_data)
plt.bar(range(2),survived_data,color=["red","dodgerblue"],width=0.3,alpha=0.3)
plt.annotate(f'{survived_data[0]}', xy=(0, survived_data[0]), xytext=(0, survived_data[0]+10),ha='center', va='bottom', fontsize=14)
plt.annotate(f'{survived_data[1]}', xy=(1, survived_data[1]), xytext=(1, survived_data[1]+10),ha='center', va='bottom', fontsize=14)
plt.xticks([0,1],["未生存","生存"])
plt.xlabel("生存情况",fontdict={"fontsize":12})
plt.ylabel("人数",fontdict={"fontsize":12})
plt.title("乘客生存情况",fontdict={"fontsize":14})
plt.show()

# 定义函数 percentage_above_bar_relative_to_xgroup， 使得流失百分比显示在条形上方
def percentage_above_bar_relative_to_xgroup(ax):
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = f'{(100 * p.get_height() / total) :.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=14, ha='center', va='bottom')

# Pclass对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
ax1 = sns.countplot(x = "Pclass", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax1)
# 设置图片标题
plt.title('Survive by Pclass', fontsize=20)
plt.show()

# Sex对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
ax2 = sns.countplot(x = "Sex", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax2)
plt.title('Survive by Sex', fontsize=20)
plt.show()

# SibSp对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
ax3 = sns.countplot(x = "SibSp", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax3)
plt.title('Survive by SibSp', fontsize=20)
plt.show()

# Parch对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
ax4 = sns.countplot(x="Parch", hue="Survived", data=data1)
percentage_above_bar_relative_to_xgroup(ax4)
plt.title('Survive by Parch', fontsize=20)
plt.show()

# Embarked对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
ax5 = sns.countplot(x="Embarked", hue="Survived", data=data1)
percentage_above_bar_relative_to_xgroup(ax5)
plt.title('Survive by Embarked', fontsize=20)
plt.show()

# Age对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
sns.boxplot(x="Survived", y="Age", data=data1)
plt.title('Survive by Age', fontsize=20)
plt.show()

# Fare对生存的影响
sns.set_theme(rc={'figure.figsize':(12,8)})
sns.boxplot(x="Survived", y="Fare", data=data1)
plt.title('Survive by Age', fontsize=20)
plt.show()

feature_cols = ["Pclass","Age","SibSp","Parch","Fare"]
X = data1[feature_cols] # 特征
y = data1.Survived # 输出/目标变量 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1, stratify = y)
model = LogisticRegression(penalty = 'none')
model.fit(X_train,y_train)
coef = model.coef_[0]
exp_coef = np.exp(coef)
df = pd.DataFrame(exp_coef, 
             X.columns, 
             columns=['coef']).sort_values(by='coef', ascending=False)
print(df)
train_accuracy = model.score(X_train, y_train)
y_pred = model.predict(X_test) 
y_pred_prob = model.predict_proba(X_test) 

test_accuracy = model.score(X_test, y_test)
print("Accuracy", test_accuracy)

test_precision = metrics.precision_score(y_test, y_pred)
print("Precision", test_precision)

test_recall = metrics.recall_score(y_test, y_pred)
print("Recall:", test_recall)

test_f1 = metrics.f1_score(y_test, y_pred)
print("test_f1:", test_f1)