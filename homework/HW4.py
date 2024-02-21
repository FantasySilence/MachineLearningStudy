import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

matplotlib.rcParams['font.sans-serif'] = ['STSong']
matplotlib.rcParams['axes.unicode_minus'] = False
path = os.path.dirname(os.getcwd())
sys.path.append(path)

from common.filesio import FilesIO

data = pd.read_csv(FilesIO.getHomeworkData("IRIS.csv"))

sns.set_theme(rc={'figure.figsize':(15,12)})
sns.set_style('ticks')
for i, feature in enumerate(data.columns[:-1]):
    plt.subplot(221+i)
    sns.boxplot(x='species', y=feature, data=data)
plt.show()

# 划分测试集与训练集
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], 
                                                    test_size=0.3, random_state=0)

# 创建参数网格
param_grids = {
              "ccp_alpha": [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
              "max_depth": [1,2,3,4],
              "min_samples_leaf": [3,5,10,20,30,50]  
              }
# 寻找最优分类树
gs_cv = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grids, cv=5, scoring='accuracy')
gs_cv.fit(X_train, y_train)
for key, value in gs_cv.best_params_.items():
    print("最优的",key, ":", value)
clf_best = gs_cv.best_estimator_

plt.figure(figsize = [15,12])
feature_name = list(X_train.columns)
plot_tree(clf_best, filled=True, feature_names = feature_name, class_names = None)
plt.title("最优的决策树")
plt.show()

plt.figure(figsize = [15,12])
importances = clf_best.feature_importances_
weights = pd.Series(importances, index=feature_name)
print(weights.sort_values(ascending=False))
weights.sort_values().plot(kind = 'barh')
plt.title('特征重要性')
plt.show()

# 在测试集上进行预测，并计算相关指标
y_pred = clf_best.predict(X_test)
y_pred_prob = clf_best.predict_proba(X_test)

# 正确率
test_accuracy = clf_best.score(X_test, y_test)
print("Accuracy=%.3f"%test_accuracy)
# 精确率
test_precision = metrics.precision_score(y_test, y_pred, average = 'micro')
print("Precision=%.3f"%test_precision)
# 召回率
test_recall = metrics.recall_score(y_test, y_pred, average = 'micro')
print("Recall=%.3f"%test_recall)
# F1值
test_f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
print("test_f1=%.3f"%test_f1)


param_grids = {
               "n_estimators": [50,100,200,300],
               "max_features": ["sqrt", "log2", None]
              }
gs_cv1 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grids, cv=5, scoring='accuracy')
gs_cv1.fit(X_train, y_train)
for key, value in gs_cv1.best_params_.items():
    print("最优的",key, ":", value)
rf_best = gs_cv1.best_estimator_

plt.figure(figsize = [15,12])
importances = rf_best.feature_importances_
weights = pd.Series(importances, index=feature_name)
print(weights.sort_values(ascending=False))
weights.sort_values().plot(kind = 'barh')
plt.title('特征重要性')
plt.show()

y_pred = rf_best.predict(X_test)
y_pred_prob = rf_best.predict_proba(X_test)

# 正确率
test_accuracy = rf_best.score(X_test, y_test)
print("Accuracy=%.3f"%test_accuracy)
# 精确率
test_precision = metrics.precision_score(y_test, y_pred, average = 'micro')
print("Precision=%.3f"%test_precision)
# 召回率
test_recall = metrics.recall_score(y_test, y_pred, average = 'micro')
print("Recall=%.3f"%test_recall)
# F1值
test_f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
print("test_f1=%.3f"%test_f1)

