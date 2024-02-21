# 从'Credit.csv'中读取数据
import os
import sys
import pandas as pd
import seaborn as sns

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from common.filesio import FilesIO
credit = pd.read_csv(FilesIO.getHomeworkData("Credit.csv"))
credit.head()
credit.info()
credit.describe()
credit.hist(figsize=(20,15))
sns.set_theme(rc={'figure.figsize':(12,8)})
sns.set_style('ticks')
sns.boxplot(x='Gender', y='Balance', data=credit)
sns.boxplot(x='Student', y='Balance', data=credit)
sns.boxplot(x='Married', y='Balance', data=credit)
sns.boxplot(x='Ethnicity',y='Balance',data=credit)
sns.boxplot(x='Education', y='Balance', data=credit)
sns.boxplot(x='Cards', y='Balance', data=credit)
credit.plot(kind='scatter',x='Income',y='Balance',figsize=(10,10))
credit.plot(kind='scatter',x='Limit',y='Balance',figsize=(10,10))
credit.plot(kind='scatter',x='Rating',y='Balance',figsize=(10,10))
credit.plot(kind='scatter',x='Age',y='Balance',figsize=(10,10))
