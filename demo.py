import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['STsong']
matplotlib.rcParams['axes.unicode_minus'] = False



data = pd.read_csv("D:\python_project\some_code\Study\Machine_Learning\data\Titanic.csv")
data1 = data.dropna()


def percentage_above_bar_relative_to_xgroup(ax):
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = f'{(100 * p.get_height() / total) :.1f}%'
            ax.annotate(percentage, 
                        (p.get_x() + p.get_width() / 2, p.get_height()), 
                        size=14, ha='center', va='bottom')

plt.figure(figsize=(24, 16))
plt.subplot(2, 3, 1)
ax1 = sns.countplot(x = "Pclass", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax1)
plt.title('Survive by Pclass', fontsize=20)
plt.subplot(2, 3, 2)
ax2 = sns.countplot(x = "Sex", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax2)
plt.title('Survive by Sex', fontsize=20)
plt.subplot(2, 3, 3)
ax3 = sns.countplot(x = "SibSp", hue = "Survived", data = data1)
percentage_above_bar_relative_to_xgroup(ax3)
plt.title('Survive by SibSp', fontsize=20)
plt.subplot(2, 3, 4)
ax4 = sns.countplot(x="Parch", hue="Survived", data=data1)
percentage_above_bar_relative_to_xgroup(ax4)
plt.title('Survive by Parch', fontsize=20)
plt.subplot(2, 3, 5)
ax5 = sns.countplot(x="Embarked", hue="Survived", data=data1)
percentage_above_bar_relative_to_xgroup(ax5)
plt.title('Survive by Embarked', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()