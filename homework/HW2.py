# powered by:@御河DE天街

# 导库
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression

path = os.path.dirname(os.getcwd())
sys.path.append(path)

from common.filesio import FilesIO

# 读取数据并划分测试集，训练集
data=pd.read_csv(FilesIO.getHomeworkData('Credit.csv'))
X=data[['Limit','Rating']].values
y=data['Balance'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# 数据标准化
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test=(X_test-np.mean(X_test))/np.std(X_test)

# 进行线性回归
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)

# 结果分析
y_pred = linreg.predict(X_test)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
R2 = linreg.score(X_test, y_test)
print('%.3f' % MSE)
print('%.3f' % RMSE)
print('%.3f' % R2)

# 画出测试值在测试集上的对比图
plt.figure()
plt.plot(y_test,y_pred, 'bo',label="predict") ## bo: 蓝色的圈
plt.plot(y_test,y_test, 'r--',label="true") ## r--: 红色虚线
plt.legend(loc="lower right") #显示图中的标签
plt.xlabel("True value")
plt.ylabel('Predicted value')
plt.show()

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b',label="predict") ## 预测值用蓝色线表示
plt.plot(range(len(y_pred)), y_test, 'r',label="test") ## 真实值用红色线表示
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("Index of Products")
plt.ylabel('Sales')
plt.show()


# 如果引入分类变量

# 读取数据并划分测试集，训练集
data=pd.read_csv(FilesIO.getHomeworkData('Credit.csv'))
X=data[['Limit','Rating','Income','Cards','Education','Gender','Student']]
X=pd.get_dummies(X,drop_first=True)
y=data['Balance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# 进行线性回归
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print('截距项theta0为：%.3f'% linreg.intercept_)
for i in range(len(linreg.coef_)):
    print('回归系数theta'+str(i+1)+'为：%.3f'% linreg.coef_[i])

# 结果分析
y_pred = linreg.predict(X_test)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
R2 = linreg.score(X_test, y_test)
print('均方误差为: %.3f' % MSE)
print('均方根误差为: %.3f' % RMSE)
print('R2值为: %.3f' % R2)

# 画出测试值在测试集上的对比图
plt.figure()
plt.plot(y_test,y_pred, 'bo',label="predict") ## bo: 蓝色的圈
plt.plot(y_test,y_test, 'r--',label="true") ## r--: 红色虚线
plt.legend(loc="lower right") #显示图中的标签
plt.xlabel("True value")
plt.ylabel('Predicted value')
plt.show()

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b',label="predict") ## 预测值用蓝色线表示
plt.plot(range(len(y_pred)), y_test, 'r',label="test") ## 真实值用红色线表示
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("Index of Products")
plt.ylabel('Sales')
plt.show()


# 导库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics

# 新代价函数
def cost_fnc(x,y,*theta):
    import numpy as np
    m=x.shape[0]
    X=np.c_[np.ones(x.shape[0]),x]
    Theta=np.array(list(theta))
    cost=0
    for i in range(m):
        h_theta=np.dot(X[i],Theta.reshape(X[i].shape[0]))
        cost=cost+(h_theta-y[i])**2
    average_cost=1/(2*m)*cost
    return average_cost

# 新梯度函数
def gradient_fnc(x,y,*theta):
    import numpy as np
    m=x.shape[0]
    X=np.c_[np.ones(x.shape[0]),x]
    Theta=np.array(list(theta))
    dj_theta=np.zeros(Theta.shape)
    for i in range(m):
        h_theta=np.dot(X[i],Theta.reshape(X[i].shape[0]))
        dj_theta=dj_theta+(h_theta-y[i])*X[i]
    dj_theta=dj_theta/m
    return dj_theta

# 新梯度下降算法函数
def gradient_descent(x, y, alpha, max_iter,tol, *theta_init):
    
    """
    theta_init：theta_i的初始值
    alpha: 学习率
    max_iter: 循环/迭代的最大次数
    tol: 收敛条件:如果当前一步的代价J_current与上一步的代价J_old之差的绝对值<tol,则停止
    cost_fnc: 代价函数
    gradient_fnc: 梯度函数
    """

    Js=[]
    theta=list(theta_init)
    theta_hist=[]
    i=0
    while i<max_iter:
        J_old=cost_fnc(x,y,*theta)
        dj_theta=gradient_fnc(x,y,*theta)
        theta=list(map(lambda x,y:x-alpha*y,theta,dj_theta))
        J_current=cost_fnc(x,y,*theta)
        Js.append(J_current)
        theta_hist.append(theta)
        if abs(J_current-J_old)<tol:break
        i=i+1
    return  theta,Js,theta_hist

# 读取数据并划分测试集，训练集
data=pd.read_csv(FilesIO.getHomeworkData('Credit.csv'))
X=data[['Limit','Rating','Income','Cards','Education','Gender','Student']]
X=pd.get_dummies(X,drop_first=True)
y=data['Balance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

alpha = 0.01
max_iter = 100000   #学习率以及收敛条件需要反复调整
tol = 0.00000001
theta0_init = 0
theta1_init = 0
theta2_init = 0
theta3_init = 0
theta4_init = 0
theta5_init = 0
theta6_init = 0
theta7_init = 0

theta,Js,theta_hist  = gradient_descent(X_train, y_train, alpha, max_iter, tol,
                                        theta0_init,theta1_init,theta2_init,theta3_init,theta4_init,
                                        theta5_init,theta6_init,theta7_init)

print(theta)

# 结果分析
X_test=np.c_[np.ones(X_test.shape[0]),X_test]
y_pred=[np.dot(X_test[i],theta) for i in range(len(X_test))]
MSE_y = metrics.mean_squared_error(y_pred, y_test)
RMSE_y = np.sqrt(MSE_y)
R2=1-((y_test - y_pred)** 2).sum()/((y_test - y_test.mean()) ** 2).sum()
print('%.3f' % MSE_y)
print('%.3f' % RMSE_y)
print('%.3f' % R2)

# 画出测试值在测试集上的对比图
plt.figure()
plt.plot(y_test,y_pred, 'bo',label="predict") ## bo: 蓝色的圈
plt.plot(y_test,y_test, 'r--',label="true") ## r--: 红色虚线
plt.legend(loc="lower right") #显示图中的标签
plt.xlabel("True value")
plt.ylabel('Predicted value')
plt.show()

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b',label="predict") ## 预测值用蓝色线表示
plt.plot(range(len(y_pred)), y_test, 'r',label="test") ## 真实值用红色线表示
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("Index of Products")
plt.ylabel('Sales')
plt.show()