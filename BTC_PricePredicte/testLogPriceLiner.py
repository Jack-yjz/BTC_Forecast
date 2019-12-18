# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import datasets
import json
import math
import datetime
import numpy as np

def time_long(timeStamp):
    """
    计算时间差
    :param timeStamp: 时间戳
    :return: 相差的天数
    """
    timeStamp = int(str(timeStamp)[:-3])
    dateArray = datetime.datetime.fromtimestamp(timeStamp)
    dd = dateArray.strftime("%Y-%m-%d")
    dd = datetime.datetime.strptime(dd, "%Y-%m-%d")
    d1 = datetime.datetime(2009, 1, 3)
    day_num = (dd - d1).days
    return int(abs(day_num))


def analysisJsonData():
    print('解析数据')
    dataMat = []; labelMat = []; origData = []
    # 读取数据
    with open(r'btc_price.txt', 'r') as f:
        datas = json.load(f)
    
    for dic in datas['data']['items']:
        dataMat.append([math.log(1.0, 10), math.log(time_long(dic['timestamp']), 10)])  # 开始加入的1.0就是X0，对应参数W0
        labelMat.append(dic['close'])
        origData.append(time_long(dic['timestamp']))

    return dataMat, labelMat, origData

def main():
    dataMat, labelMat, origData = analysisJsonData()
    labelMats = []
    for i in labelMat:
        labelMats.append(math.log(i,10))
    X_train, X_test, y_train, y_test = train_test_split(dataMat, labelMats, test_size=0.2, random_state=1)
    
    from sklearn.linear_model import Lasso


    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
    print(lasso00001.coef_)
    print(lasso00001.intercept_)


    y_pred = lasso00001.predict(X_test)
    from sklearn import metrics
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('MSE:', MSE)       #MSE: 0.11876567909701435
    print('RMSE:', RMSE)     #RMSE: 0.3446239676763854
    

    # 使用别人的模型测试预测数据，利用平方差和均方差根
    w0,w1 = np.array(X_test).T
    X_test_other = w1
    y_pred_other = 5.84*X_test_other-17.01
    MSE_2 = metrics.mean_squared_error(y_test, y_pred_other)
    RMSE_2 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_other))
    print('MSE_2:', MSE_2)     #MSE_2: 0.11966383807561337
    print('RMSE_2:', RMSE_2)   #RMSE_2: 0.34592461328389656
    

    #plt.semilogy(origData, labelMat,lw=2)
    # 在纵坐标为log(币价)，横坐标为log(币龄)，训练出的结果是5.72 和-16.627
    '''
    plt.plot(origData, np.array(labelMats),'red')
    price_log_list_original = []
    for line in origData:
        price_log_list_original.append(5.7281125 * math.log(line, 10) - 16.62788617732808)

    plt.plot(origData, price_log_list_original, label='line 1 oneSelf')
    '''

    # 画曲线，纵坐标还是币价，是一个曲线价格曲线，横坐标还是天数
    plt.plot(origData, 10**np.array(labelMats), 'red')
    price_list = []
    for line in origData:
        price_list.append(10**(5.82589664*math.log(line,10)-16.951201463753417))
    plt.semilogy(origData, price_list,lw=2)

    # 和别人的对比
    other_price_list = []
    for line in origData:
        other_price_list.append(10 ** (5.84 * math.log(line, 10) - 17.01))
    plt.semilogy(origData, other_price_list, lw=2)
    plt.show()


if __name__ == '__main__':
    main()

