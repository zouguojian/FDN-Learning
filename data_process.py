# -- coding: utf-8 --


from sklearn.ensemble import GradientBoostingRegressor

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np

url='/Users/guojianzou/Documents/博士相关论文/Housing prices/data/归一化2.xlsx'
reader=pd.read_excel(io=url)
keys=reader.keys()
# print(list(reader.keys()))

data=reader.values[:,5:].astype(dtype=np.float32)


# print(reader.values[0,2:])
print('！！！----------------begin-------------------！！！')
print('the training data shape is : ', data.shape)

def re_index(observed_v=None, predicted_v=None):
    error = np.array(observed_v) - np.array(predicted_v)
    average_error = np.mean(np.fabs(error))
    print("mae error is : %f" % (average_error))
    #                            print(test_y_)
    rmse_error = np.sqrt(np.mean(np.square(np.array(observed_v) - np.array(predicted_v))))
    print("rmse error is : %f" % (rmse_error))
    cor = np.mean(np.multiply((observed_v - np.mean(observed_v)),
                              (predicted_v - np.mean(predicted_v)))) / (
                  np.std(predicted_v) * np.std(observed_v))
    #                           COR.append(cor)
    print('correlation coefficient is: %f' % (cor))
    return average_error,rmse_error, cor

def figure_show():
    return


def normalization(data):
    for i in range(data.shape[1]):
        _range = np.max(data[:,i]) - np.min(data[:,i])
        data[:,i]=(data[:,i] - np.min(data[:,i])) / _range
    return data


def generator_data(data):
    # data=normalization(data)
    train_x,train_y=data[:,2:],data[:,0]
    return train_x,train_y,data

# '''
# X, y = make_regression(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

train_x,train_y,data=generator_data(data)

reg = LinearRegression()
print(reg)
# for i in range(10):
reg.fit(train_x,train_y)

print('the whole parameter of the model : ',reg.get_params())
# GradientBoostingRegressor(random_state=0)

pre=reg.predict(train_x)
# print('Predict regression target for x :', pre)
# print(pre.shape)
r=reg.score(train_x, train_y)
print('Return the coefficient of determination R2 of the prediction : ',r)
re_index(observed_v=train_y,predicted_v=pre)

print(keys[7:])
print(reg.coef_)

feature_importance=reg.coef_

# feature_importance=(feature_importance/feature_importance.max())
sorted_idx=np.argsort(feature_importance)
pos=np.arange(sorted_idx.shape[0])+1.0

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/Library/Fonts/Songti.ttc')

plt.figure()

plt.barh(pos,feature_importance[sorted_idx],align='center')
print(train_x.shape, train_y.shape)
print(sorted_idx,len(sorted_idx))
plt.yticks(pos,keys[7:][sorted_idx],FontProperties=font)
plt.xlabel('relative')
plt.title('coefficient value')
plt.show()