# -*- coding: utf-8 -*-
import csv

def getstart(file_path):
    data=[]
    csvFile=open(file_path,'r', errors='ignore')
    reader=csv.reader(csvFile)
    for line in reader:data.append(line)
    return data

# import numpy as np
# data=getstart('/Users/guojianzou/Documents/program/shanghai_weather/around_weathers_2017_7_test.csv')
# data=np.array(data)
# print(data[:,2:9].shape)



