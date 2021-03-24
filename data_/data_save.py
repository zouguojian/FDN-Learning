# -- coding: utf-8 --

import os
import csv

def go_though(open_file):
    for root, dirs, files in os.walk(r''+str(open_file)):
        for file in files:
            # # 获取文件所属目录
            # print(root)
            # 获取文件路径
            print(os.path.join(root, file))

def re_hour(hour):
    if len(str(hour))<2:return '0'+str(hour)
    else:return str(hour)

def re_month(month):
    if len(str(month))<2:return '0'+str(month)
    else:return str(month)

def re_day(day):
    if len(str(day))<2:return '0'+str(day)
    else:return str(day)

from datetime import datetime
def re_time(time_list):    #[year,month,day,hour]
    time =''.join([time_list[i]+'-' for i in range(len(time_list))]).strip('-')
    time = datetime.strptime(time, '%Y-%m-%d-%H')
    return time

import numpy as np
def witer_(open_file,day,month,year,format,writer):
    for d in range(day, 32):
        if os.path.exists(open_file + year + re_month(month) + re_day(d) + format):
            read = pd.read_csv(open_file + year + re_month(month) + re_day(d) + format)
            print(list(np.reshape(sites[:,0],newshape=(-1))))
            data=read[list(sites[:,0])]
            # return read.loc[read['城市'] == '上海'].values
            # print(open_file + year + re_month(month) + re_day(d) + format)

def go_though(open_file,day,month,year,format,write_file,write_name):
    file = open(write_file+write_name, 'w', encoding='utf-8')
    writer=csv.writer(file)
    writer.writerow(data_colum)
    for m in range(month,13):
        if day!=1:
            witer_(open_file, day, m, year, format, writer)
            day=1
        else:
            witer_(open_file, day, m, year, format, writer)
    file.close()
    return

import pandas as pd
def read_site(file):
    read=pd.read_excel(file)
    print(list(read.keys()))
    return read.loc[read['城市']=='上海'].values

data_colum=['time','site','AQI','PM2.5','PM2.5_24h','PM10','PM10_24h','SO2','SO2_24h','NO2','NO2_24h','O3','O3_24h','O3_8h','O3_8h_24h','CO','CO_24h']
open_file='/Users/guojianzou/Downloads/站点_20200101-20201231/china_sites_'
open_site='/Users/guojianzou/Downloads/站点列表-2021.01.01起.xlsx'
write_file='/Users/guojianzou/Downloads/'
write_name='2020.csv'
day=5
month=1
year='2020'

#读取站点信息，包括：['监测点编码', '监测点名称', '城市', '经度', '纬度', '对照点']
sites=read_site(open_site)
print(sites)

# 遍历数据源，并开始存储
go_though(open_file,day,month,year,'.csv',write_file,write_name)
