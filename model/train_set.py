# -*- coding: utf-8 -*-
import csv

def getstart(file_path):
    data=[]
    csvFile=open(file_path,'r', errors='ignore')
    reader=csv.reader(csvFile)
    for line in reader:data.append(line)
    return data