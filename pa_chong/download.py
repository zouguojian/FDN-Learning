# -- coding: utf-8 --
import urllib3
import json

url='http://www.pm25.in/api/querys/aqi_details.json'

response1 = urllib3.PoolManager()

re=response1.request('GET',url=url,fields={'city':'shanghai','token':'5j1znBVAsnSf5xQyNQyq'})
re=re.data
re = re.decode('utf-8')
print(re)
