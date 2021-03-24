# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:55:13 2018

@author: Administrator
"""
import numpy as np
import math
class Gauss(object):
    def __init__(self,u=0.0, e=1):
        '''
        :param u:
        :param e:
        '''
        self.u=u
        self.e=e

    def gauss(self,x,y,target_x=0.0, target_y=0.0):
        '''
        :param x:
        :param y:
        :param target_x:
        :param target_y:
        :return: a value for site weighted
        '''
        x-=target_x
        y-=target_y
        A=(1.0/( 2.0 * 3.141592654 * self.e * self.e))
        B=math.exp(-((x-self.u) * (x-self.u) + (y-self.u) * (y-self.u))/(2 * self.e*self.e))
        return A*B

    def add_result(self,encoder_list,weights):
        sum_=sum([encoder_list[:,i,:]*weights[0] for i in range(len(weights))])
        return sum_

if __name__=='__main__':
    import numpy as np

    site=[[121.412, 31.1654],[118.775,32.0],[120.543,31.2994],
                                                                       [120.87,32.02],[120.294,31.56],[119.9633,31.762],
                                                                       [119.6707, 32.1875],[120.2072, 30.2111],[121.554, 29.8906],
                                                                       [120.576, 30.007],[120.1,30.8867],[120.726, 30.7478],
                                                                       [121.419, 28.6542], [122.3094, 29.9558]]
    site=np.array(site)
    gauss=Gauss()
    sites=[gauss.gauss(x=s[0],y=s[1],target_x=site[0,0],target_y=site[0,1]) for s in site]
    print(np.array(sites)/sum(sites))

