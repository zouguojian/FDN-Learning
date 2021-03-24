# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:35:03 2018

@author: butany
"""

import tensorflow as tf
import Encoder_Inference_1
import Encoder_Inference_2
import Encoder_Inference_3
import Gauss_Inference
import LSTM_Inference
import testSet
import numpy as np

tf.reset_default_graph()
batch_size=64
learning_rate=0.0005
out_num=1

epochs=100

stride_size=100
windows_size=64
time_size=3

shanghai_stations=9
hangzhou_stations=11
suzhou_staions=8

layer_one=64
layer_two=128
layer_three=256

hidden_layer_num=1
SH=np.array([[121.412],[31.1654]])
HZ=np.array([[120.211,120.063,119.026,120.348,120.127,120.19,120.157,120.12,120.301,120.27,120.088],
            [30.21,30.2747,29.625,30.3058,30.2456,30.2692,30.2897,30.3119,30.4183,30.1819,30.1808]])
SZ=np.array([[120.561,120.628,120.591,120.596,120.613,120.543,120.669,120.641],
            [31.2472,31.2864,31.3019,31.3264,31.2703,31.2994,31.3097,31.3708]])
def SHantoEncode(X):
    antoEncoder=Encoder_Inference_1.AutoEncoder(X,layer_one,layer_two,layer_three)
    (layer1_weight,layer2_weight,layer3_weight)=antoEncoder.encoderWeight()
    (layer1_bias,layer2_bias,layer3_bias)=antoEncoder.encoderBias()
    Inputs=antoEncoder.encoder(X,layer1_weight,layer2_weight,layer3_weight,layer1_bias,layer2_bias,layer3_bias)
    (layer4_weight,layer5_weight,layer6_weight)=antoEncoder.decoderWeight(Inputs)
    (layer4_bias,layer5_bias,layer6_bias)=antoEncoder.decoderBias()
    result=antoEncoder.decoder(Inputs,layer4_weight,layer5_weight,layer6_weight,layer4_bias,layer5_bias,layer6_bias)
    return (Inputs,result)
def HZantoEncode(X):
    antoEncoder=Encoder_Inference_2.AutoEncoder(X,layer_one,layer_two,layer_three)
    (layer1_weight,layer2_weight,layer3_weight)=antoEncoder.encoderWeight()
    (layer1_bias,layer2_bias,layer3_bias)=antoEncoder.encoderBias()
    Inputs=antoEncoder.encoder(X,layer1_weight,layer2_weight,layer3_weight,layer1_bias,layer2_bias,layer3_bias)
    (layer4_weight,layer5_weight,layer6_weight)=antoEncoder.decoderWeight(Inputs)
    (layer4_bias,layer5_bias,layer6_bias)=antoEncoder.decoderBias()
    result=antoEncoder.decoder(Inputs,layer4_weight,layer5_weight,layer6_weight,layer4_bias,layer5_bias,layer6_bias)
    return (Inputs,result)
def SZantoEncode(X):
    antoEncoder=Encoder_Inference_3.AutoEncoder(X,layer_one,layer_two,layer_three)
    (layer1_weight,layer2_weight,layer3_weight)=antoEncoder.encoderWeight()
    (layer1_bias,layer2_bias,layer3_bias)=antoEncoder.encoderBias()
    Inputs=antoEncoder.encoder(X,layer1_weight,layer2_weight,layer3_weight,layer1_bias,layer2_bias,layer3_bias)
    (layer4_weight,layer5_weight,layer6_weight)=antoEncoder.decoderWeight(Inputs)
    (layer4_bias,layer5_bias,layer6_bias)=antoEncoder.decoderBias()
    result=antoEncoder.decoder(Inputs,layer4_weight,layer5_weight,layer6_weight,layer4_bias,layer5_bias,layer6_bias)
    return (Inputs,result)
def train():
    (SH_test,SZ_test,HZ_test)=testSet.Dataset()
    X1=tf.placeholder(tf.float32,[None,shanghai_stations*7],name="X1")
    X2=tf.placeholder(tf.float32,[None,hangzhou_stations*7],name="X2")
    X3=tf.placeholder(tf.float32,[None,suzhou_staions*7],name="X3")
    Y=tf.placeholder(tf.float32,[batch_size,1],name="Y")
    keep_prob=tf.placeholder("float")
#    上海,杭州，苏州天气自编码正向传播
    (SH_mid_result,SH_end_result)=SHantoEncode(X1)
    (HZ_mid_result,HZ_end_result)=HZantoEncode(X2)
    (SZ_mid_result,SZ_end_result)=SZantoEncode(X3)
    
#    高斯函数的使用,加权的过程
    SH_Gauss=Gauss_Inference.Gauss(SH)
    g=SH_Gauss.gauss(HZ)
    SH_mid_result=SH_Gauss.addResult(SH_mid_result,g,HZ_mid_result)
    g=SH_Gauss.gauss(SZ)
    SH_mid_result=SH_Gauss.addResult(SH_mid_result,g,SZ_mid_result)
    
#    LSTM的正向传播
    SH_mid_result=tf.reshape(SH_mid_result,shape=(batch_size,time_size,SH_mid_result.shape[1]),name='shape')
    lstm=LSTM_Inference.LSTM(layer_three,hidden_layer_num,time_size,out_num)
    result=lstm.full_Connect(SH_mid_result,batch_size,keep_prob)
    
    saver=tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
#        测试整个模型
        SH_test_low=0*shanghai_stations
        HZ_test_low=0*hangzhou_stations
        SZ_test_low=0*suzhou_staions
        SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
        HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
        SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
#        数据位置的复原
        for epoch in range(10):
            while(SH_test_hight<=SH_test.shape[0]):
                SH_X=SH_test[SH_test_low:SH_test_hight,2:9]
                HZ_X=HZ_test[HZ_test_low:HZ_test_hight,2:9]
                SZ_X=SZ_test[SZ_test_low:SZ_test_hight,2:9]
                SH_X=np.reshape(SH_X,[-1,shanghai_stations*7])
                HZ_X=np.reshape(HZ_X,[-1,hangzhou_stations*7])
                SZ_X=np.reshape(SZ_X,[-1,suzhou_staions*7])
                print(SH_X.shape)
                print(HZ_X.shape)
                print(SZ_X.shape)
#                标签以上海师范大学为预测目标
                label=list()
                for line in range(windows_size):
                    if str(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2][1])=='Shangshida':
#                        print(SH_train_low+(time_size*(line+1))*shanghai_stations)
                        label.append(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2,2:3])
                label=np.array(label)
                print(label.shape)
                s,_=sess.run((result),
                             feed_dict={X1:SH_X,X2:HZ_X,X3:SZ_X,Y:label,keep_prob:1})
                print("After %d epochs ,the entire cross_entropy is： %f"%(epoch,s))

                SH_test_low=stride_size*shanghai_stations+SH_test_low
                HZ_test_low=stride_size*hangzhou_stations+HZ_test_low
                SZ_test_low=stride_size*suzhou_staions+SZ_test_low
                SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
                print(SH_test_hight)
                HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
                SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
            SH_test_low=epoch*shanghai_stations
            HZ_test_low=epoch*hangzhou_stations
            SZ_test_low=epoch*suzhou_staions
            SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
            HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
            SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size 
                
def main(argv=None):
    train()

if __name__ == '__main__':
    main()