# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:00:09 2018

@author: butany
"""

import tensorflow as tf
import Encoder_Inference_1
import Encoder_Inference_2
import Encoder_Inference_3
import Gauss_Inference
import LSTM_Inference
import trainSet
import numpy as np
import testSet
import matplotlib.pyplot as plt
import datetime

tf.reset_default_graph()
batch_size=64
learning_rate=0.0005
out_num=1

epochs=101

stride_size=100
windows_size=64
time_size=3
#站点数目
shanghai_stations=9
hangzhou_stations=11
suzhou_staions=8

layer_one=128
layer_two=256
layer_three=512
hidden_layer=128
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
#    训练集的获取
    (SH_train,SZ_train,HZ_train)=trainSet.Dataset()
#    测试集的获取
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
    SH_cross_entropy=tf.sqrt(tf.reduce_mean(tf.square(X1-SH_end_result)))
    HZ_cross_entropy=tf.sqrt(tf.reduce_mean(tf.square(X2-HZ_end_result)))
    SZ_cross_entropy=tf.sqrt(tf.reduce_mean(tf.square(X3-SZ_end_result)))
    Sum_cross_entropy=SH_cross_entropy+HZ_cross_entropy+SZ_cross_entropy
    SH_train_op = tf.train.AdamOptimizer(learning_rate).minimize(SH_cross_entropy)
    HZ_train_op = tf.train.AdamOptimizer(learning_rate).minimize(HZ_cross_entropy)
    SZ_train_op = tf.train.AdamOptimizer(learning_rate).minimize(SZ_cross_entropy)
    Sum_train_op = tf.train.AdamOptimizer(learning_rate).minimize(Sum_cross_entropy)
    
#    高斯函数的使用,加权的过程
    SH_Gauss=Gauss_Inference.Gauss(SH)
    g=SH_Gauss.gauss(HZ)
    SH_mid_result=SH_Gauss.addResult(SH_mid_result,g,HZ_mid_result)
    g=SH_Gauss.gauss(SZ)
    SH_mid_result=SH_Gauss.addResult(SH_mid_result,g,SZ_mid_result)
    
#    LSTM的正向传播
    SH_mid_result=tf.reshape(SH_mid_result,shape=(batch_size,time_size,SH_mid_result.shape[1]),name='shape')
    lstm=LSTM_Inference.LSTM(hidden_layer,hidden_layer_num,time_size,out_num)
    result=lstm.full_Connect(SH_mid_result,batch_size,keep_prob)
    
    cross_entropy=tf.sqrt(tf.reduce_mean(tf.square(Y-result)))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    saver=tf.train.Saver()
#    数据集的划分
    SH_train_low=0*shanghai_stations
    HZ_train_low=0*hangzhou_stations
    SZ_train_low=0*suzhou_staions
    SH_train_hight=SH_train_low+windows_size*shanghai_stations
    HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations
    SZ_train_hight=SZ_train_low+windows_size*suzhou_staions
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time=datetime.datetime.now()
#       预训练自编码部分
        train_steps=0
        steps=0
        loss=list()
        for epoch in range(20):
            while(SH_train_hight<=SH_train.shape[0]):
                SH_X=SH_train[SH_train_low:SH_train_hight,2:9]
                HZ_X=HZ_train[HZ_train_low:HZ_train_hight,2:9]
                SZ_X=SZ_train[SZ_train_low:SZ_train_hight,2:9]
                SH_X=np.reshape(SH_X,[-1,shanghai_stations*7])
                HZ_X=np.reshape(HZ_X,[-1,hangzhou_stations*7])
                SZ_X=np.reshape(SZ_X,[-1,suzhou_staions*7])
               
                s,_=sess.run((Sum_cross_entropy,
                              Sum_train_op),feed_dict={X1:SH_X,X2:HZ_X,X3:SZ_X})
                print("After %d epochs and %d train_steps,the Sum_cross_entropy is： %f"%(epoch,train_steps,s))

                SH_train_low=SH_train_hight
                HZ_train_low=HZ_train_hight
                SZ_train_low=SZ_train_hight
                SH_train_hight=SH_train_low+windows_size*shanghai_stations
                HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations
                SZ_train_hight=SZ_train_low+windows_size*suzhou_staions
                train_steps+=1
            SH_train_low=0*shanghai_stations
            HZ_train_low=0*hangzhou_stations
            SZ_train_low=0*suzhou_staions
            SH_train_hight=SH_train_low+windows_size*shanghai_stations
            HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations 
            SZ_train_hight=SZ_train_low+windows_size*suzhou_staions 
#        训练整个模型
        SH_train_low=0*shanghai_stations
        HZ_train_low=0*hangzhou_stations
        SZ_train_low=0*suzhou_staions
        SH_train_hight=SH_train_low+windows_size*shanghai_stations*time_size
        HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations*time_size
        SZ_train_hight=SZ_train_low+windows_size*suzhou_staions*time_size
#        数据位置的复原
        for epoch in range(epochs):
            if epoch==0:
                Label=list()
                Predict=list()
#                count是用来限定测试集预测范围的
                count=0
                SH_test_low=0*shanghai_stations
                HZ_test_low=0*hangzhou_stations
                SZ_test_low=0*suzhou_staions
                SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
                HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
                SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
                while(SH_test_hight<=SH_test.shape[0]):
                    if count>=4:
                        break
                    SH_X=SH_test[SH_test_low:SH_test_hight,2:9]
                    HZ_X=HZ_test[HZ_test_low:HZ_test_hight,2:9]
                    SZ_X=SZ_test[SZ_test_low:SZ_test_hight,2:9]
                    SH_X=np.reshape(SH_X,[-1,shanghai_stations*7])
                    HZ_X=np.reshape(HZ_X,[-1,hangzhou_stations*7])
                    SZ_X=np.reshape(SZ_X,[-1,suzhou_staions*7])
#                标签以上海师范大学为预测目标
                    label=list()
                    for line in range(windows_size):
                        if str(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2][1])=='Shangshida':
#                        print(SH_train_low+(time_size*(line+1))*shanghai_stations)
                            label.append(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2,3:4])
                    label=np.reshape(np.array(label),[1,-1])[0]
                    s=sess.run((result),
                                 feed_dict={X1:SH_X,X2:HZ_X,X3:SZ_X,keep_prob:1})
                    s=np.reshape(s,[1,batch_size])[0]
                    for i in range(batch_size):
                        Label.append(float(label[i]))
                        Predict.append(s[i])
                    count+=1
                    SH_test_low=SH_test_hight
                    HZ_test_low=HZ_test_hight
                    SZ_test_low=SZ_test_hight
                    
                    SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
                    HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
                    SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
#                计算预测的误差，以及显示预测效果折线图
                Label=np.reshape(np.array(Label),[1,-1])[0,0:128]
                Predict=np.array(Predict)[0:128]
                error=Label-Predict
                average_Error=np.mean(np.fabs(error))
                print("After %d epochs and %d steps, MAE error is : %f"%(epoch,steps,average_Error))
#                            print(test_y_)
                RMSE_Error=np.sqrt(np.mean(np.square(np.array(Label)-np.array(Predict))))
                print("After %d epochs and %d steps, RMSE error is : %f"%(epoch,steps,RMSE_Error))
                cor=np.mean(np.multiply((Label-np.mean(Label)),
                                                   (Predict-np.mean(Predict))))/(np.std(Predict)*np.std(Label))
                print ('The correlation coefficient is: %f'%(cor))
                if epoch==0:
                        plt.figure() 
#                Label是真实值,蓝色的为真实值
                        plt.plot(Label,'b*:',label=u'actual value')  
#                Predict是预测值，红色的为预测值
                        plt.plot(Predict,'r*:',label=u'predicted value')
#                       让图例生效
#                       plt.legend()
                        plt.xlabel("Time(hours)",fontsize=17)  
                        plt.ylabel("PM2.5(ug/m3)",fontsize=17)  
                        plt.title("The prediction of PM2.5  (epochs ="+str(epoch)+")",fontsize=17)
            
            while(SH_train_hight<=SH_train.shape[0]):
                SH_X=SH_train[SH_train_low:SH_train_hight,2:9]
                HZ_X=HZ_train[HZ_train_low:HZ_train_hight,2:9]
                SZ_X=SZ_train[SZ_train_low:SZ_train_hight,2:9]
                SH_X=np.reshape(SH_X,[-1,shanghai_stations*7])
                HZ_X=np.reshape(HZ_X,[-1,hangzhou_stations*7])
                SZ_X=np.reshape(SZ_X,[-1,suzhou_staions*7])
#                print(SH_X.shape)
#                print(HZ_X.shape)
#                print(SZ_X.shape)
#                标签以上海师范大学为预测目标
                label=list()
                for line in range(windows_size):
                    if str(SH_train[SH_train_low+(time_size*(line+1))*shanghai_stations+2][1])=='Shangshida':
#                        print(SH_train_low+(time_size*(line+1))*shanghai_stations)
                        label.append(SH_train[SH_train_low+(time_size*(line+1))*shanghai_stations+2,3:4])
                label=np.array(label)
#                print(label.shape)
                Loss,_=sess.run((cross_entropy,train_op),
                             feed_dict={X1:SH_X,X2:HZ_X,X3:SZ_X,Y:label,keep_prob:0.5})
#                print("After %d epochs and %d steps,the entire cross_entropy is： %f"%(epoch,steps,s))
                if steps%100==0:
                    loss.append(Loss)
                SH_train_low=SH_train_hight
                HZ_train_low=HZ_train_hight
                SZ_train_low=SZ_train_hight
                
                SH_train_hight=SH_train_low+windows_size*shanghai_stations*time_size
                HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations*time_size
                SZ_train_hight=SZ_train_low+windows_size*suzhou_staions*time_size
                steps+=1
            SH_train_low=0*shanghai_stations
            HZ_train_low=0*hangzhou_stations
            SZ_train_low=0*suzhou_staions
            SH_train_hight=SH_train_low+windows_size*shanghai_stations*time_size
            HZ_train_hight=HZ_train_low+windows_size*hangzhou_stations*time_size
            SZ_train_hight=SZ_train_low+windows_size*suzhou_staions*time_size 
#            准备测试预测的数据集
            SH_test_low=0*shanghai_stations
            HZ_test_low=0*hangzhou_stations
            SZ_test_low=0*suzhou_staions
            SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
            HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
            SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
            if epoch%1==0:
                Label=list()
                Predict=list()
#                count是用来限定测试集预测范围的
                count=0
                while(SH_test_hight<=SH_test.shape[0]):
                    if count>=4:
                        break
                    SH_X=SH_test[SH_test_low:SH_test_hight,2:9]
                    HZ_X=HZ_test[HZ_test_low:HZ_test_hight,2:9]
                    SZ_X=SZ_test[SZ_test_low:SZ_test_hight,2:9]
                    SH_X=np.reshape(SH_X,[-1,shanghai_stations*7])
                    HZ_X=np.reshape(HZ_X,[-1,hangzhou_stations*7])
                    SZ_X=np.reshape(SZ_X,[-1,suzhou_staions*7])
#                标签以上海师范大学为预测目标
                    label=list()
                    for line in range(windows_size):
                        if str(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2][1])=='Shangshida':
#                        print(SH_train_low+(time_size*(line+1))*shanghai_stations)
                            label.append(SH_test[SH_test_low+(time_size*(line+1))*shanghai_stations+2,3:4])
                    label=np.reshape(np.array(label),[1,-1])[0]
                    s=sess.run((result),
                                 feed_dict={X1:SH_X,X2:HZ_X,X3:SZ_X,keep_prob:1})
                    s=np.reshape(s,[1,batch_size])[0]
                    for i in range(batch_size):
                        Label.append(float(label[i]))
                        Predict.append(s[i])
                    count+=1
                    SH_test_low=SH_test_hight
                    HZ_test_low=HZ_test_hight
                    SZ_test_low=SZ_test_hight
                    
                    SH_test_hight=SH_test_low+windows_size*shanghai_stations*time_size
                    HZ_test_hight=HZ_test_low+windows_size*hangzhou_stations*time_size
                    SZ_test_hight=SZ_test_low+windows_size*suzhou_staions*time_size
#                计算预测的误差，以及显示预测效果折线图
                Label=np.reshape(np.array(Label),[1,-1])[0]
                Predict=np.array(Predict)
                if epoch==100:
                    L=list()
                    P=list()
                    for i in range(Predict.shape[0]):
                        L.append(Label[i])
                        P.append(Predict[i])
#                    print(L,P)
                Label=Label[0:128]
                Predict=Predict[0:128]
                error=Label-Predict
                average_Error=np.mean(np.fabs(error))
                print("After %d epochs and %d steps, MAE error is : %f"%(epoch,steps,average_Error))
#                            print(test_y_)
                RMSE_Error=np.sqrt(np.mean(np.square(np.array(Label)-np.array(Predict))))
                print("After %d epochs and %d steps, RMSE error is : %f"%(epoch,steps,RMSE_Error))
                cor=np.mean(np.multiply((Label-np.mean(Label)),
                                                   (Predict-np.mean(Predict))))/(np.std(Predict)*np.std(Label))
                print ('The correlation coefficient is: %f'%(cor))
                if epoch==10 or epoch==20 or epoch==30 or epoch==40 or epoch==50 or epoch==60 or epoch==70 or epoch==80 or epoch==90 or epoch==100:
                        plt.figure() 
#                Label是真实值,蓝色的为真实值
                        plt.plot(Label,'b*:',label=u'actual value')  
#                Predict是预测值，红色的为预测值
                        plt.plot(Predict,'r*:',label=u'predicted value')
#                       让图例生效
#                       plt.legend()
                        plt.xlabel("Time(hours)",fontsize=17)  
                        plt.ylabel("PM2.5(ug/m3)",fontsize=17)  
                        plt.title("The prediction of PM2.5  (epochs ="+str(epoch)+")",fontsize=17)
#        print(loss)    
#            运行时间的差
        end_time=datetime.datetime.now()
        total_time=end_time-start_time
        print("Total runing times is : %f"%total_time.total_seconds())
def main(argv=None):
    train()

if __name__ == '__main__':
    main()