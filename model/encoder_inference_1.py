# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:09:26 2018

@author: Administrator
"""
import tensorflow as tf
class AutoEncoder(object):
    def __init__(self,inputs,Layer1_Num,Layer2_Num,Layer3_Num):
        self.X=inputs
        self.Layer1_Num=Layer1_Num
        self.Layer2_Num=Layer2_Num
        self.Layer3_Num=Layer3_Num
    def encoderWeight(self):
        with tf.variable_scope('Encoder_weight1',reuse=tf.AUTO_REUSE):
            layer1_weight=tf.get_variable('layer1',[self.X.shape[1],self.Layer1_Num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer2_weight=tf.get_variable('layer2',[self.Layer1_Num,self.Layer2_Num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer3_weight=tf.get_variable('layer3',[self.Layer2_Num,self.Layer3_Num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        return (layer1_weight,layer2_weight,layer3_weight)
    def encoderBias(self):
        with tf.variable_scope('Encoder_bias1',reuse=tf.AUTO_REUSE):
            layer1_bias=tf.get_variable('bias1',[self.Layer1_Num],
                                        initializer=tf.constant_initializer(0.1))
            layer2_bias=tf.get_variable('bias2',[self.Layer2_Num],
                                        initializer=tf.constant_initializer(0.1))
            layer3_bias=tf.get_variable('bias3',[self.Layer3_Num],
                                        initializer=tf.constant_initializer(0.1)) 
        return (layer1_bias,layer2_bias,layer3_bias)

    def decoderWeight(self,Inputs):
        with tf.variable_scope('Decoder_weight1',reuse=tf.AUTO_REUSE):
            layer4_weight=tf.get_variable('layer4',[Inputs.shape[1],self.Layer2_Num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer5_weight=tf.get_variable('layer5',[self.Layer2_Num,self.Layer3_Num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer6_weight=tf.get_variable('layer6',[self.Layer3_Num,self.X.shape[1]],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        return (layer4_weight,layer5_weight,layer6_weight)
    def decoderBias(self):
        with tf.variable_scope('Decoder_bias1',reuse=tf.AUTO_REUSE):
            layer4_bias=tf.get_variable('bias4',[self.Layer2_Num],
                                        initializer=tf.constant_initializer(0.1))
            layer5_bias=tf.get_variable('bias5',[self.Layer3_Num],
                                        initializer=tf.constant_initializer(0.1))
            layer6_bias=tf.get_variable('bias6',[self.X.shape[1]],
                                        initializer=tf.constant_initializer(0.1))
        return (layer4_bias,layer5_bias,layer6_bias)
#    以上四个函数完成权重和偏置的初始化
    def encoder(self,X,layer1_weight,layer2_weight,layer3_weight,layer1_bias,layer2_bias,layer3_bias):
        layer1=tf.nn.relu(tf.matmul(self.X,layer1_weight)+layer1_bias)
        layer2=tf.nn.relu(tf.matmul(layer1,layer2_weight)+layer2_bias)
        layer3=tf.nn.relu(tf.matmul(layer2,layer3_weight)+layer3_bias)
        return layer3
#自编码部分
    def decoder(self,Inputs,layer4_weight,layer5_weight,layer6_weight,layer4_bias,layer5_bias,layer6_bias):
        layer4=tf.nn.relu(tf.matmul(Inputs,layer4_weight)+layer4_bias)
        layer5=tf.nn.relu(tf.matmul(layer4,layer5_weight)+layer5_bias)
        layer6=tf.nn.relu(tf.matmul(layer5,layer6_weight)+layer6_bias)
        return layer6
#解码部分