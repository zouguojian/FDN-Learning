# -- coding: utf-8 --

import tensorflow as tf
from model.auto_encoder import Auto_encoder
from model.hyparameter import parameter
from model.gauss_inference import Gauss
from model.lstm_inference import LSTM
import argparse
import model.test_set as test_set
import model.train_set as train_set
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

para=parameter(argparse.ArgumentParser())
hp=para.get_para()

def figure_show(observed_v=None, predicted_v=None, epoch=0):
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.ylim(0,200)
    #                    test_y_是真实值,蓝色的为真实值
    plt.plot(observed_v, color='darkblue', marker='*', linestyle=':', label='actual value')
    #                test_y_out是预测值，红色的为预测值
    plt.plot(predicted_v, color='darkred', marker='*', linestyle=':', label='predicted value')
    # plt.plot(aqi_v, color='orange', marker='*', linestyle=':', label='aqi')
    # 让图例生效
    #                        plt.legend(loc='upper right')
    plt.xlabel("Time(hours)", fontsize=17)
    plt.ylabel("PM2.5(ug/m3)", fontsize=17)
    # x_major_locator = MultipleLocator(20)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(50)
    # # 把y轴的刻度间隔设置为10，并存在变量里
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # # 把y轴的主刻度设置为10的倍数
    # plt.xlim(-0.5, 11)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(0, 200)
    # plt.xlim(0,160)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.title("The prediction of PM2.5  (epochs =" + str(epoch) + ")", fontsize=17)
    plt.show()

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

class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, hp.input_features], name="input_x")
        self.y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.x_=tf.placeholder(tf.float32,[len(hp.city_list),None,hp.input_features])

        self.x_lstm = tf.placeholder(tf.float32, [None, hp.layer_dict[-1]], name="input_x")
        self.keep_prob = tf.placeholder("float")
        self.auto_list = [None for _ in range(len(hp.city_list))]
        self.auto_e=[None for _ in range(len(hp.city_list))]
        self.pre_train_op=[None for _ in range(len(hp.city_list))]
        self.pre_loss=[None for _ in range(len(hp.city_list))]
        for city_i in range(len(hp.city_list)):
            self.auto_e[city_i], self.pre_train_op[city_i], self.pre_loss[city_i] = self.pre_train(city=hp.city_list[city_i],
                                                                                                   input=self.x,
                                                                                                   layer_dict=hp.layer_dict)

        self.auto_list = [None for _ in range(len(hp.city_list))]
        self.model()
        self.saver = tf.train.Saver(max_to_keep=1)

    def anti_encoder(self, city, input, layer_dict):
        return

    def model(self):
        '''
        :return:
        '''

        #encoder for each city
        for city_i in range(len(hp.city_list)):
            self.auto_list[city_i], _, _ = self.pre_train(city=hp.city_list[city_i],input=self.x_[city_i],layer_dict=hp.layer_dict)

        encoder_list=self.auto_list

        # gauss
        gauss = Gauss()
        city_sites=np.array(hp.city_sites)
        weights=[gauss.gauss(x=s[0],y=s[1],target_x=city_sites[0,0],target_y=city_sites[0,1]) for s in city_sites]
        self.gauss_result=gauss.add_result(encoder_list,weights)

        # lstm
        mid_result = tf.reshape(self.gauss_result, shape=(hp.batch_size, hp.input_length, hp.layer_dict[-1]), name='shape')
        lstm = LSTM(input=mid_result, batch_size=hp.batch_size, layer_num=hp.hidden_layer, nodes=hp.hidden_size)
        self.pre=lstm.prediction(hp.output_length)

        #define the model loss and operator in the whole training processing, and update the model weights or parameters
        self.cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pre)))
        self.train_op = tf.train.AdamOptimizer(hp.learning_rate).minimize(self.cross_entropy)

    def pre_train(self, city, input, layer_dict):
        '''
        :param city:
        :param input:
        :param layer_dict:
        :return: [encoder ouput, train_op]
        '''

        with tf.variable_scope(name_or_scope=city, reuse=tf.AUTO_REUSE):
            auto_object = Auto_encoder(x=input, layer_dict=layer_dict)
            loss_ = auto_object.loss()
            pre_train_op = tf.train.AdamOptimizer(hp.pre_learning_rate).minimize(loss=loss_)
        return auto_object.e, pre_train_op, loss_

    def test(self):
        test_data = test_set.getstart(file_path=hp.file_test)
        test_data = np.array(test_data)
        test_data = np.array(test_data[:, 2:], dtype=np.float32)
        # evaluate the whole model
        with tf.Session() as sess:
            # self.saver.restore(sess,save_path='weights/pollutant.ckpt-0')
            model_file = tf.train.latest_checkpoint('weights/')
            self.saver.restore(sess, model_file)
            predict, observed, test_y, target_index = list(), list(), list(), 0  # 0 represent the beginning of the target city :shanghai
            while (target_index + len(hp.city_list) * (hp.input_length + hp.output_length + 1) <= test_data.shape[0]):
                # to obtain the encoder result for each city
                test_x_ = []
                for city_i in range(len(hp.city_list)):
                    test_x = []
                    for i in range(city_i + target_index, test_data.shape[0], len(hp.city_list)):
                        test_x.append(test_data[i])
                        if city_i == 0 and len(test_x) % hp.input_length == 0 and i + len(hp.city_list) * (
                                hp.output_length + 1) <= test_data.shape[0]:  # shanghai
                            y = [test_data[k, 1] for k in
                                 range(i + len(hp.city_list), i + len(hp.city_list) * (hp.output_length + 1),
                                       len(hp.city_list))]
                            test_y.append(y)
                        if len(test_x) == hp.batch_size * hp.input_length:
                            break
                    test_x_.append(test_x)
                if len(test_y) != hp.batch_size: break
                # to obtain the whole prediction model results
                pre = sess.run(self.pre, feed_dict={self.x_: test_x_, self.y: np.reshape(np.array(test_y), newshape=[-1,
                                                                                                                     hp.output_length])})
                predict.append(pre)
                observed.append(test_y)
                test_y = list()
                target_index += hp.pre_step * len(hp.city_list) * hp.batch_size
        predict = np.reshape(np.array(predict, dtype=float), newshape=[-1])
        observed = np.reshape(np.array(observed, dtype=float), newshape=[-1])
        # print(predict.shape,observed.shape)
        re_index(observed, predict)
        figure_show(observed,predict,100)
        return

    def evaluate(self):
        test_data= test_set.getstart(file_path=hp.file_test)
        test_data=np.array(test_data)
        test_data=np.array(test_data[:,2:],dtype=np.float32)
        # evaluate the whole model
        with tf.Session() as sess:
            # self.saver.restore(sess,save_path='weights/pollutant.ckpt-0')
            model_file = tf.train.latest_checkpoint('weights/')
            self.saver.restore(sess, model_file)
            predict, observed, test_y, target_index = list(), list(), list(), 0  # 0 represent the beginning of the target city :shanghai
            while (target_index + len(hp.city_list) * (hp.input_length + hp.output_length + 1) <= test_data.shape[0]):
                # to obtain the encoder result for each city
                test_x_=[]
                for city_i in range(len(hp.city_list)):
                    test_x = []
                    for i in range(city_i + target_index, test_data.shape[0], len(hp.city_list)):
                        test_x.append(test_data[i])
                        if city_i == 0 and len(test_x) % hp.input_length == 0 and i + len(hp.city_list)* (hp.output_length+1)<=test_data.shape[0]:  # shanghai
                            y = [test_data[k, 1] for k in range(i + len(hp.city_list), i + len(hp.city_list) * (hp.output_length + 1),len(hp.city_list))]
                            test_y.append(y)
                        if len(test_x) == hp.batch_size * hp.input_length:
                            break
                    test_x_.append(test_x)
                if len(test_y) != hp.batch_size: break
                # to obtain the whole prediction model results
                pre = sess.run(self.pre, feed_dict={self.x_:test_x_,self.y: np.reshape(np.array(test_y),newshape=[-1,hp.output_length])})
                predict.append(pre)
                observed.append(test_y)
                test_y=list()
                target_index += hp.pre_step*len(hp.city_list)*hp.batch_size
        predict=np.reshape(np.array(predict,dtype=float),newshape=[-1])
        observed=np.reshape(np.array(observed,dtype=float),newshape=[-1])
        # print(predict.shape,observed.shape)
        re_index(observed,predict)

    def train(self):
        '''
        this function contains pre train and train steps
        :return:
        '''
        train_data= train_set.getstart(file_path=hp.file_train)
        train_data=np.array(train_data)
        train_data=np.array(train_data[:,2:],dtype=np.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = datetime.datetime.now()

            # pre- training processing
            for epoch in range(hp.pre_epoch):
                loss_=[]
                for city_i in range(len(hp.city_list)):
                    train_x = []
                    for i in range(city_i, len(train_data), len(hp.city_list)):
                        train_x.append(train_data[i][2:])
                        if len(train_x)==hp.batch_size:
                            l, _ = sess.run((self.pre_loss[city_i],self.pre_train_op[city_i]), feed_dict={self.x:np.array(train_x)})
                            loss_.append(l)
                            train_x=[]
                    print("after %d epoch,the self loss of city %s is : %f." % (epoch,hp.city_list[city_i] ,sum(loss_)/len(loss_)))

            # training the whole model, and test the model on each epoch
            min_loss=1000000
            for epoch in range(hp.epoch):
                loss_,train_y, target_index = list(),list(), 0 # 0 represent the beginning of the target city :shanghai
                while(target_index+len(hp.city_list)*(hp.input_length+hp.output_length+1)<=train_data.shape[0]):
                    # to obtain the encoder result for each city
                    train_x_=[]
                    for city_i in range(len(hp.city_list)):
                        train_x = []
                        for i in range(city_i+target_index, train_data.shape[0], len(hp.city_list)):
                            train_x.append(train_data[i])
                            if city_i==0 and len(train_x)%hp.input_length==0 and i + len(hp.city_list)* (hp.output_length+1)<=train_data.shape[0]: #shanghai
                                y=[train_data[k,1] for k in range(i+len(hp.city_list), i + len(hp.city_list)* (hp.output_length+1),len(hp.city_list))]
                                train_y.append(y)
                            if len(train_x) == hp.batch_size * hp.input_length:
                                break
                        train_x_.append(train_x)
                    if len(train_y)!=hp.batch_size:break
                    # to obtain the whole prediction model results
                    l,_=sess.run((self.cross_entropy,self.train_op),feed_dict={self.x_:np.array(train_x_),
                                                                               self.y:np.reshape(np.array(train_y),newshape=[-1,hp.output_length])})
                    loss_.append(l)
                    print("in the current %d index, the training loss is : %f." % (target_index, l))
                    train_y=list()
                    target_index+=hp.step * len(hp.city_list)*hp.batch_size
                print("after %d epoch,the minimum whole pollutant concentration prediction loss is : %f." % (epoch, min_loss))
                if min_loss>sum(loss_) / len(loss_):
                    min_loss=sum(loss_) / len(loss_)
                    self.saver.save(sess, save_path='weights/pollutant.ckpt', global_step=epoch)
                    self.evaluate()

            end_time = datetime.datetime.now()
            total_time = end_time - start_time
            print("total runing times is : %f" % total_time.total_seconds())

if __name__ == '__main__':
    model=Model()
    # model.train()
    model.test()
    # model.evaluate()