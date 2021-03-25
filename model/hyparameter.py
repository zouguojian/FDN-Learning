# -- coding: utf-8 --
import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--city_list',type=list,default=['ShangHai','NanJing','SuZhou',
                                                                  'NanTong','WuXi','ChangZhou',
                                                                  'ZhenJiang','HangZhou','NingBo',
                                                                  'ShaoXing','HuZhou','JiaXing',
                                                                  'TaiZhou','ZhouShan'],help='city name')
        self.parser.add_argument('--city_sites',type=list,default=[[121.412, 31.1654],[118.775,32.0],[120.543,31.2994],
                                                                   [120.87,32.02],[120.294,31.56],[119.9633,31.762],
                                                                   [119.6707, 32.1875],[120.2072, 30.2111],[121.554, 29.8906],
                                                                   [120.576, 30.007],[120.1,30.8867],[120.726, 30.7478],
                                                                   [121.419, 28.6542], [122.3094, 29.9558]],help='city site')
        self.parser.add_argument('--layer_dict',type=list,default=[64,128,256],help='layer dict')
        self.parser.add_argument('--pre_learning_rate', type=float, default=0.001, help='pre learning rate')
        self.parser.add_argument('--input_features', type=int, default=7, help='input feature')
        self.parser.add_argument('--pre_step', type=int, default=1, help='pre step')
        self.parser.add_argument('--pre_epoch', type=int, default=0, help='pre epoch')

        self.parser.add_argument('--epoch', type=int, default=100, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out')
        self.parser.add_argument('--city_num', type=int, default=14, help='total number of city')
        self.parser.add_argument('--features', type=int, default=7, help='numbers of the feature')

        self.parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')
        self.parser.add_argument('--output_num', type=int, default=7, help='numbers of the feature')
        self.parser.add_argument('--full_size', type=int, default=7, help='numbers of the feature')

        self.parser.add_argument('--input_length', type=int, default=3, help='input length')
        self.parser.add_argument('--output_length', type=int, default=1, help='output length')

        self.parser.add_argument('--training_set_rate', type=float, default=1.0, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.0, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=1.0, help='test set rate')

        self.parser.add_argument('--file_train', type=str,
                                 default='data/train_around_weather.csv',
                                 help='training set file address')
        self.parser.add_argument('--file_val', type=str,
                                 default='/Users/guojianzou/Documents/program/shanghai_weather/val_around_weather.csv',
                                 help='validate set file address')
        self.parser.add_argument('--file_test', type=str,
                                 default='data/around_weathers_2017_7_test.csv',
                                 help='test set file address')

        self.parser.add_argument('--file_out', type=str, default='model/ckpt', help='file out')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)