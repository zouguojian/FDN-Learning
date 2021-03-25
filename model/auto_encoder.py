# -- coding: utf-8 --
import tensorflow as tf

class Auto_encoder(object):
    def __init__(self,x,layer_dict):
        '''
        :param x:
        :param layer_dict:
        e: encoder
        d: decoder
        '''
        self.x=x
        self.layer_dict=layer_dict
        self.layer_encoder=[None for _ in range(len(self.layer_dict))]
        self.layer_decoder=[None for _ in range(len(self.layer_dict))]
        self.e=self.encoder_layer()
        self.d=self.decoder_layer()

    def encoder_layer(self):
        '''
        :return: [batch*time_size,hidden_size]
        '''

        for i in range (len(self.layer_dict)):
            with tf.variable_scope(name_or_scope='encoder_layer'+str(i)):
                if i==0:
                    self.layer_encoder[i] = tf.layers.dense(inputs=self.x,
                                                     units=self.layer_dict[i],
                                                     activation=tf.nn.relu)
                else:
                    self.layer_encoder[i]=tf.layers.dense(inputs=self.layer_encoder[i-1],
                                                   units=self.layer_dict[i],
                                                   activation=tf.nn.relu)
        #     print('the encoder %d layer, and the layer output tensor shape is %s.'%(i+1,str(self.layer_encoder[i].shape)))
        # print('\n')
        return self.layer_encoder[-1]

    def decoder_layer(self):
        '''
        :return: [batch*time_size,input_size]
        '''

        for i in range (len(self.layer_dict)-1,-1,-1):
            with tf.variable_scope(name_or_scope='decoder_layer'+str(i)):
                if i==len(self.layer_dict)-1:
                    self.layer_decoder[i] = tf.layers.dense(inputs=self.layer_encoder[-1],
                                                     units=self.layer_dict[i-1],
                                                     activation=tf.nn.relu)
                elif i==0:
                    self.layer_decoder[i]=tf.layers.dense(inputs=self.layer_decoder[i+1],
                                                   units=self.x.shape[-1],
                                                   activation=tf.nn.relu)
                else:
                    self.layer_decoder[i]=tf.layers.dense(inputs=self.layer_decoder[i+1],
                                                   units=self.layer_dict[i-1],
                                                   activation=tf.nn.relu)
            # print('the decoder %d layer, and the layer output tensor shape is %s.' % (i + 1, str(self.layer_decoder[i].shape)))
        return self.layer_decoder[0]

    def loss(self):
        '''
        used to calculate the average value between input and output for the auto encoder
        :return:
        '''
        l=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x=self.x,y=self.d))))
        return l

if __name__=='__main__':
    with tf.Session() as sess:
        x=tf.placeholder(shape=[None,100],name='input_x',dtype=tf.float32)
        layer_dict_=[128,256,512]

        a=tf.Variable(initial_value=tf.random_normal(shape=(32,100),mean=0.01,stddev=1.0,dtype=tf.float32))
        auto = Auto_encoder(x,layer_dict=layer_dict_)
        sess.run(tf.global_variables_initializer())
        # e=sess.run(encoder,feed_dict={x:a})
        loss=sess.run(auto.loss(),feed_dict={x:sess.run(a)})
        print(a.shape, auto.d.shape)
        print(loss)