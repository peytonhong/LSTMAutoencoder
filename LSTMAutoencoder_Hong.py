# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:58:18 2018
LSTM Autoencoder
@author: HyoSung Hong
"""

import tensorflow as tf
import numpy as np

seq_length = 3
input_size = 1
hidden_size = 10

inputs=[]
predinputs=[]
for i in range(-50,50):
    input_ = [[i],[i+1],[i+2]]
    predinput_ = [[i+3],[i+4],[i+5]]
    inputs.append(input_)
    predinputs.append(predinput_)
inputs = np.array(inputs, dtype=np.float32,)
predinputs = np.array(predinputs, dtype=np.float32,)

testX=np.array([[[50],[51],[52]]],dtype=np.float32)
predtestX=np.array([[[53],[54],[55]]],dtype=np.float32)

X = tf.placeholder(tf.float32, [None, seq_length, input_size]) # for time_major = False
X_predictor = tf.placeholder(tf.float32, [None, seq_length, input_size])

# Encoder layer configuration
with tf.variable_scope("encoder"):
    encoder = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_encoder_cells = tf.contrib.rnn.MultiRNNCell([encoder for _ in range(1)])
    _encoder_output, _encoder_state = tf.nn.dynamic_rnn(multi_encoder_cells, X, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)

# Decoder layer configuration
with tf.variable_scope("decoder"):
    decoder_inputs = tf.zeros_like(X) # for decoder zero input
    decoder = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_decoder_cells = tf.contrib.rnn.MultiRNNCell([decoder for _ in range(1)])
    _decoder_output, _decoder_state = tf.nn.dynamic_rnn(multi_decoder_cells, decoder_inputs, initial_state=_encoder_state, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)
    
    _decoder_weight = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1, dtype=tf.float32))
    decoder_bias = tf.Variable(tf.constant(0.1, shape=[input_size], dtype=tf.float32))
    decoder_weight = tf.tile(tf.expand_dims(_decoder_weight, 0), [tf.shape(X)[0],1,1]) # to match the shape with output. [hidden,input_size]->[batch,hidden,input]
    
    decoder_outputs = tf.matmul(_decoder_output, decoder_weight) + decoder_bias
    
    X_reverse = tf.reverse(X, axis=[1]) # reverse the time_step's order. (ex: x1,x2,x3 -> x3,x2,x1)

# Predictor layer configuration
with tf.variable_scope("predictor"):
    predictor_inputs = tf.zeros_like(X) # for decoder zero input
    predictor = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_predictor_cells = tf.contrib.rnn.MultiRNNCell([predictor for _ in range(1)])
    _predictor_output, _predictor_state = tf.nn.dynamic_rnn(multi_predictor_cells, predictor_inputs, initial_state=_encoder_state, dtype=tf.float32)

    _predictor_weight = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1, dtype=tf.float32))
    predictor_bias = tf.Variable(tf.constant(0.1, shape=[input_size], dtype=tf.float32))
    predictor_weight = tf.tile(tf.expand_dims(_predictor_weight, 0), [tf.shape(X)[0],1,1]) # to match the shape with output. [hidden,input_size]->[batch,hidden,input]
    
    predictor_outputs = tf.matmul(_predictor_output, predictor_weight) + predictor_bias

# loss calculation
decoder_loss = tf.reduce_mean(tf.square(decoder_outputs - X_reverse))
predictor_loss = tf.reduce_mean(tf.square(predictor_outputs - X_predictor))
loss = decoder_loss + predictor_loss
train = tf.train.AdamOptimizer().minimize(loss)

# session open
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training step
    for i in range(50000):
        _, step_loss = sess.run([train, loss], feed_dict={X: inputs, X_predictor: predinputs})
        
        if step_loss < 1e-3:
            break
        
        if i%500 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))
    
    # Test step
    loss, dec, pred = sess.run([loss, decoder_outputs, predictor_outputs], feed_dict={X: testX, X_predictor: predtestX})
    print("Test loss: {}".format(loss))




