# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:58:18 2018
LSTM Autoencoder
@author: HyoSung Hong
Input: Vehicle's attitude signals that comprise 3-axis gyro, acceleration 
       as well as roll, pitch, yaw angle.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

seq_length = 10
input_size = 1
output_size = 1
hidden_size = 200
depth = 1

def MinMaxScaler(data):
	''' Min Max Normalization

	Parameters
	----------
	data : numpy.ndarray
		input data to be normalized
		shape: [Batch size, dimension]

	Returns
	----------
	data : numpy.ndarry
		normalized data
		shape: [Batch size, dimension]

	References
	----------
	.. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

	'''
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
	# noise term prevents the zero division
	return numerator / (denominator + 1e-7)

# training inputs
print('Preparing input data...')
# Open, High, Low, Volume, Close
xy = np.loadtxt('_sim_flag3_R_P_Y_Cpp.dat')
#xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy[:, -3]
y = xy[:, -3]  # Close as label

if x.ndim == 1:
    x = np.expand_dims(x,1) # shape matching for input dim is 1
if y.ndim == 1:
    y = np.expand_dims(y,1) # shape matching for output dim is 1

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - 2*seq_length):
	_x = x[i : i+seq_length]
	_y = y[i+seq_length : i+2*seq_length] # for prediction
	dataX.append(_x)
	dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[:train_size]), np.array(dataX[train_size:])
trainY, testY = np.array(dataY[:train_size]), np.array(dataY[train_size:])


X = tf.placeholder(tf.float32, [None, seq_length, input_size]) # for time_major = False
X_predictor = tf.placeholder(tf.float32, [None, seq_length, output_size])


# Encoder layer configuration
with tf.variable_scope("encoder"):
    encoder = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_encoder_cells = tf.contrib.rnn.MultiRNNCell([encoder for _ in range(depth)])
    _encoder_state = multi_encoder_cells.zero_state(tf.shape(X)[0], dtype=tf.float32)
    _encoder_output, _encoder_state = tf.nn.dynamic_rnn(multi_encoder_cells, X, initial_state=_encoder_state, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)


# Decoder layer configuration
with tf.variable_scope("decoder"):
    decoder_inputs = tf.zeros_like(X) # for decoder zero input
    decoder = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_decoder_cells = tf.contrib.rnn.MultiRNNCell([decoder for _ in range(depth)])
    _decoder_output, _decoder_state = tf.nn.dynamic_rnn(multi_decoder_cells, decoder_inputs, initial_state=_encoder_state, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)
    
    _decoder_weight = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1, dtype=tf.float32))
    decoder_bias = tf.Variable(tf.constant(0.1, shape=[output_size], dtype=tf.float32))
    decoder_weight = tf.tile(tf.expand_dims(_decoder_weight, 0), [tf.shape(X)[0],1,1]) # to match the shape with output. [hidden,input_size]->[batch,hidden,input]

    decoder_outputs = tf.matmul(_decoder_output, decoder_weight) + decoder_bias
    
    X_reverse = tf.reverse(X, axis=[1]) # reverse the time_step's order. (ex: x1,x2,x3 -> x3,x2,x1)

# Predictor layer configuration
with tf.variable_scope("predictor"):
    predictor_inputs = tf.zeros_like(X) # for decoder zero input
    predictor = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    multi_predictor_cells = tf.contrib.rnn.MultiRNNCell([predictor for _ in range(depth)])
    _predictor_output, _predictor_state = tf.nn.dynamic_rnn(multi_predictor_cells, predictor_inputs, initial_state=_encoder_state, dtype=tf.float32)

    _predictor_weight = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1, dtype=tf.float32))
    predictor_bias = tf.Variable(tf.constant(0.1, shape=[output_size], dtype=tf.float32))
    predictor_weight = tf.tile(tf.expand_dims(_predictor_weight, 0), [tf.shape(X)[0],1,1]) # to match the shape with output. [hidden,input_size]->[batch,hidden,input]
    
    predictor_outputs = tf.matmul(_predictor_output, predictor_weight) + predictor_bias

# loss calculation
decoder_loss = tf.reduce_mean(tf.square(decoder_outputs - X[:,:,-3:]))
predictor_loss = tf.reduce_mean(tf.square(predictor_outputs - X_predictor))
#loss = decoder_loss + predictor_loss
loss = decoder_loss
train = tf.train.AdamOptimizer().minimize(loss)

# session open
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training step
    for i in range(1501):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, X_predictor: trainY})

        if step_loss < 1e-5:
            break
        
        if i%10 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))
    # Test step
    loss, dec, pred = sess.run([loss, decoder_outputs, predictor_outputs], feed_dict={X: trainX, X_predictor: trainY})
    print("Test loss: {}".format(loss))


check_index = 1000 # check index among test data
test_time = np.linspace(0,0.01*seq_length*2,seq_length*2)
test_time = np.reshape(test_time, [2*seq_length,1])
plt.plot(test_time, np.array([trainX[check_index,:,:], trainY[check_index,:,:]]).flatten(), label='test data')
plt.plot(test_time[:seq_length], dec[check_index,:,:], label='decoded data')
plt.plot(test_time[seq_length:], pred[check_index,:,:], label='predicted data')
plt.grid()
plt.legend(loc='lower right')


