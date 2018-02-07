# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:58:18 2018
LSTM Autoencoder
@author: HyoSung Hong
Input: Vehicle's attitude signals that comprise 3-axis gyro, acceleration 
       as well as roll, pitch, yaw angle.
input sequence change: 
    before: [1 2 3] [2 3 4] [3 4 5]
    after:  [1 2 3] [4 5 6] [7 8 9]
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

seq_length = 20
pred_seq_length = 5 # must be smaller than seq_length
input_size = 1
output_size = 1
hidden_size = 500
depth = 1
decoderReverse = False # if True, decoder output sequence is reversed. e.g. x3, x2, x1
zero_input = True # input type of predictor layer
useSavedVariables = False # if False, training process is executed.
batch_mix = True

assert (seq_length >= pred_seq_length), "'pred_seq_length' must be smaller or equal to 'seq_length'!"

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
    #numerator = data - np.min(data, 0)
    #denominator = np.max(data, 0) - np.min(data, 0)
        
    angle_limit = 15 #[degree]
    numerator = data - (-angle_limit*np.pi/180)
    denominator = (angle_limit*np.pi/180) - (-angle_limit*np.pi/180)
    
    	
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)*2 - 1
    #return numerator / (denominator + 1e-7)

# training inputs
print('Preparing input data...')
data_train = np.loadtxt('_sim_flag3_R_P_Y_Cpp_jagal_train.dat')
data_test = np.loadtxt('_sim_flag3_R_P_Y_Cpp_jagal_test.dat')
#data_test[:,1:4] = data_test[:,1:4]*np.pi/180 # convert [deg] to [rad]
data_train = MinMaxScaler(data_train)
data_test = MinMaxScaler(data_test)

x_train, y_train = data_train[:, -3], data_train[:, -3]
x_test, y_test = data_test[:, -3], data_test[:, -3]

if x_train.ndim == 1:
    x_train = np.expand_dims(x_train,1) # shape matching for input dim is 1
if y_train.ndim == 1:
    y_train = np.expand_dims(y_train,1) # shape matching for output dim is 1
    
if x_test.ndim == 1:
    x_test = np.expand_dims(x_test,1) # shape matching for input dim is 1
if y_test.ndim == 1:
    y_test = np.expand_dims(y_test,1) # shape matching for output dim is 1

# build a dataset
trainX = []
trainY = []
testX = []
testY = []
validationX = []
validationY = []
for i in range(0, (len(y_train) - (seq_length+pred_seq_length)), seq_length):    
    	_x = x_train[i : i+seq_length]
    	_y = y_train[i+seq_length : i+(seq_length+pred_seq_length)] # for prediction
    	trainX.append(_x)
    	trainY.append(_y)

for i in range(0, (len(y_test) - (seq_length+pred_seq_length)), seq_length):
    	_x = x_test[i : i+seq_length]
    	_y = y_test[i+seq_length : i+(seq_length+pred_seq_length)] # for prediction
    	testX.append(_x)
    	testY.append(_y)

for i in range(len(y_test) - (seq_length+pred_seq_length)):
    	_x = x_test[i : i+seq_length]
    	_y = y_test[i+seq_length : i+(seq_length+pred_seq_length)] # for prediction
    	validationX.append(_x)
    	validationY.append(_y)

# train/test split
train_size, test_size = len(trainY), len(testY)

trainX, trainY = np.array(trainX, dtype=np.float32), np.array(trainY, dtype=np.float32)
testX, testY = np.array(testX, dtype=np.float32), np.array(testY, dtype=np.float32)
validationX, validationY = np.array(validationX, dtype=np.float32), np.array(validationY, dtype=np.float32)

mini_batch_size = 30 # full batch if (mini_batch_size == train_size)
batch_index_jump = int(train_size/mini_batch_size)

X = tf.placeholder(tf.float32, [None, seq_length, input_size]) # for time_major = False
X_predictor = tf.placeholder(tf.float32, [None, pred_seq_length, output_size])

def lstm_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        return cell

# Encoder layer configuration
with tf.variable_scope("encoder"):
    multi_encoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(depth)])
    encoder_zero_state = multi_encoder_cells.zero_state(tf.shape(X)[0], dtype=tf.float32)
    _encoder_output, _encoder_state = tf.nn.dynamic_rnn(multi_encoder_cells, X, initial_state=encoder_zero_state, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)
   
# Decoder layer configuration
with tf.variable_scope("decoder"):
    decoder_inputs = tf.zeros_like(X) # for decoder zero input    
    multi_decoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(depth)])
    _decoder_output, _decoder_state = tf.nn.dynamic_rnn(multi_decoder_cells, decoder_inputs, initial_state=_encoder_state, dtype=tf.float32) # time_major = False (batch,seq_length,input_size)
    decoder_outputs = tf.contrib.layers.fully_connected(_decoder_output, output_size, activation_fn=None)
    X_reverse = tf.reverse(X, axis=[1]) # reverse the time_step's order. (ex: x1,x2,x3 -> x3,x2,x1)

# Predictor layer configuration
with tf.variable_scope("predictor"):
    predictor_inputs = tf.zeros([tf.shape(X)[0], pred_seq_length, output_size]) # for predictor zero input
    multi_predictor_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(depth)])
    if zero_input is True:
        _predictor_output, _predictor_state = tf.nn.dynamic_rnn(multi_predictor_cells, predictor_inputs, initial_state=_encoder_state, dtype=tf.float32)
        predictor_outputs = tf.contrib.layers.fully_connected(_predictor_output, output_size, activation_fn=None)
    else:
        _predictor_state = _encoder_state
        predictor_output_list = []
        predictor_output = tf.zeros([tf.shape(X)[0], input_size]) # initial zero input
        for seq in range(pred_seq_length):
            _predictor_output, _predictor_state = multi_predictor_cells(predictor_output, _predictor_state)
            predictor_output = tf.contrib.layers.fully_connected(_predictor_output, output_size, activation_fn=None)
            predictor_output_list.append(predictor_output)
        predictor_outputs = tf.stack(predictor_output_list, axis=1)   
        
# loss calculation
if decoderReverse == True:
    decoder_loss = tf.reduce_mean(tf.square(decoder_outputs - X_reverse))
else:
    decoder_loss = tf.reduce_mean(tf.square(decoder_outputs - X))
predictor_loss = tf.reduce_mean(tf.square(predictor_outputs - X_predictor))
loss = decoder_loss + predictor_loss
train = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver() # for saving all variables

# session open
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    loss_plot = []
    if useSavedVariables == True:
        # Restore variables from disk.
        saver.restore(sess, "./variable_save/LSTMAutoencoder_vehicle_2.ckpt")
    else:
        # Training step
        for epoch in range(201):
            loss_sum = 0
            if batch_mix == False:
                batches = np.arange(0, train_size, mini_batch_size)
                np.random.shuffle(batches) # shuffle the mini batches
                for mini_batch in batches:
                    _, step_loss = sess.run([train, loss], 
                                            feed_dict={X: trainX[mini_batch:mini_batch+mini_batch_size], 
                                            X_predictor: trainY[mini_batch:mini_batch+mini_batch_size]})
                    loss_sum += step_loss
            else:
                for i in range(batch_index_jump):
                    _, step_loss = sess.run([train, loss], 
                                            feed_dict={X: trainX[i:train_size:batch_index_jump], 
                                            X_predictor: trainY[i:train_size:batch_index_jump]})
                    loss_sum += step_loss
            loss_plot.append(loss_sum / int(train_size/mini_batch_size))
            
            if step_loss < 1e-5:
                break
            
            if epoch%10 == 0:
                print("[Epoch: {}] loss: {}".format(epoch, step_loss))
        
        # Save the variables to disk.
        save_path = saver.save(sess, "./variable_save/LSTMAutoencoder_vehicle_2.ckpt")
    
    # Test step
    loss_train, dec_train, pred_train = sess.run([loss, decoder_outputs, predictor_outputs], feed_dict={X: trainX, X_predictor: trainY})
    print("Train loss: {}".format(loss_train))
    
    loss_test, dec_test, pred_test    = sess.run([loss, decoder_outputs, predictor_outputs], feed_dict={X: testX, X_predictor: testY})
    print("Test loss: {}".format(loss_test))
    
    loss_validation, dec_validation, pred_validation = sess.run([predictor_loss, decoder_outputs, predictor_outputs], feed_dict={X: validationX, X_predictor: validationY})
    print("Validation loss: {}".format(loss_validation))
    
    
check_index = 20 # check index among test data
test_time = np.linspace(0, 0.01*(seq_length+pred_seq_length-1), (seq_length+pred_seq_length))
test_time = np.reshape(test_time, [(seq_length+pred_seq_length), 1])
plt.figure(1)
plt.plot(test_time, np.concatenate([testX[check_index,:,:], testY[check_index,:,:]]), label='test data')
plt.plot(test_time[:seq_length], dec_test[check_index,:,:], label='decoded data')
plt.plot(test_time[seq_length:], pred_test[check_index,:,:], label='predicted data')
plt.grid()
plt.legend(loc='lower right')


trainX_vector = np.reshape(trainX, (np.size(trainX),1))
dec_train_vector = np.reshape(dec_train, (np.size(dec_train),1))
pred_train_vector = np.reshape(pred_train, (np.size(pred_train),1))

testX_vector = np.reshape(testX, (np.size(testX),1))
dec_test_vector = np.reshape(dec_test, (np.size(dec_test),1))
pred_test_vector = np.reshape(pred_test, (np.size(pred_test),1))

trainY_vector = np.reshape(trainY, (np.size(trainY),1))
testY_vector = np.reshape(testY, (np.size(testY),1))

train_time = np.linspace(0, (np.size(trainX_vector)-1)*0.01, np.size(trainX_vector))
#train_pred_time = np.linspace(seq_length*0.01, (np.size(trainX_vector)+seq_length-1)*0.01, np.size(trainX_vector))
train_pred_time = np.array([[i+j for i in np.arange(pred_seq_length)*0.01] for j in (np.arange(0, len(pred_train)*seq_length*0.01, seq_length*0.01)+seq_length*0.01)]).flatten()

test_time = np.linspace(0, (np.size(testX_vector)-1)*0.01, np.size(testX_vector))
#test_pred_time = np.linspace(seq_length*0.01, (np.size(testX_vector)+seq_length-1)*0.01, np.size(testX_vector))
test_pred_time = np.array([[i+j for i in np.arange(pred_seq_length)*0.01] for j in (np.arange(0, len(pred_test)*seq_length*0.01, seq_length*0.01)+seq_length*0.01)]).flatten()

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(train_time, trainX_vector, label='train data')
plt.plot(train_time, dec_train_vector, label='decoded', marker='.')
plt.plot(train_pred_time, pred_train_vector, label='predicted', marker='.')
plt.grid()
plt.legend(loc='lower right')
plt.title('Train data result')

plt.subplot(2,1,2)
plt.plot(test_time, testX_vector, label='test data')
plt.plot(test_time, dec_test_vector, label='decoded', marker='.')
plt.plot(test_pred_time, pred_test_vector, label='predicted', marker='.')
plt.grid()
plt.legend(loc='lower right')
plt.title('Test data result')

plt.figure(3)
plt.plot(loss_plot)
plt.title('Loss')

test_time2 = np.linspace(0, (np.size(validationY[:,-1,:])-1)*0.01, np.size(validationY[:,-1,:]))

plt.figure(4)
plt.plot(test_time2, validationY[:,-1,:], label='Test data')
plt.plot(test_time2, pred_validation[:,-1,:], label='predicted', marker='.')
plt.grid()
plt.legend(loc='lower right')
plt.title('Test data result at each time stamp (5-step prediction)')

# result data saving
result = np.stack([validationY[:,-1,:], pred_validation[:,-1,:]], axis=1)
result = np.squeeze(result, axis=2)
np.savetxt('result.txt', result, fmt='%.8e',delimiter=',')
