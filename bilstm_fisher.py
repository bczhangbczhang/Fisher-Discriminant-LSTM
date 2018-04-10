
import tensorflow as tf
import math
import numpy as np
import scipy.io as sio 
import os
import random
import csv

theta = 0.1
alpha = 0.5
delta = 0.01
classes=12

def fisher_criterion(features, labels,delta):
	with tf.variable_scope('means', reuse=True):
		means = tf.get_variable('means')

	inter = 0
	labels = tf.reshape(labels, [-1])
	len=tf.size(labels)
	len=tf.cast(len,tf.float32)
	means_batch1 = tf.gather(means, labels)
	for i in range(classes-1):
		a=tf.reshape(means[0, :],[1,-1])
		means_aga=tf.concat(0,[means[1:,:],a] )
		inter=inter+tf.nn.l2_loss(means-means_aga)
	inter=inter/(classes-1)/classes
	intra=tf.nn.l2_loss(features - means_batch1)/len
	loss = intra-delta*inter

	return loss

def update_means(features, labels, alpha):
	with tf.variable_scope('means', reuse=True):
		means = tf.get_variable('means')

	labels = tf.reshape(labels, [-1])
	means_batch = tf.gather(means, labels)
	diff = means_batch - features

	unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
	appear_times = tf.gather(unique_count, unique_idx)
	appear_times = tf.reshape(appear_times, [-1, 1])

	diff = diff / tf.cast((1 + appear_times), tf.float32)
	diff = alpha * diff
	means = tf.scatter_sub(means, labels, diff)

	return means

class Config(object):
	"""
	a class to store parameters,

	"""

	def __init__(self, X_train, X_test):
		# Input data
		self.train_count = len(X_train)  # 3500 training series
		self.test_data_count = len(X_test)  # 2047 testing series
		self.n_steps = len(X_train[0])  # 1000 time_steps per series

		# Trainging

		self.learning_rate = 0.002
		self.training_epochs = 100
		self.batch_size = 200
		# LSTM structure
		self.n_inputs = len(X_train[0][0])
		self.n_hidden = 128  # nb of neurons inside the neural network
		self.n_classes = classes  # Final output classes
		self.W = {
			'hidden': tf.Variable(tf.random_normal([self.n_inputs, 2*self.n_hidden])),
			'output': tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]))
		}
		self.biases = {
			'hidden': tf.Variable(tf.random_normal([2*self.n_hidden], mean=1.0)),
			'output': tf.Variable(tf.random_normal([self.n_classes]))
		}


def LSTM_Network(feature_mat, config):

	feature_mat = tf.transpose(feature_mat, [1, 0, 2])
	feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
	hidden = tf.nn.relu(tf.matmul(
		feature_mat, config.W['hidden']
	) + config.biases['hidden'])
	hidden = tf.split(0, config.n_steps, hidden)

	lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)

	outputs, _w, _e = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, hidden, dtype=tf.float32)

	lstm_mean_output = tf.reduce_mean(outputs,0)
	return lstm_mean_output

def one_hot(label):

	label_num = len(label)
	new_label = label.reshape(label_num)
	n_values = np.max(new_label) + 1
	return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


if __name__ == "__main__":
	mat_path=u'data/database.mat'
	data_mat=sio.loadmat(mat_path)
	data_train=data_mat['data_train'][0]
	data_test= data_mat['data_test'][0]

	num_train = len(data_train)
	num_test = len(data_test)
	num_time = len(data_train[0][0])

	X_train = np.zeros([num_train, num_time, 6], dtype='float32')
	X_test = np.zeros([num_test, num_time, 6], dtype='float32')
	y_train = np.zeros([num_train, 1], dtype='int32')
	y_test = np.zeros([num_test, 1], dtype='int32')
	for i in range(num_train):
		for j in range(num_time):
			X_train[i][j][0] = data_train[i][0][j][0]
			X_train[i][j][1] = data_train[i][0][j][1]
			X_train[i][j][2] = data_train[i][0][j][2]
			X_train[i][j][3] = data_train[i][0][j][3]
			X_train[i][j][4] = data_train[i][0][j][4]
			X_train[i][j][5] = data_train[i][0][j][5]
		y_train[i][0] = data_train[i][0][0][6]

	for i in range(num_test):
		for j in range(num_time):
			X_test[i][j][0] = data_test[i][0][j][0]
			X_test[i][j][1] = data_test[i][0][j][1]
			X_test[i][j][2] = data_test[i][0][j][2]
			X_test[i][j][3] = data_test[i][0][j][3]
			X_test[i][j][4] = data_test[i][0][j][4]
			X_test[i][j][5] = data_test[i][0][j][5]
		y_test[i][0] = data_test[i][0][0][6]

	y_train_hot = one_hot(y_train)
	y_test_hot = one_hot(y_test)

	config = Config(X_train, X_test)

	X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
	Y_one_hot = tf.placeholder(tf.int32, [None, config.n_classes])
	Y = tf.placeholder(tf.int32, [None, 1])


	with tf.variable_scope('means'):
		means = tf.get_variable('means', [config.n_classes, 2*config.n_hidden], dtype=tf.float32, \
								  initializer=tf.constant_initializer(0), trainable=False)
	LSTM_feature = LSTM_Network(X, config)
	fisher_cri = fisher_criterion(LSTM_feature, Y,delta)
	update_means = update_means(LSTM_feature, Y, alpha)

	Y_predict=tf.matmul(LSTM_feature, config.W['output']) + config.biases['output']
	softmax_loss=tf.nn.softmax_cross_entropy_with_logits(logits=Y_predict, labels=Y_one_hot)
	cost = tf.reduce_mean(softmax_loss)+ theta*fisher_cri

	optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(cost)
	correct_pred = tf.equal(tf.argmax(Y_predict, 1), tf.argmax(Y_one_hot, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

	Y_right=tf.argmax(Y_one_hot,1)
	Y_pre=tf.argmax(Y_predict,1)

	saver=tf.train.Saver()
	sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
	tf.global_variables_initializer().run()

	iter=0

	for i in range(config.training_epochs):
		for start, end in zip(range(0, config.train_count, config.batch_size),
							  range(config.batch_size, config.train_count + 1, config.batch_size)):

			cen,_,accuracy_out_train,loss_train=sess.run([update_means,optimizer,accuracy,cost], feed_dict={X: X_train[start:end],
										   Y: y_train[start:end],Y_one_hot:y_train_hot[start:end]})
			iter += 1
			print("traing iter: {},".format(iter) + \
				  " train accuracy : {},".format(accuracy_out_train) + \
				  " train loss : {}".format(loss_train))
			if iter>1000:

				if iter % 200 == 0:
					saver.save(sess, "model_fisher/model.ckpt" + str(iter))

		pred_out, accuracy_out_test, loss_test = sess.run([Y_predict, accuracy, cost], feed_dict={
												X: X_test, Y: y_test,Y_one_hot:y_test_hot})
		print("epoch: {},".format(i)+\
			  " test accuracy : {},".format(accuracy_out_test)+\
			  " loss : {}".format(loss_test))
	saver.restore(sess, "model_fisher/model.ckpt1600")
	accuracy_out_test, loss_out = sess.run([accuracy, cost], feed_dict={
											X: X_test, Y: y_test,Y_one_hot:y_test_hot})
	print(" test accuracy : {},".format(accuracy_out_test)+\
		  " loss : {}".format(loss_out))

	sess.close()





