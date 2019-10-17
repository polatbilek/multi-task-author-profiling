import tensorflow as tf
from parameters import FLAGS
import numpy as np
import sys

class network(object):

	############################################################################################################################
	def __init__(self, embeddings):
		
		self.prediction = []
		self.cells = {}

		with tf.device('/cpu:0'):
			# create word embeddings
			self.tf_embeddings = tf.Variable(tf.constant(0.0, shape=[len(embeddings), len(embeddings[0])]), trainable=False, name="tf_embeddings")
			self.embedding_placeholder = tf.placeholder(tf.float32, [len(embeddings), len(embeddings[0])])
			# initialize this once  with sess.run when the session begins
			self.embedding_init = self.tf_embeddings.assign(self.embedding_placeholder)

		with tf.device('/gpu:0'):
			# RNN placeholders
			self.X = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

			self.Y_age = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.numof_age_classes])
			self.Y_gender = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.numof_gender_classes])
			self.Y_job = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.numof_job_classes])

			self.sequence_length = tf.placeholder(tf.int32, FLAGS.batch_size)
			self.reg_param = tf.placeholder(tf.float32, shape=[])


			# create GRU cells
			with tf.variable_scope("cells"):
				self.cells['global-fw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.global_rnn_cell_size, activation=tf.sigmoid,
													  name="forward-cells-global")
				self.cells['global-bw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.global_rnn_cell_size, activation=tf.sigmoid,
													  name="backword-cells-global")

				self.cells['age-fw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="forward-cells-age")
				self.cells['age-bw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="backword-cells-age")

				self.cells['job-fw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="forward-cells-job")
				self.cells['job-bw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="backword-cells-job")

				self.cells['gender-fw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="forward-cells-gender")
				self.cells['gender-bw'] = tf.nn.rnn_cell.GRUCell(num_units=FLAGS.semantic_rnn_cell_size, activation=tf.sigmoid,
													  name="backword-cells-gender")


			with tf.variable_scope("fully_connecteds"):
				# weigths
				self.weights = {
					'fc_age': tf.Variable(tf.random_normal([2 * FLAGS.semantic_rnn_cell_size, FLAGS.numof_age_classes]),
										  name="fc-age-weights"),
					'fc_job': tf.Variable(tf.random_normal([2 * FLAGS.semantic_rnn_cell_size, FLAGS.numof_job_classes]),
										  name="fc-job-weights"),
					'fc_gender': tf.Variable(tf.random_normal([2 * FLAGS.semantic_rnn_cell_size, FLAGS.numof_gender_classes]),
											name="fc-gender-weights")}
				# biases
				self.bias = {'fc_age': tf.Variable(tf.random_normal([FLAGS.numof_age_classes]), name="fc-age-bias-noreg"),
							 'fc_job': tf.Variable(tf.random_normal([FLAGS.numof_job_classes]), name="fc-job-bias-noreg"),
							 'fc_gender': tf.Variable(tf.random_normal([FLAGS.numof_gender_classes]), name="fc-gender-bias-noreg")}


			# initialize the computation graph for the neural network
			self.rnn()
			self.architecture()
			self.backward_pass()



	############################################################################################################################
	def architecture(self):
		with tf.device('/gpu:0'):
			# FC layer for reducing the dimension to 2(# of classes)
			self.logits_age    = tf.tensordot(self.rnn_output_age, self.weights["fc_age"], axes=1) + self.bias["fc_age"]
			self.logits_job    = tf.tensordot(self.rnn_output_job, self.weights["fc_job"], axes=1) + self.bias["fc_job"]
			self.logits_gender = tf.tensordot(self.rnn_output_gender, self.weights["fc_gender"], axes=1) + self.bias["fc_gender"]

			# predictions
			self.prediction_age = tf.nn.softmax(self.logits_age)
			self.prediction_job = tf.nn.softmax(self.logits_job)
			self.prediction_gender = tf.nn.softmax(self.logits_gender)

			# calculate accuracies
			self.correct_pred_age = tf.equal(tf.argmax(self.prediction_age, 1), tf.argmax(self.Y_age, 1))
			self.accuracy_age = tf.reduce_mean(tf.cast(self.correct_pred_age, tf.float32))

			self.correct_pred_job = tf.equal(tf.argmax(self.prediction_job, 1), tf.argmax(self.Y_job, 1))
			self.accuracy_job = tf.reduce_mean(tf.cast(self.correct_pred_job, tf.float32))

			self.correct_pred_gender = tf.equal(tf.argmax(self.prediction_gender, 1), tf.argmax(self.Y_gender, 1))
			self.accuracy_gender = tf.reduce_mean(tf.cast(self.correct_pred_gender, tf.float32))

			return self.prediction_age, self.prediction_job, self.prediction_gender, \
				   self.accuracy_age, self.accuracy_job, self.accuracy_gender



	############################################################################################################################
	def backward_pass(self):
		with tf.device('/gpu:0'):
			# calculate loss
			self.loss_age = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_age, labels=self.Y_age))
			self.loss_job = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_job, labels=self.Y_job))
			self.loss_gender = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_gender, labels=self.Y_gender))

			self.loss = self.loss_age + self.loss_job + self.loss_gender # total loss is sum of 3 tasks' loss

			# add L2 regularization
			self.l2 = self.reg_param * sum(
				tf.nn.l2_loss(tf_var)
				for tf_var in tf.trainable_variables()
				if not ("noreg" in tf_var.name or "bias" in tf_var.name)
			)
			self.loss += self.l2

			# optimizer
			self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
			self.train = self.optimizer.minimize(self.loss)

			return self.loss, self.train



	############################################################################################################################
	def rnn(self):
		with tf.device('/cpu:0'):
			# embedding layer
			self.rnn_input = tf.nn.embedding_lookup(self.tf_embeddings, self.X)

			# global rnn layer
			(self.outputs, self.output_states) = tf.nn.bidirectional_dynamic_rnn(self.cells["global-fw"],
												self.cells["global-bw"],
												self.rnn_input,
												self.sequence_length,
												dtype=tf.float32,
												scope="global-rnn")

			# concatenate the backward and forward cells of global to feed them into higher layer
			self.global_rnn_output = tf.concat([self.outputs[0], self.outputs[1]], -1)
			self.seqlen_semantic_layer = [2*FLAGS.global_rnn_cell_size for i in range(FLAGS.batch_size)]


			# age rnn layer
			(self.outputs_age, self.output_states_age) = tf.nn.bidirectional_dynamic_rnn(self.cells["age-fw"],
													self.cells["age-bw"],
													self.global_rnn_output,
													self.sequence_length, 
													dtype=tf.float32,
													scope="age-rnn")


			# job rnn layer
			(self.outputs_job, self.output_states_job) = tf.nn.bidirectional_dynamic_rnn(self.cells["job-fw"],
													self.cells["job-bw"],
													self.global_rnn_output,
													self.sequence_length, 
													dtype=tf.float32,
													scope="job-rnn")


			# gender rnn layer
			(self.outputs_gender, self.output_states_gender) = tf.nn.bidirectional_dynamic_rnn(self.cells["gender-fw"],
													self.cells["gender-bw"],
													self.global_rnn_output,
													self.sequence_length,
													dtype=tf.float32,
													scope="gender-rnn")


			# concatenate the backward and forward cells of semantic layers
			self.rnn_output_age = tf.concat([self.output_states_age[0], self.output_states_age[1]], 1)
			self.rnn_output_job = tf.concat([self.output_states_job[0], self.output_states_job[1]], 1)
			self.rnn_output_gender = tf.concat([self.output_states_gender[0], self.output_states_gender[1]], 1)


			# reshape the output for the next layers
			self.rnn_output_age = tf.reshape(self.rnn_output_age,
										 [FLAGS.batch_size, 2 * FLAGS.semantic_rnn_cell_size])

			self.rnn_output_job = tf.reshape(self.rnn_output_job,
										 [FLAGS.batch_size, 2 * FLAGS.semantic_rnn_cell_size])

			self.rnn_output_gender = tf.reshape(self.rnn_output_gender,
										 [FLAGS.batch_size, 2 * FLAGS.semantic_rnn_cell_size])


			return self.rnn_output_age, self.rnn_output_job, self.rnn_output_gender
