import tensorflow as tf
from preprocess import *
import sys

#####################################################################################################################
##loads a model and tests it
#####################################################################################################################
def test(network, test_data, test_users, vocabulary, embeddings, ground_truth):
	saver = tf.train.Saver(max_to_keep=None)


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.device('/gpu:0'):
		with tf.Session(config=config) as sess:

			# init variables
			#init = tf.global_variables_initializer()
			#sess.run(init)
			#sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings})

			# load the model from checkpoint file if it is required
			load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
			saver.restore(sess, load_as)
			print("Loading the pretrained model from: " + str(load_as))


			test_batch_count = int(len(test_data) / FLAGS.batch_size)

			batch_accuracy_age = 0.0
			batch_accuracy_job = 0.0
			batch_accuracy_gender = 0.0
			batch_loss = 0.0
			batch_loss_age = 0.0
			batch_loss_job = 0.0
			batch_loss_gender = 0.0

			# TESTING
			for batch in range(test_batch_count):
				# Prepare the batch
				test_batch_x, \
				test_batch_y_age, \
				test_batch_y_job, \
				test_batch_y_gender, \
				test_batch_seqlen = prepWordBatchData(test_data,
											test_users,
											ground_truth,
											vocabulary,
											batch)

				# Prepare feed values
				feed_dict = {network.X: test_batch_x,
							 network.Y_age: test_batch_y_age,
							 network.Y_job: test_batch_y_job,
							 network.Y_gender: test_batch_y_gender,
							 network.sequence_length: test_batch_seqlen,
							 network.reg_param: FLAGS.l2_reg_lambda}

				# Run the computational graph
				loss_age, loss_job, loss_gender, loss, \
				prediction_age, prediction_job, prediction_gender, \
				accuracy_age, accuracy_job, accuracy_gender = sess.run(
					[network.loss_age, network.loss_job, network.loss_gender, network.loss,
					 network.prediction_age, network.prediction_job, network.prediction_gender,
					 network.accuracy_age, network.accuracy_job, network.accuracy_gender], feed_dict=feed_dict)

				# Calculate the metrics
				batch_loss += loss

				batch_loss_age += loss_age
				batch_loss_job += loss_job
				batch_loss_gender += loss_gender

				batch_accuracy_age += accuracy_age
				batch_accuracy_job += accuracy_job
				batch_accuracy_gender += accuracy_gender


			batch_accuracy_age /= test_batch_count
			batch_accuracy_job /= test_batch_count
			batch_accuracy_gender /= test_batch_count

			print(" , Loss= " + "{0:5.4f}".format(batch_loss) +
				  " , Age accuracy= " + "{0:0.5f}".format(batch_accuracy_age) +
				  " , Job accuracy= " + "{0:0.5f}".format(batch_accuracy_job) +
				  " , Gender accuracy= " + "{0:0.5f}".format(batch_accuracy_gender))

