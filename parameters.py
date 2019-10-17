class flags(object):

	def __init__(self):

		#set sizes
		self.test_set_size = 0.15
		self.validation_set_size = 0.1
		self.training_set_size = 0.75


		#input file paths
		self.word_embed_path = "/home/darg1/Desktop/ozan/crawl-300d-2M.vec"#"C:\\Users\\polat\\Desktop\\fasttext\\crawl-300d-2M.vec"
		self.data_path = "/home/darg1/Desktop/ozan/set"#"C:\\Users\\polat\\Desktop\\new_blogs"

		#output file paths
		self.model_path = "/home/darg1/Desktop/ozan/models"
		self.model_name = "en-model-0.001-0.0001-0.ckpt"
		self.log_path = "/home/darg1/Desktop/ozan/logs.txt"


		#optimization parameters
		self.model_save_threshold_age = 0.75
		self.model_save_threshold_job = 0.75
		self.model_save_threshold_gender = 0.75
		self.use_pretrained_model = False
		self.optimize = False #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.01]
		self.reg_param = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
		self.rnn_cell_sizes = [90, 120, 150]


		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.0005
		self.learning_rate = 0.001

		# Number of classes
		self.numof_age_classes = 8
		self.numof_job_classes = 40
		self.numof_gender_classes = 2

			#RNN
		self.embedding_size = 300
		self.global_rnn_cell_size = 150
		self.semantic_rnn_cell_size = 100
		self.fc_size = 2*self.semantic_rnn_cell_size


		##########################################################################################################################
		# Training parameters
		self.batch_size = 16
		self.num_epochs = 2
		self.evaluate_every = 15

FLAGS = flags()
