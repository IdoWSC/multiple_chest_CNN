import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os
from imageio import imread
from progress.bar import Bar
from random import shuffle
import imgaug as ia
import imgaug.augmenters as iaa


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NUM_OF_GPUS = len(get_available_gpus())


class MultiChestVGG:

    place_holder_shape = [None, 224, 224, 3]
    conv_layers = 5
    conv_units = (2, 2, 3, 3, 3)
    conv_filter_sizes = (64, 128, 256, 512, 512)
    conv_batch_norm = (True, True, False, False, False)
    fc_layers = 3
    fc_layer_sizes = (4096, 4096, 1000)
    fc_dropout_layers = (True, True, False)
    fc1_layer_in_size = 25088
    net_name = 'MultiChestVGG'


    def __init__(self, im_shape=None, num_of_classes=1, use_softmax=False, is_training=True, sess=None,
                 use_batch_norm=False, use_common_conv_weights=True):

        if im_shape is not None and im_shape[0] is not None:
            self.place_holder_shape = [None]
            self.place_holder_shape.extend(im_shape)
        elif im_shape is not None:
            self.place_holder_shape = im_shape

        self.is_training = is_training
        self.num_of_classes = num_of_classes
        self.use_softmax = use_softmax

        self.conv_batch_norm = tuple([bool(use_batch_norm * layer) for layer in self.conv_batch_norm])

        with tf.variable_scope('input'):
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
            self.PA_images  = tf.placeholder(tf.float32, self.place_holder_shape, name='PA_images')
            self.LAT_images = tf.placeholder(tf.float32, self.place_holder_shape, name='LAT_images')

        self.parameters = dict()
        self.layers = dict()

        with tf.variable_scope(self.net_name):
            with tf.variable_scope('conv_variables'):
                self._build_conv_parameters(use_common_weights=use_common_conv_weights)
            with tf.variable_scope('fc_variables'):
                self._build_fc_parameters()
            self._build_net()
            self._build_classifier_layer()

        self.add_summeries()

        self.sess = None
        self.assign_session(sess)

    def _build_conv_parameters(self, trainable=False, use_common_weights=True):
        """
        :param layers: int - number of layers
        :param units: iterable of ints, amount of units per layer, len(units) = layers
        :param out_filtsers_size: iterable of ints, amount of filters in each unit of the layer,
        len(out_filters_size) = layers
        :return:
        """
        out_filters = 0  # init

        with tf.device('/cpu:0'):
            for angle in ['PA', 'LAT']:
                shape_calculator = np.array(self.place_holder_shape[1:3])
                for layer in range(self.conv_layers):
                    for unit in range(self.conv_units[layer]):

                        if not layer and not unit:
                            in_filters = self.PA_images.get_shape()[-1]
                        else:
                            in_filters = out_filters

                        out_filters = self.conv_filter_sizes[layer]

                        name_extension = '{}_{}_{}'.format(angle, layer + 1, unit + 1)

                        if not use_common_weights:
                            var_extension = name_extension
                        else:
                            var_extension = '{}_{}'.format(layer + 1, unit + 1)

                        with tf.variable_scope('conv_{}'.format(var_extension), reuse=tf.AUTO_REUSE):
                            self.parameters['conv_{}_W'.format(name_extension)] = \
                                tf.get_variable('weights',
                                                shape=[3, 3, in_filters, out_filters],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                trainable=trainable)

                            self.parameters['conv_{}_b'.format(name_extension)] = \
                                tf.get_variable('biases',
                                                shape=out_filters,
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer(),
                                                trainable=trainable)

                    shape_calculator = shape_calculator // 2

                    if self.conv_batch_norm[layer]:
                        bn_param_shape = np.append(shape_calculator.copy(), out_filters)

                        if use_common_weights:
                            name_extension = '{}'.format(layer + 1)
                        else:
                            name_extension = '{}_{}'.format(angle, layer + 1)

                        with tf.variable_scope('bn_{}'.format(name_extension), reuse=tf.AUTO_REUSE):
                            self.parameters['conv_{}_{}_bn_scale'.format(angle, layer + 1)] = \
                                tf.get_variable('conv{}_bn_scale'.format(layer + 1),
                                                shape=bn_param_shape,
                                                dtype=np.float32,
                                                initializer=tf.ones_initializer())

                            self.parameters['conv_{}_{}_bn_beta'.format(angle, layer + 1)] = \
                                tf.get_variable('conv{}_bn_beta'.format(layer + 1),
                                                shape=bn_param_shape,
                                                dtype=np.float32,
                                                initializer=tf.zeros_initializer())

                            self.parameters['conv_{}_{}_bn_pop_mean'.format(angle, layer + 1)] = \
                                tf.get_variable('conv{}_bn_pop_mean'.format(layer + 1),
                                                shape=bn_param_shape,
                                                dtype=np.float32,
                                                initializer=tf.zeros_initializer(),
                                                trainable=False)

                            self.parameters['conv_{}_{}_bn_pop_var'.format(angle, layer + 1)] = \
                                tf.get_variable('conv{}_bn_pop_var'.format(layer + 1),
                                                shape=bn_param_shape,
                                                dtype=np.float32,
                                                initializer=tf.ones_initializer(),
                                                trainable=False)

    def _build_fc_parameters(self, trainable=True):

        layer_size = 0  # init
        with tf.device('/cpu:0'):
            for layer in range(self.fc_layers):
                if not layer:
                    in_size = 2 * self.fc1_layer_in_size
                else:
                    in_size = layer_size

                layer_size = self.fc_layer_sizes[layer]
                name_extension = '{}'.format(self.conv_layers + layer + 1)
                with tf.variable_scope('fc{}'.format(name_extension)):
                    self.parameters['fc{}_W'.format(name_extension)] = \
                        tf.get_variable('weights',
                                        shape=[in_size, layer_size],
                                        dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        trainable=trainable)

                    self.parameters['fc{}_b'.format(name_extension)] = \
                        tf.get_variable('biases',
                                        shape=[layer_size],
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer(),
                                        trainable=trainable)

            with tf.variable_scope('final'):
                if self.use_softmax or self.num_of_classes > 2:
                    self.parameters['final_W'] = \
                    tf.get_variable('weights',
                                    shape=[layer_size, self.num_of_classes],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=trainable)

                    self.parameters['final_b'] = \
                    tf.get_variable('biases',
                                    shape=[self.num_of_classes],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=trainable)
                else:
                    self.parameters['final_W'] = \
                    tf.get_variable('weights',
                                    shape=[layer_size, 1],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=trainable)

                    self.parameters['final_b'] = \
                    tf.get_variable('biases',
                                    shape=[1],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=trainable)

    def _build_net(self):
        def build_structure_per_gpu(num_gpus, **kwargs):

            in_splits = {}
            for k, v in kwargs.items():
                in_splits[k] = tf.split(v, num_gpus)
            out_split = []
            for i in range(num_gpus):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                    with tf.variable_scope('GPU_{}'.format(i)):
                        out_split.append(model_net('gpu', i, x={'PA' : in_splits['x_PA'][i],
                                                                'LAT': in_splits['x_LAT'][i]}))

            return tf.concat(out_split, axis=0)

        def build_structure_for_cpu(**kwargs):
            in_splits = {}
            for k, v in kwargs.items():
                in_splits[k] = v
            with tf.device('/cpu:0'):
                out_split = model_net('cpu', 0, x={'PA' : in_splits['x_PA'],
                                                       'LAT': in_splits['x_LAT']})
            return out_split

        def model_net(device_type, device, x):

            fc_in = []
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            for angle, scan in x.items():
                with tf.variable_scope(angle):
                    # zero-mean input
                    with tf.variable_scope('preprocess_{}_{}'.format(angle, device)) as scope:
                        layer_input = scan - mean

                    for layer in range(self.conv_layers):
                        for unit in range(self.conv_units[layer]):
                            name_extension = '{}_{}_{}'.format(angle, layer + 1, unit + 1)
                            with tf.variable_scope('conv_{}_{}'.format(name_extension, device)):
                                conv = tf.nn.conv2d(layer_input,
                                                    self.parameters['conv_{}_W'.format(name_extension)],
                                                    [1, 1, 1, 1],
                                                    padding='SAME')

                                out = tf.nn.bias_add(conv, self.parameters['conv_{}_b'.format(name_extension)])

                                self.layers['conv_{}_{}_{}'.format(name_extension, device_type, str(device))] = \
                                    tf.nn.relu(out)

                            layer_input = self.layers['conv_{}_{}_{}'.format(name_extension, device_type, str(device))]

                        self.layers['pool_{}_{}_{}_{}'.format(angle, layer + 1, device_type, str(device))] = \
                            tf.nn.max_pool(layer_input,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='SAME',
                                           name='pool1')

                        if self.conv_batch_norm[layer]:
                            with tf.variable_scope('bn', reuse=tf.AUTO_REUSE):
                                self.layers['bn_{}_{}_gpu{}'.format(angle, layer + 1, device)] = \
                                    batch_norm(self.layers['pool_{}_{}_{}_{}'.format(angle,
                                                                                     layer + 1,
                                                                                     device_type,
                                                                                     str(device))],
                                               self.is_training,
                                               angle,
                                               layer,
                                               not bool(device))
                            layer_input = self.layers['bn_{}_{}_gpu{}'.format(angle, layer + 1, device)]
                        else:
                            layer_input = self.layers['pool_{}_{}_{}_{}'.format(angle, layer + 1, device_type, str(device))]
                    fc_in.append(tf.reshape(layer_input,
                                            [-1, self.fc1_layer_in_size]))  # squash tensor coming from conv to fc

            layer_input = tf.concat(fc_in, axis=1)

            with tf.variable_scope('fully_connected'):
                for layer in range(self.fc_layers):
                    name_extension = '{}'.format(self.conv_layers + layer + 1)
                    with tf.variable_scope('fc{}_{}'.format(name_extension, device)):
                        fc_mul = tf.matmul(layer_input, self.parameters['fc{}_W'.format(name_extension)])
                        fc_lin = tf.nn.bias_add(fc_mul, self.parameters['fc{}_b'.format(name_extension)])

                        self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))] = tf.nn.relu(fc_lin)
                        if self.fc_dropout_layers[layer]:
                            self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))] = \
                                tf.nn.dropout(self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))],
                                              keep_prob=self.dropout_keep_prob)

                    layer_input = self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))]

                with tf.variable_scope('final_{}'.format(device)):
                    fc_mul = tf.matmul(layer_input, self.parameters['final_W'])
                    fc_lin = tf.reshape(tf.nn.bias_add(fc_mul, self.parameters['final_b']), (-1, ))

                layer_input = fc_lin
                net_output = layer_input
            return net_output

        def batch_norm(batch, is_training, angle, layer, use_pre_weights, decay=0.99, epsilon=1e-5):
            scale = self.parameters['conv_{}_{}_bn_scale'.format(angle, layer + 1)]
            beta  = self.parameters['conv_{}_{}_bn_beta'.format(angle, layer + 1)]

            if use_pre_weights or not is_training:
                pop_mean = self.parameters['conv_{}_{}_bn_pop_mean'.format(angle, layer + 1)]
                pop_var  = self.parameters['conv_{}_{}_bn_pop_var'.format(angle, layer + 1)]
            else:
                pop_mean = tf.get_variable('pop_mean',
                                           shape=batch.get_shape()[1:],
                                           dtype=np.float32,
                                           initializer=tf.zeros_initializer(),
                                           trainable=False)

                pop_var = tf.get_variable('pop_var',
                                          shape=batch.get_shape()[1:],
                                          dtype=np.float32,
                                          initializer=tf.ones_initializer(),
                                          trainable=False)

            if is_training:
                batch_mean, batch_var = tf.nn.moments(batch, [0])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(batch,
                                                     batch_mean, batch_var, beta, scale, epsilon)
            else:
                return tf.nn.batch_normalization(batch,
                                                 pop_mean, pop_var, beta, scale, epsilon)
        if len(get_available_gpus()):
            self.final_layer = build_structure_per_gpu(len(get_available_gpus()),
                                                       x_PA=self.PA_images,
                                                       x_LAT=self.LAT_images)
        else:
            self.final_layer = build_structure_for_cpu(x_PA=self.PA_images,
                                                       x_LAT=self.LAT_images)

        # with tf.device('/cpu:0'):
        #     if self._softmax or self.num_of_classes > 2:
        #         self.output = tf.nn.softmax(self.fc_out)
        #     else:
        #         self.output = tf.nn.sigmoid(self.fc_out)

    def _build_classifier_layer(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope('classification'):
                if self.use_softmax or self.num_of_classes > 2:
                    self.probs = tf.nn.softmax(self.final_layer, name='probability')
                    self.predictions = tf.argmax(self.probs, axis=1, name='prediction')
                else:
                    self.probs = tf.nn.sigmoid(self.final_layer, name='probability')
                    self.predictions = tf.cast(tf.round(self.probs, name='prediction'), tf.int64)

    def load_weights_npz(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[k].assign(weights[k]))

    def load_weights_from_saver(self, weight_file, sess, saver=None):

        if saver is None:
            saver = tf.train.Saver()
        print('\nweights file is: ' + weight_file + '\n')
        saver.restore(sess, weight_file)
        return saver

    def load_conv_weights_npz(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if not k.startswith('conv'):
                continue
            print(i, k, np.shape(weights[k]))
            for angle in ['PA', 'LAT']:
                layers_and_params = k.split('_')
                layers_and_params[0] = layers_and_params[0].replace('conv', '')
                parameters_name_list = ['conv', angle]
                parameters_name_list.extend(layers_and_params)
                parameter_name = '_'.join(parameters_name_list)
                sess.run(self.parameters[parameter_name].assign(weights[k]))

    def add_summeries(self):
        for parameter, value in self.parameters.items():
            tf.summary.histogram(parameter, value)
        for activation, value in self.layers.items():
            tf.summary.histogram(activation, value)

    def predict(self, images_dict):

        angle_keys = ('PA', 'LAT')
        if not all([key in images_dict.keys() for key in angle_keys]):
            raise ValueError('No PA or LAT keys in images')
        x = {}
        for key in angle_keys:
            images = images_dict[key]
            if isinstance(images, list):
                x[key] = np.array(images)
            elif len(images.shape) == 3:
                x[key] = np.expand_dims(images, axis=0)
            else:
                x[key] = images
        if x[angle_keys[0]].shape != x[angle_keys[1]].shape:
            raise ValueError('PA and LAT are not in the same shape')

        return self.sess.run(self.predictions, feed_dict={self.PA_images: x['PA'], self.LAT_images: x['LAT'],
                                                          self.dropout_keep_prob: 1.0})

    def start_and_assign_session(self):
        self.assign_session(tf.Session())

    def assign_session(self, sess):
        self.sess = sess


class TrainNet:

    @classmethod
    def with_construct_BatchOrganiser(cls, net, shuffled_loc, batch_size, learning_rate=0.001, test_split=0.2, test_file=None):

        data_set = BatchOrganiser(shuffled_loc, batch_size, test_split, test_file)
        trainer = cls(net, data_set, learning_rate)
        return trainer

    def __init__(self, net, data_set, learning_rate=0.001):

        self.net = net
        self.learning_rate = learning_rate


        self.data_set = data_set
        self.train_size = self.data_set.train_size
        self.test_size = self.data_set.test_size
        self.end_of_train = self.data_set.end_of_train

        self.train_batch_size = self.data_set.train_batch_size
        self.test_batch_size = self.data_set.test_batch_size

        self._build_trainer()

    def _build_trainer(self):

        with tf.name_scope('train'):
            with tf.device('/cpu:0'):
                self.ground_truth = tf.placeholder(tf.int64, (None,), name='y')
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                print('\nBuilding xentropy Loss\n')
                with tf.variable_scope('cost_function'):
                    if self.net.use_softmax or self.net.num_of_classes > 1:
                        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                           logits=self.net.final_layer))
                    else:

                        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.ground_truth, tf.float32),
                                                                                           logits=tf.reshape(self.net.final_layer, (-1, ))))
                    tf.summary.scalar('train_cost', self.cost)

            with tf.variable_scope('optimizer'):

                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9,
                                                           epsilon=1e-10).minimize(self.cost,
                                                                                   global_step=self.global_step)

            self._merged_train_summary = tf.summary.merge_all()
            with tf.variable_scope('validation_set'):
                self.predicted = tf.placeholder(tf.int64, (None,), name='y_pred')
                self.validation_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.ground_truth, self.predicted), tf.float32))
                self.validation_accuracy_summary = tf.summary.scalar('validation_accuracy', self.validation_accuracy)


        # saver
        self.saver = tf.train.Saver()

    def assign_session(self, sess=None):
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.net.assign_session(self.sess)

    def init_variables(self):
        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def load_weights(self, weight_file, sess):
        return self.net.load_weights_from_saver(weight_file, sess, self.saver)

    def load_weights_npz(self, weight_file, sess):
        return self.net.load_weights_npz(weight_file, sess)

    def load_conv_weights_npz(self, weight_file, sess):
        return self.net.load_conv_weights_npz(weight_file, sess)

    def batch_advance(self):

        self.train_batch, self.train_labels, self.batch_paths = self.data_set.get_train_batch()

    def batch_pass(self, sess, dropout_keep_prob, train_writer=None, add_summary=False, add_metadata=False):
        def run_summary():
            feed_dict = {self.net.PA_images:  self.train_batch['PA'],
                         self.net.LAT_images: self.train_batch['LAT'],
                         self.ground_truth: self.train_labels,
                         self.net.dropout_keep_prob: 1.0}
            summary, batch_number = sess.run([self._merged_train_summary, self.global_step], feed_dict=feed_dict)
            train_writer.add_summary(summary, batch_number)

        def run_batch():
            feed_dict = {self.net.PA_images:  self.train_batch['PA'],
                         self.net.LAT_images: self.train_batch['LAT'],
                         self.ground_truth: self.train_labels,
                         self.net.dropout_keep_prob: dropout_keep_prob}
            if add_metadata:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, c, batch_number = sess.run([self.optimizer, self.cost, self.global_step],
                                              feed_dict=feed_dict,
                                              options=run_options,
                                              run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step{}'.format(batch_number))
            else:
                _, c, batch_number = sess.run([self.optimizer, self.cost, self.global_step], feed_dict=feed_dict)
            return c, batch_number

        # BATCH LOADING
        self.batch_advance()

        if add_summary:
            run_summary()

        # Run optimization op (backprop)
        cost, batch_number = run_batch()

        return cost, self.batch_paths, batch_number

    def train_accuracy(self, sess):

        return sess.run(self.cost, {self.net.PA_images:  self.train_batch['PA'],
                                    self.net.LAT_images: self.train_batch['LAT'],
                                    self.net.dropout_keep_prob: 1.0,
                                    self.ground_truth: self.train_labels})

    def get_validation_accuracy(self, sess):

        def argsort(seq):
            # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
            # by unutbu
            return sorted(range(len(seq)), key=seq.__getitem__, reverse=False)

        def generate_log_str_list():
            """
            generate a list with a starting header if desired
            :return: list with strings of headers
            """
            return []

        def add_documenation_to_log_str():
            # log_str_list.append('\n')
            dist = np.absolute(test_labels - net_output)
            for i, path in enumerate(batch_paths):
                log_str_list.append(path + ' ; ground_truth: {}, predicted: {}, dist: {}'.format(test_labels[i],
                                                                                                    net_output[i],
                                                                                                    dist[i]))

        def sort_documentation_list():
            """
            sort test documentation list so that the failures are on top
            :return:
            """
            dist = np.absolute(test_labels - net_output)
            dist_arg_sorted = argsort(dist.tolist())

            sorted_log_str_list = [log_str_list[i] for i in dist_arg_sorted]
            add_line_feed_buffer_to_log(sorted_log_str_list, self.data_set.test_batch_size)
            return sorted_log_str_list

        def add_line_feed_buffer_to_log(log_list, n=10):
            """
            add a line feed char every n rows in log list
            :param n: int
            :return:
            """
            for i in range(int(len(log_list)/n)):
                log_list.insert(n*i, '\n')
            log_list.append('\n')

        ground_truth_arr = np.zeros((self.data_set.test_size,), dtype=np.uint64)
        output_arr = np.zeros((self.data_set.test_size,), dtype=np.uint64)

        log_str_list = generate_log_str_list()

        #### Progress Bar Handling ####
        num_of_test_batches = self.data_set.test_size / self.data_set.test_batch_size
        if self.data_set.test_size % self.data_set.test_batch_size:
            num_of_test_batches += 1
        percantage_jump = num_of_test_batches / 100
        bar = Bar('Testing Validation Set', max=100)
        last_percentage_completed = -1
        ####################################
        j = 0
        while True:

            test_batch, test_labels, batch_paths = self.data_set.get_test_batch()

            batch_size_gap = len(output_arr[j * self.test_batch_size: (j + 1) * self.test_batch_size]) - \
                             len(test_batch['PA'])

            # if in the last batch, batch size is shortened for multiple gpu usage
            if self.data_set.end_of_test and batch_size_gap:
                ground_truth_arr = ground_truth_arr[:-batch_size_gap]
                output_arr = output_arr[:-batch_size_gap]

            if not test_batch['PA'].size:
                break
            feed_dict = {self.net.PA_images:  test_batch['PA'],
                         self.net.LAT_images: test_batch['LAT'],
                         self.net.dropout_keep_prob: 1.0,
                         self.ground_truth: test_labels}

            net_output = sess.run(self.net.predictions, feed_dict=feed_dict)
            try:
                ground_truth_arr[j * self.test_batch_size: (j + 1) * self.test_batch_size] = test_labels
                output_arr[j * self.test_batch_size: (j + 1) * self.test_batch_size] = net_output
            except IndexError:
                ground_truth_arr[j * self.test_batch_size:] = test_labels
                output_arr[j * self.test_batch_size:] = net_output
            add_documenation_to_log_str()
            percentage_completed = int(j / percantage_jump)
            if percentage_completed != last_percentage_completed:
                last_percentage_completed = percentage_completed
                bar.next()
            if self.data_set.end_of_test:
                break
            j += 1

        feed_dict = {self.ground_truth: ground_truth_arr,
                     self.predicted: output_arr,
                     self.net.dropout_keep_prob: 1.0}
        acc_validation, summary = sess.run([self.validation_accuracy, self.validation_accuracy_summary],
                                           feed_dict=feed_dict)
        bar.finish()

        return acc_validation, summary, sort_documentation_list()

    def save_model(self, sess, save_path):
        self.saver.save(sess=sess, save_path=save_path)

    def check_end_of_train_set(self):
        return self.data_set.end_of_train

    def set_learning_rate(self, new_learning_rate=None):
        if new_learning_rate is None or new_learning_rate == 0.0:
            self.learning_rate = self.learning_rate
        else:
            self.learning_rate = new_learning_rate
        return

    def jump_forward_to_batch(self, dst, current=0):

        if dst - current < 0:
            raise ValueError('dst batch num {} is lower than current batch number {}, '
                             'dst - current must be > 0'.format(dst, current))
        elif dst == current:
            pass
        else:
            for i in range(int(dst - current)):
                self.data_set.progress_one_batch()

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def calc_number_of_train_batches(self):
        return self.calc_bacth_size(self.train_size, self.train_batch_size)

    def calc_number_of_test_batches(self):
        return self.calc_bacth_size(self.test_size, self.test_batch_size)

    @staticmethod
    def calc_bacth_size(set_size, batch_size):
        return int(set_size / batch_size) + bool(set_size % batch_size)

    @staticmethod
    def determine_batch_num(epoch, num_of_batches, overall_batch_num, start_epoch):
        """
        in case epoch = 0 than batch_num is equal to start_from_batch
        """
        if not epoch:
            batch_num = overall_batch_num
        elif epoch == start_epoch:
            batch_num = overall_batch_num % (start_epoch * num_of_batches)
        else:
            batch_num = 0
        return batch_num



class BatchOrganiser:
    """class for arranging batches for training.
    if test split == 0 then a path of test folder given at test_graph_loc will be used to load test_graph_loc.
    else test set will be empty"""
    def __init__(self, train_file_graph_path, train_batch_size, test_batch_size=None, test_split=0.2, test_file_graph_path=None,
                 num_of_frames=16, require_resize=False, resize_height=112, resize_width=112,
                 save_frame_location='', shuffle_each_epoch=False, image_augmentations=False,
                 augmentation_probability=0.25):

        self.train_batch_size = train_batch_size

        if test_batch_size is None:
            self.test_batch_size = self.train_batch_size
        else:
            self.test_batch_size = test_batch_size

        self.test_split = test_split
        self.test_file_graph_path = test_file_graph_path
        self.shuffle_each_epoch = shuffle_each_epoch

        self.num_of_frames = num_of_frames
        self.resize = require_resize
        self.height = resize_height
        self.width = resize_width

        self.save_location = save_frame_location

        train_file_graph = open(train_file_graph_path, 'r')
        self.clips = train_file_graph.read().split('\n')

        # correction for txt file ending with \n
        if self.clips[-1] == '':
            del self.clips[-1]

        if len(self.clips[0].split(',')) != len(self.clips[1].split(',')):  # first row intro check
            del self.clips[0]

        train_file_graph.close()
        if not self.test_split:
            self.train_size = len(self.clips)
            self.train = (clip for clip in self.clips)

            if self.test_file_graph_path is not None:
                self._load_test_set_from_file()
            else:
                self.test = []
        else:
            self.train_size = int(len(self.clips) * (1 - test_split))
            self.test_size = len(self.clips) - self.train_size

            self.train = (clip for clip in self.clips[:self.train_size])
            self.test = (clip for clip in self.clips[self.train_size:])

        self.end_of_train = False
        self.end_of_test = False

        self.image_augmentations = image_augmentations
        if self.image_augmentations:
            self.init_augmentations(augmentation_probability)
        else:
            self.aug = None

    def _load_test_set_from_file(self):
        test_file_graph = open(self.test_file_graph_path, 'r')
        test_images = test_file_graph.read().split('\n')

        # correction for txt file ending with \n
        if test_images[-1] == '':
            del test_images[-1]
        if len(test_images[0].split(',')) != len(test_images[1].split(',')):  # first row intro check
            del test_images[0]

        test_file_graph.close()
        self.test_size = len(test_images)
        self.test = (clip for clip in test_images)

    def init_augmentations(self, augmentation_probability):
        self.aug = augment_function(augmentation_probability)

    def get_train_batch(self, with_save_frames=False):

        if self.end_of_train:
            """new epoch - restart batch extraction"""
            self.train = (clip for clip in self.clips[:self.train_size])
            if self.shuffle_each_epoch:
                self.shuffle_train_set()
            self.end_of_train = False

        batch = {'LAT': [], 'PA' : []}
        labels = []
        batch_paths = []
        for i in range(self.train_batch_size):

            try:
                path = next(self.train)
            except StopIteration:
                self.end_of_train = True
                break

            label, sample = self.get_sample(path, with_save_frames)
            # for key, val in sample.items():
            #     sample[key] = np.tile(np.expand_dims(val, -1), [1, 1, 3])

            batch_paths.append(path)
            labels.append(label)
            batch['LAT'].append(sample['LAT'])
            batch['PA'].append(sample['PA'])

        return_batch = {'LAT': np.array(batch['LAT']),
                        'PA' : np.array(batch['PA'])}

        return return_batch, np.array(labels), batch_paths

    def get_test_batch(self, with_save_frames=False):

        if self.end_of_test:
            """new epoch - restart batch extraction"""
            if not self.test_split:
                if self.test_file_graph_path is not None:
                    self._load_test_set_from_file()
                else:
                    self.test = []
            else:
                self.test = (clip for clip in self.clips[self.train_size:])

            self.end_of_test = False

        batch = {'LAT': [], 'PA': []}
        labels = []
        batch_paths = []
        for i in range(self.test_batch_size):

            try:
                path = next(self.test)
            except StopIteration:
                self.end_of_test = True
                break

            label, sample = self.get_sample(path, with_save_frames, False)
            batch_paths.append(path)
            labels.append(label)
            batch['LAT'].append(sample['LAT'])
            batch['PA'].append(sample['PA'])

        return_batch = {'LAT': np.array(batch['LAT']),
                        'PA': np.array(batch['PA'])}

        return return_batch, np.array(labels), batch_paths

    def get_sample(self, path, with_save_frames=False, im_aug=True):

        label = int(os.path.split(os.path.split(os.path.split(path)[0])[0])[-1])

        images = self.load_file(path)

        if im_aug and self.image_augmentations:
            images_for_train = {}
            for angle, im in images.items():
                images_for_train[angle] = self.im_aug(im)
        else:
            images_for_train = images

        return label, images_for_train

    @staticmethod
    def load_file(path):
        base_name = '_'.join(path.split('_')[:-1])
        file_ext = path.split('.')[-1]
        im = {}
        im['LAT'] = np.tile(np.expand_dims(imread(base_name + '_LAT.' + file_ext), -1), [1, 1, 3])
        im['PA']  = np.tile(np.expand_dims(imread(base_name + '_PA.'  + file_ext), -1), [1, 1, 3])
        return im

    def im_aug(self, images_to_aug):
        return self.aug(image=images_to_aug)

    def shuffle_train_set(self):
        train_list = list(self.train)
        shuffle(train_list)
        self.train = (clip for clip in train_list)
        print('\nShuffling Train Set Order\n')

    def progress_one_batch(self):
        """
        progresses self.train without supplying a batch. Used to jump to a specific batch number
        :return: None
        """
        if self.end_of_train:
            """new epoch - restart batch extraction"""
            self.train = (clip for clip in self.clips[:self.train_size])
            if self.shuffle_each_epoch:
                self.shuffle_train_set()
            self.end_of_train = False

        for i in range(self.train_batch_size):
            try:
                path = next(self.train)
            except StopIteration:
                self.end_of_train = True
                break


class TrainLog:
    """
    Run A log for training a net.
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.verify_existance_or_create_dir(log_dir)
        self.logs = dict()

    @classmethod
    def construct_by_dir_and_label(cls, logs_directory, log_label):
        log_dir = os.path.join(logs_directory, log_label)
        cls.verify_existance_or_create_dir(logs_directory)
        cls.verify_existance_or_create_dir(log_dir)
        return cls(log_dir)

    def add_log(self, log_name):

        if log_name.find('.') == -1:
            log_file = log_name + '.txt'
        else:
            log_file = log_name

        log_path = os.path.join(self.log_dir, log_file)
        log = open(log_path, 'a')
        log.write('*' * len(log_name) + '\n')
        log.write(log_name.replace('_', ' ').upper() + '\n')
        log.write('*' * len(log_name) + '\n')

        log.close()

        self.logs[log_name] = log_path

        return {log_name: log_path}

    def get_log_names(self):
        return [key for key in self.logs.keys()]

    def get_log_path(self, log_name):
        return self.logs[log_name]

    def write_to_log(self, log_name, entries):

        # if not isinstance(entries, list) or not isinstance(entries, tuple):
        #     entries = [entries]
        log = open(self.logs[log_name], 'a')
        log.write('\n')

        for entry in entries:
            log.write(entry + '\n')

        log.close()

    @staticmethod
    def verify_existance_or_create_dir(path):
        if not os.path.exists(path):
            os.mkdir(path)

def augment_function(augmentation_probability):
    sometimes_aug = lambda aug: iaa.Sometimes(augmentation_probability, aug)  # sometimes augment function
    seq = iaa.Sequential(
        [
            sometimes_aug(iaa.Add((-50, 50))),
            iaa.Fliplr(augmentation_probability),
            sometimes_aug(iaa.Affine(
                scale={
                    'x': (0.8, 1.2),
                    'y': (0.8, 1.2),
                },
                translate_percent={
                    'x': (-0.1, 0.1),
                    'y': (-0.1, 0.1)
                },
                shear=(-30, 30),

            )),
            sometimes_aug(iaa.PerspectiveTransform(
                scale=(0.0, 0.08),
                keep_size=True,

            ))

        ]
    )
    return seq

if __name__ == '__main__':
    a = MultiChestVGG(use_batch_norm=True)
    merge_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./graph/test")
    with tf.Session() as sess:
        # Tensorboard
        train_writer.add_graph(sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        a.load_conv_weights_npz(r'./vgg_weights/vgg16_weights.npz', sess)


