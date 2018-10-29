import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os
from scipy.misc import imread, imresize
from progress.bar import Bar
from random import shuffle


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class MultiChestVGG:

    place_holder_shape = [None, 224, 224, 3]
    conv_layers = 5
    conv_units = (2, 2, 3, 3, 3)
    conv_filter_sizes = (64, 128, 256, 512, 512)
    fc_layers = 3
    fc_layer_sizes = (4096, 4096, 1000)
    fc1_layer_in_size = 25088


    def __init__(self, im_shape=None, num_of_classes=2, use_softmax=False, is_training=True):

        if im_shape is not None and im_shape[0] is not None:
            self.place_holder_shape = [None]
            self.place_holder_shape.extend(im_shape)
        elif im_shape is not None:
            self.place_holder_shape = im_shape

        self.is_training = is_training
        self.num_of_classes = num_of_classes
        self.use_softmax = use_softmax

        self.PA_images  = tf.placeholder(tf.float32, self.place_holder_shape, name='PA_images')
        self.LAT_images = tf.placeholder(tf.float32, self.place_holder_shape, name='LAT_images')

        self.parameters = dict()
        self.layers = dict()

        self._build_conv_parameters()
        self._build_fc_parameters()
        self._build_net()
        self._build_classifier_layer()

        self.add_summeries()

    def _build_conv_parameters(self, trainable=False):
        """
        :param layers: int - number of layers
        :param units: iterable of ints, amount of units per layer, len(units) = layers
        :param out_filtsers_size: iterable of ints, amount of filters in each unit of the layer,
        len(out_filters_size) = layers
        :return:
        """
        out_filters = 0  # init

        with tf.device('/cpu:0'):
            for layer in range(self.conv_layers):
                for unit in range(self.conv_units[layer]):

                    if not layer and not unit:
                        in_filters = self.PA_images.get_shape()[-1]
                    else:
                        in_filters = out_filters

                    out_filters = self.conv_filter_sizes[layer]

                    name_extension = '{}_{}'.format(layer + 1, unit + 1)
                    with tf.variable_scope('conv{}'.format(name_extension)):
                        self.parameters['conv{}_W'.format(name_extension)] = \
                            tf.get_variable('weights',
                                            shape=[3, 3, in_filters, out_filters],
                                            dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            trainable=trainable)

                        self.parameters['conv{}_b'.format(name_extension)] = \
                            tf.get_variable('biases',
                                            shape=out_filters,
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer(),
                                            trainable=trainable)

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
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        out_split.append(model_net('gpu', i, x={'PA' : in_splits['x_PA'][i],
                                                                'LAT': in_splits['x_LAT'][i]}))

            return tf.concat(out_split, axis=0)

        def build_structure_for_cpu(**kwargs):
            in_splits = {}
            for k, v in kwargs.items():
                in_splits[k] = v
            with tf.device('/cpu:0'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    out_split = model_net('cpu', 0, x={'PA' : in_splits['x_PA'],
                                                       'LAT': in_splits['x_LAT']})
            return out_split

        def model_net(device_type, device, x):

            fc_in = []
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            for angle, scan in x.items():
                # zero-mean input
                with tf.variable_scope('preprocess_{}_{}'.format(angle, device)) as scope:
                    layer_input = scan - mean

                for layer in range(self.conv_layers):
                    for unit in range(self.conv_units[layer]):
                        name_extension = '{}_{}'.format(layer + 1, unit + 1)
                        with tf.variable_scope('conv{}_{}_{}'.format(name_extension, angle, device)):
                            conv = tf.nn.conv2d(layer_input,
                                                self.parameters['conv{}_W'.format(name_extension)],
                                                [1, 1, 1, 1],
                                                padding='SAME')

                            out = tf.nn.bias_add(conv, self.parameters['conv{}_b'.format(name_extension)])

                            self.layers['conv{}_{}_{}_{}'.format(name_extension, angle, device_type, str(device))] = \
                                tf.nn.relu(out)

                        layer_input = self.layers['conv{}_{}_{}_{}'.format(name_extension, angle,
                                                                           device_type, str(device))]

                    self.layers['pool{}_{}_{}_{}'.format(layer, angle, device_type, str(device))] = \
                        tf.nn.max_pool(layer_input,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       name='pool1')

                    layer_input = self.layers['pool{}_{}_{}_{}'.format(layer, angle, device_type, str(device))]
                fc_in.append(tf.reshape(layer_input,
                                        [-1, self.fc1_layer_in_size]))  # squash tensor coming from conv to fc

            layer_input = tf.concat(fc_in, axis=1)

            for layer in range(self.fc_layers):
                name_extension = '{}'.format(self.conv_layers + layer + 1)
                with tf.variable_scope('fc{}_{}'.format(name_extension, device)):
                    fc_mul = tf.matmul(layer_input, self.parameters['fc{}_W'.format(name_extension)])
                    fc_lin = tf.nn.bias_add(fc_mul, self.parameters['fc{}_b'.format(name_extension)])

                    self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))] = tf.nn.relu(fc_lin)

                layer_input = self.layers['fc{}_{}_{}'.format(name_extension, device_type, str(device))]

            with tf.variable_scope('final_{}'.format(device)):
                fc_mul = tf.matmul(layer_input, self.parameters['final_W'])
                fc_lin = tf.nn.bias_add(fc_mul, self.parameters['final_b'])

            layer_input = fc_lin
            net_output = layer_input
            return net_output

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
            sess.run(self.parameters[k].assign(weights[k]))

    def add_summeries(self):
        for parameter, value in self.parameters.items():
            tf.summary.histogram(parameter, value)
        for activation, value in self.layers.items():
            tf.summary.histogram(activation, value)


class TrainNet:

    @classmethod
    def with_construct_BatchOrganiser(cls, net, shuffled_loc, batch_size, learning_rate=0.001, learning_rate_decay=0.0,
                                      epsilon=1e-8, test_split=0.2, test_file=None):

        data_set = BatchOrganiser(shuffled_loc, batch_size, test_split, test_file)
        trainer = cls(net, data_set, learning_rate, learning_rate_decay, epsilon)
        return trainer

    def __init__(self, net, data_set, learning_rate=0.001, learning_rate_decay=0.0, epsilon=1e-8):

        self.net = net
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epsilon = epsilon

        self.data_set = data_set
        self.train_size = self.data_set.train_size
        self.test_size = self.data_set.test_size
        self.end_of_train = self.data_set.end_of_train

        self.train_batch_size = self.data_set.train_batch_size
        self.test_batch_size = self.data_set.test_batch_size

        self._build_trainer()

    def _build_trainer(self):

        with tf.device('/cpu:0'):
            self.ground_truth = tf.placeholder(tf.float32, (None,), name='y')
            print('\nBuilding xentropy Loss\n')
            with tf.variable_scope('cost'):
                if self.net.use_softmax or self.net.num_of_classes:
                    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                       logits=self.net.probs))
                else:

                    self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                       logits=self.net.probs))
                tf.summary.scalar('train_accuracy', self.cost)

        with tf.variable_scope('train'):
            self.learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph,
                                                       decay=self.learning_rate_decay,
                                                       epsilon=self.epsilon).minimize(self.net.cost,
                                                                                      colocate_gradients_with_ops=True)

        self._merged_train_summary = tf.summary.merge_all()
        with tf.variable_scope('validation_set'):
            self.predicted = tf.placeholder(tf.int64, (None,), name='y_pred')
            self.validation_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.net.ground_truth, self.predicted), tf.float32))
            self.validation_accuracy_summary = tf.summary.scalar('validation_accuracy', self.validation_accuracy)


        # saver
        self.saver = tf.train.Saver()

    def load_weights(self, weight_file, sess):
        return self.net.load_weights_from_saver(weight_file, sess, self.saver)

    def load_weights_npz(self, weight_file, sess):
        return self.net.load_weights_npz(weight_file, sess)

    def load_conv_weights_npz(self, weight_file, sess):
        return self.net.load_conv_weights_npz(weight_file, sess)

    def batch_advance(self):

        self.train_batch, self.train_labels, self.batch_paths = self.data_set.get_train_batch()

    def batch_pass(self, sess, train_writer=None, batch_number=0, add_summary=False, add_metadata=False):
        def run_summary():
            feed_dict = {self.net.PA_images:  self.train_batch['PA'],
                         self.net.LAT_images: self.train_batch['LAT'],
                         self.ground_truth: self.train_labels,
                         self.learning_rate_ph: self.learning_rate}
            summary = sess.run(self._merged_train_summary, feed_dict=feed_dict)
            train_writer.add_summary(summary, batch_number)

        def run_batch():
            feed_dict = {self.net.PA_images:  self.train_batch['PA'],
                         self.net.LAT_images: self.train_batch['LAT'],
                         self.net.ground_truth: self.train_labels,
                         self.learning_rate_ph: self.learning_rate}
            if add_metadata:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, c = sess.run([self.optimizer, self.cost],
                                      feed_dict=feed_dict,
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step{}'.format(batch_number))
            else:
                _, c = sess.run([self.optimizer, self.cost], feed_dict=feed_dict)
            return c

        # BATCH LOADING
        self.batch_advance()

        if add_summary:
            run_summary()

        # Run optimization op (backprop)
        cost = run_batch()

        return cost, self.batch_paths

    def train_accuracy(self, sess):

        return sess.run(self.cost, {self.net.images: self.train_batch,
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

            if not test_batch.size:
                break
            feed_dict = {self.net.images: test_batch,
                         self.ground_truth: test_labels}

            net_output = sess.run(self.net.output, feed_dict=feed_dict)
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

    @staticmethod
    def init_variables(sess):

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)


class BatchOrganiser:
    """class for arranging batches for training.
    if test split == 0 then a path of test folder given at test_graph_loc will be used to load test_graph_loc.
    else test set will be empty"""
    def __init__(self, train_file_graph_path, train_batch_size, test_batch_size=None, test_split=0.2, test_file_graph_path=None,
                 num_of_frames=16, require_resize=False, resize_height=112, resize_width=112,
                 save_frame_location='', shuffle_each_epoch=False):

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

            label, sample = self.get_sample(path, with_save_frames)
            batch_paths.append(path)
            labels.append(label)
            batch['LAT'].append(sample['LAT'])
            batch['PA'].append(sample['PA'])

        return_batch = {'LAT': np.array(batch['LAT']),
                        'PA': np.array(batch['PA'])}

        return return_batch, np.array(labels), batch_paths

    def get_sample(self, path, with_save_frames=False):

        label = os.path.split(os.path.split(path)[0])[-1]

        images = self.load_file(path)

        return label, images

    @staticmethod
    def load_file(path):
        base_name = '_'.join(path.split('_')[:-1])
        file_ext = path.split('.')[-1]
        im = {}
        im['LAT'] = imread(base_name + '_LAT' + file_ext)
        im['PA']  = imread(base_name + '_PA'  + file_ext)
        return im

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


if __name__ == '__main__':
    a = MultiChestVGG()
    merge_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./graph/test")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Tensorboard
        init = tf.global_variables_initializer()
        sess.run(init)
        a.load_conv_weights_npz(r'./vgg_weights/vgg16_weights.npz', sess)
        train_writer.add_graph(sess.graph)

