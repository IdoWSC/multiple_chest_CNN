import time
from configparser import ConfigParser
import logging
import os
from chest_vgg import *


def init_logging(out_path='', level=logging.INFO):

    if not out_path:
        output_folder = r'C:\Projects\replay_animation_detection\inception_optimization_results'
        out_path = os.path.join(output_folder, 'graphics_to_db_conversion.log')
    handlers = [logging.FileHandler(out_path),
                logging.StreamHandler()]

    msg_format = '[%(asctime)s] %(message)s'
    formatter = logging.Formatter(msg_format)

    logging.basicConfig(handlers=handlers, level=level, format=msg_format)


config = ConfigParser()
config.read('init.ini')

now = time.localtime(time.time())

label = str(now.tm_year)[-2:] + '{0:02d}'.format(now.tm_mon) + '{0:02d}'.format(now.tm_mday) + '_' + \
        '{0:02d}'.format(now.tm_hour) + '{0:02d}'.format(now.tm_min) + '{0:02d}'.format(now.tm_sec)

models_dir = config['train']['models_dir']
label = label + '_' + config['train']['label_addition']
model_save_dir = os.path.join(models_dir, label)
os.makedirs(model_save_dir, exist_ok=True)
init_logging(os.path.join(model_save_dir, 'train.log'))
logging.info('train label is {}'.format(label))


def model_fit(random_train_mapping, model_save_dir, conv_weights_path, test_split=0.2, random_test_mapping=None,
              epochs=20, train_batch_size=10, test_batch_size=None, train_summary=True, parameter_string='',
              learning_rate=0.0001, display_step=100, start_from_batch=0, dropout=0.5):

    net = MultiChestVGG()

    logging.info('\ncreating BatchOrganiser')
    samples = BatchOrganiser(random_train_mapping, train_batch_size, test_batch_size, test_split, random_test_mapping,
                             shuffle_each_epoch=True)

    logging.info(('\ncreating TrainNet object with learning rate of {}'.format(learning_rate)))
    trainer = TrainNet(net,
                       samples,
                       learning_rate=learning_rate)

    train_writer = tf.summary.FileWriter(os.path.join(os.path.join(model_save_dir, 'train_graph'), 'graph'))

    logging.info('\ntrain set size: {0} \ntest set size: {1}\n'.format(trainer.train_size, trainer.test_size))

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    save_path = os.path.join(model_save_dir, 'model_weights')

    # Best validation accuracy seen so far.
    best_validation_accuracy = 0

    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0

    # Stop optimization if no improvement found in this many iterations.
    require_improvement = 1000

    logging.info("\nModel ready.\n")

    num_of_train_batches = trainer.calc_number_of_train_batches()
    num_of_test_batches  = trainer.calc_number_of_test_batches()

    unimprovement_break_cond = False

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        trainer.assign_session(sess)
        # Tensorboard
        train_writer.add_graph(sess.graph)

        trainer.init_variables()
        trainer.load_conv_weights_npz(conv_weights_path, sess)

        trainer.jump_forward_to_batch(start_from_batch)
        global_step = trainer.get_global_step()
        start_epoch = int(global_step / num_of_train_batches)

        logging.info("\nStarting Training from batch number {}\n".format(start_from_batch))
        # epoch cycle
        for epoch in range(start_epoch, epochs):

            if unimprovement_break_cond:
                break

        # Training cycle
            # initialize batch num each epoch. in case epoch = 0 than batch_num is equal to start_from_batch
            batch_num = trainer.determine_batch_num(epoch, num_of_train_batches, global_step, start_epoch)

            while True:

                if not global_step % display_step:

                    c,  batch_paths, batch_errors_vec, global_step = trainer.batch_pass(sess,
                                                                                        dropout,
                                                                                        train_writer,
                                                                                        add_summary=True,
                                                                                        add_metadata=True)
                    logging.info('batch {0} cost is : {1}'.format(global_step, c))

                elif not global_step % 5:  # print progress every 5 steps
                    # try batch pass, a value error can be caused from inner objects trying to predict a 0 length batch.

                    c,  batch_paths, batch_errors_vec, global_step = trainer.batch_pass(sess,
                                                                                        dropout,
                                                                                        train_writer, add_summary=True)
                    logging.info('batch {0} cost is : {1}'.format(global_step, c))
                else:

                    c, batch_paths, batch_errors_vec, global_step = trainer.batch_pass(sess, train_writer)

                # Display logs per epoch
                if global_step % display_step == 1 and train_summary:  # enter validation set evaluation

                    # Calculate the accuracy on the training-batch.
                    acc_train = trainer.train_accuracy(sess)

                    acc_validation, summary, print_to_log = trainer.get_validation_accuracy(sess)
                    train_writer.add_summary(summary, global_step)
                    # print_to_log.insert(0, 'validation_accuracy = {}'.format(acc_validation))
                    # print_to_log.append('#' * 30)

                    if acc_validation < best_validation_accuracy:
                        # Update the best-known validation accuracy.
                        best_validation_accuracy = acc_validation

                        # Set the iteration for the last improvement to current.
                        last_improvement = global_step

                        # Save all variables of the TensorFlow graph to file.
                        trainer.save_model(sess, save_path)

                        # A string to be printed below, shows improvement found.
                        improved_str = '; best result, saving weights'
                    else:
                        # An empty string to be printed below.
                        # Shows that no improvement was found.
                        improved_str = ''
                    # Status-message for printing.
                    msg = "Iter: {0}, Train-Batch Accuracy: {1}, Validation Acc: " \
                          "{2} {3}".format(global_step, acc_train, acc_validation, improved_str)

                    # Print it.
                    logging.info(msg)

                    # print("batch*epoch num: ", overall_batch_num)

                    # unimprovement break condition
                    if global_step - last_improvement > require_improvement:
                        logging.info('no imporvement for {} itterations'.format(global_step - last_improvement))
                        # unimprovement_break_cond = True
                        # break

                batch_num += 1
                if trainer.check_end_of_train_set():
                    break

    logging.info("Optimization Finished!")

    return best_validation_accuracy


if __name__ == '__main__':

    train_graph = config['train']['train_graph']
    train_batch_size = int(config['train']['train_batch_size'])
    test_split = float(config['train']['test_split'])
    display_step = int(config['train']['display_step'])
    learning_rate = float(config['train']['learning_rate'])
    conv_weights_path = config['train']['conv_weights']

    if not test_split:
        test_graph = config['train']['test_graph']
    else:
        test_graph = None

    try:
        start_from_batch = int(config['train']['start_from_batch'])
    except KeyError:
        start_from_batch = 0

    try:
        epochs = int(config['train']['epochs'])
    except KeyError:
        epochs = 50

    parameter_string = ''

    logging.info('\nstarting session for graph file: \n' + train_graph + '\nparameter settings: \n' + parameter_string +
          '\n')

    validation_accuracy = model_fit(train_graph, model_save_dir, conv_weights_path, test_split, test_graph, epochs,
                                    train_batch_size, learning_rate=learning_rate)

