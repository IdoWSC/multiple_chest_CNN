from chest_vgg_train import *


if __name__ == '__main__':

    epochs = 50
    train_graph = config['train']['train_graph']
    train_batch_size = int(config['train']['train_batch_size'])
    test_split = float(config['train']['test_split'])

    if not test_split:
        test_graph = config['train']['test_graph']
    else:
        test_graph = None

    display_step = int(config['train']['display_step'])
    require_improvement = int(config['train']['require_improvement'])

    for key, val in config['train'].items():
        logging.info('{}: {}'.format(key, val))

    conv_weights_path = config['train']['conv_weights']

    input_angles = ['PA', 'LAT', 'multiple']

    learning_rates = [float(val) for val in config['parameter_search']['learning_rates'].split(', ')]

    use_augmentations = [bool(int(val)) for val in config['parameter_search']['use_augmentations'].split(', ')]
    for input_angle in input_angles:
        for use_augmentation in use_augmentations:
            for learning_rate in learning_rates:

                logging.info('\nstarting session for graph file: \n' + train_graph + '\nparameter settings: \n' )

                logging.info('input_angle: {}\nuse_augmentation" {}\nlearning_rate: {}\n'.format(input_angle,
                                                                                                 use_augmentation,
                                                                                                 learning_rate))

                param_model_save_dir = os.path.join(model_save_dir,
                                                    '{}Angle_augmentation{}_lr{}'.format(input_angle,
                                                                                         use_augmentation,
                                                                                         learning_rate))

                validation_accuracy = model_fit(train_graph, param_model_save_dir, conv_weights_path, input_angle,
                                                test_split, test_graph, epochs, train_batch_size,
                                                learning_rate=learning_rate, display_step=display_step,
                                                require_improvement=require_improvement,
                                                use_augmentations=use_augmentation)
                tf.reset_default_graph()
