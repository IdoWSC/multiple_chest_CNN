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

    use_batch_norm = [True, False]
    use_common_conv_weights = [True, False]
    conv_trainable = [True, False]

    optimization_iters = []

    for input_angle in input_angles:
        for use_augmentation in use_augmentations:
            for learning_rate in learning_rates:
                for batch_norm in use_batch_norm:
                    for common_conv_weights in use_common_conv_weights:
                        for train_conv in conv_trainable:



                            logging.info('\nstarting session for graph file: \n' + train_graph +
                                         '\nparameter settings: \n')

                            logging.info('input_angle: {}\n'
                                         'use_augmentation: {}\n'
                                         'learning_rate: {}\n'
                                         'batch_norm: {}\n'
                                         'common_conv: {}\n'
                                         'conv_trainable: {}\n'.format(input_angle,
                                                                       use_augmentation,
                                                                       learning_rate,
                                                                       batch_norm,
                                                                       common_conv_weights,
                                                                       train_conv))

                            param_model_save_dir = os.path.join(model_save_dir,
                                                                '{}Angle_augmentation{}_lr{}_BN{}_commonWeights{}'
                                                                'convTrain'.format(input_angle,
                                                                                   use_augmentation,
                                                                                   learning_rate,
                                                                                   common_conv_weights,
                                                                                   train_conv))

                            validation_accuracy = model_fit(train_graph, param_model_save_dir, conv_weights_path,
                                                            input_angle,test_split, test_graph, epochs,
                                                            train_batch_size, learning_rate=learning_rate,
                                                            display_step=display_step,
                                                            require_improvement=require_improvement,
                                                            use_augmentations=use_augmentation,
                                                            use_batch_norm=batch_norm,
                                                            use_common_conv_weights=common_conv_weights,
                                                            conv_trainable=train_conv)

                            iter_dict = {'input_angle': input_angle,
                                         'use_augmentation': use_augmentation,
                                         'learning_rate': learning_rate,
                                         'batch_norm': batch_norm,
                                         'common_conv': common_conv_weights,
                                         'conv_trainable': train_conv,
                                         'validation_accuracy': validation_accuracy}
                            optimization_iters.append(iter_dict)
                            tf.reset_default_graph()

    logging.info('optimization results:\n')
    optimization_iters = sorted(optimization_iters, key=lambda i: i['validation_accuracy'], reverse=True)
    for iteration in optimization_iters:
        logging.info('input_angle - {}, '
                     'use_augmentation - {}, '
                     'learning_rate - {}, ' 
                     'batch_norm - {}, ' 
                     'common_conv - {}, '
                     'conv_trainable - {}:\n'
                     'validation top accuracy: {}\n'.format(iteration['input_angle'],
                                                            iteration['use_augmentation'],
                                                            iteration['learning_rate'],
                                                            iteration['batch_norm'],
                                                            iteration['common_conv'],
                                                            iteration['conv_trainable'],
                                                            iteration['validation_accuracy']))


