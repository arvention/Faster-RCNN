from pascal_voc import get_voc_loader


def get_loader(config):
    """
    returns train and test data loader
    """
    train_data_loader = None
    test_data_loader = None

    if config.dataset == 'voc':
        if config.mode == 'train':
            data_path = config.voc_data_path + config.voc_train_data_path
            train_data_loader = get_voc_loader(data_path=data_path,
                                               dataset=config.dataset,
                                               batch_size=config.batch_size,
                                               mode='train')
            test_data_loader = get_voc_loader(data_path=data_path,
                                              dataset=config.dataset,
                                              batch_size=config.batch_size,
                                              mode='val')

        elif config.mode == 'test':
            data_path = config.voc_data_path + config.voc_test_data_path
            test_data_loader = get_voc_loader(data_path=data_path,
                                              dataset=config.dataset,
                                              batch_size=config.batch_size,
                                              mode='test')

    return train_data_loader, test_data_loader
