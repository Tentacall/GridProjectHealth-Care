import yaml

config_path = 'config/preprocessing_config.yaml'
conf = yaml.safe_load(open(config_path,'r'))
train_image_path = conf['train_dataset']['image_path']
test_image_path = conf['test_dataset']['image_path']
train_labels_path = conf['train_dataset']['label_path']
test_labels_path = conf['test_dataset']['label_path']
train_batch_size = conf['train_dataset']['batch_size']
test_batch_size = conf['test_dataset']['batch_size']
columns = conf['train_dataset']['columns']
image_size = conf['image_preprocessing']['image_size']
itype = conf['image_preprocessing']['itype']