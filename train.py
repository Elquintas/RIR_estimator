import sys
import torch
import torchaudio
import torch.nn as nn
import yaml
import pandas as pd
import lightning as L
import dataloader.load
from models.model import conv_model   #, Lightning_CNN
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print('Error reading the config file')
        sys.exit()

def train_val_test_split(cfg):

    corpus = pd.read_csv(cfg['corpus_file'], sep=',')

    
    train, test = train_test_split(corpus, 
                                   test_size=0.3, 
                                   random_state=42)
    
    # Validation and Test set have the same size, hence 0.5
    test, val = train_test_split(test, 
                                 test_size=0.5, 
                                 random_state=42)

    train_file = cfg['data_dir'] + '/manifests/train.csv'
    test_file = cfg['data_dir'] + '/manifests/test.csv'
    val_file = cfg['data_dir'] + '/manifests/val.csv'

    # saves train, val, test .csv files
    train.to_csv(train_file,index=False)
    val.to_csv(test_file,index=False)
    test.to_csv(val_file,index=False)

    return train_file, val_file, test_file

def main():

    # Load parameters.yaml file (configs)
    config_path = './configs/parameters.yaml'
    cfg = load_config(config_path)

    sequential = dataloader.load.load_data

    # Splits the full corpus previously generated in manifests
    train, val, test = train_val_test_split(cfg)

    # Create Dataset objects for the different manifests
    train_set = sequential(train)
    test_set = sequential(test)
    val_set = sequential(val)

    # Define dataloaders
    trainloader = DataLoader(dataset=train_set,
                             batch_size=cfg['batch_train'],
                             num_workers=cfg['num_workers'])

    validloader = DataLoader(dataset=val_set,
                             batch_size=cfg['batch_test'],
                             num_workers=cfg['num_workers'])

    testloader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_test'])

    

    # define model and lightning module
    cnn_model = conv_model()

    # train model
    trainer = L.Trainer(max_epochs=cfg['epochs'])
    trainer.fit(cnn_model,trainloader,validloader)

    # test model
    trainer.test(cnn_model,dataloaders=testloader)

    # Check training progress on tensorboard by:
    # $ tensorboard --logdir=lightning_logs/

if __name__ == "__main__":
    main()
