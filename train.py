import sys
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import yaml
import pandas as pd
import lightning as L
import dataloader.load

from models.model import conv_model, conv_model_multi_task
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Load config file
yaml_path = './configs/parameters.yaml'
with open(yaml_path, 'r') as file:
    cfg = yaml.safe_load(file)


def save_results_to_df(results,test_df):

    filenames = test['rir_file']

    columns=['ref_125Hz','ref_250Hz','ref_500Hz','ref_1000Hz',
             'ref_2000Hz','ref_4000Hz','pred_125Hz','pred_250Hz','pred_500Hz',
             'pred_1000Hz','pred_2000Hz','pred_4000Hz','ref_x','ref_y','ref_z',
             'pred_x','pred_y','pred_z']
    
    df = pd.DataFrame(results.numpy(),columns=columns)
    df['filenames'] = filenames
    df.to_csv(cfg['results_dir']+'/results.csv')
    return


def train_val_test_split():

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
    #config_path = './configs/parameters.yaml'
    #cfg = load_config(config_path)

    sequential = dataloader.load.load_data

    # Splits the full corpus previously generated in manifests
    train, val, test = train_val_test_split()


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
    if cfg['multi-task']:
        # Predicts mean absorption coefficients and room geometry
        cnn_model = conv_model_multi_task()
    else:
        # Predicts mean absorption coefficients
        cnn_model = conv_model()

    # train model
    trainer = L.Trainer(max_epochs=cfg['epochs'])
    trainer.fit(cnn_model,trainloader,validloader)

    # test model
    trainer.test(cnn_model,dataloaders=testloader)
    
    # Test results are stored in model.test_results tensor
    save_results_to_df(cnn_model.test_results,test)
    



    # Check training progress on tensorboard by:
    # $ tensorboard --logdir=lightning_logs/

if __name__ == "__main__":
    main()
