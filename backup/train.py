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

torch.manual_seed(42)
torch.cuda.manual_seed(42)


def main():

    train_file = './data/train_rir_corpus_materials.csv'
    test_file = './data/test_rir_corpus_materials.csv'
    val_file = './data/val_rir_corpus_materials.csv'

    sequential = dataloader.load.load_data

    train_set = sequential(train_file)
    test_set = sequential(test_file)
    val_set = sequential(val_file)

    trainloader = DataLoader(dataset=train_set,
                             batch_size=20,
                             num_workers=1)

    validloader = DataLoader(dataset=val_set,
                             batch_size=16,
                             num_workers=1)

    testloader = DataLoader(dataset=test_set,
                             batch_size=16)


    # define model and lightning module
    cnn_model = conv_model()

    # train model
    trainer = L.Trainer(max_epochs=20)
    trainer.fit(cnn_model,trainloader,validloader)

    trainer.test(cnn_model,dataloaders=testloader)


if __name__ == "__main__":
    main()
