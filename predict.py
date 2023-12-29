import sys
import yaml
import torch
import torchaudio
import torch.nn as nn
import lightning as L

from models.model import conv_model, conv_model_multi_task

# Load config file
yaml_path = './configs/parameters.yaml'
with open(yaml_path, 'r') as file:
    cfg = yaml.safe_load(file)

def print_coefs(y_hat):

    print("Frequency band absorption coefficients:\n")
    print("125Hz ---- {}".format(round(y_hat[0][0],4)))
    print("250Hz ---- {}".format(round(y_hat[0][1],4)))
    print("500Hz ---- {}".format(round(y_hat[0][2],4)))
    print("1000Hz --- {}".format(round(y_hat[0][3],4)))
    print("2000Hz --- {}".format(round(y_hat[0][4],4)))
    print("4000Hz --- {}".format(round(y_hat[0][5],4)))

def print_geom(g_hat):

    print("\nRoom geometry estimation:\n")
    print("Length (x): {} meters".format(round(g_hat[0][0],2)))
    print("Width  (y): {} meters".format(round(g_hat[0][1],2)))
    print("Height (z): {} meters".format(round(g_hat[0][2],2)))


def main():

    if len(sys.argv) != 2:
        print('usage: python predict.py /path/to/rir/file.wav')
        sys.exit()
    
    path_rir = sys.argv[1]

    waveform, sample_rate = torchaudio.load(path_rir)

    ckpt_path = "./pre-trained/checkpoint-base.ckpt"

    if cfg['multi-task'] == True:
        # Predicts mean absorption coefficients and room geometry
        cnn_model = conv_model_multi_task.load_from_checkpoint(ckpt_path)
        
        # disable randomness, dropout, etc...
        cnn_model.eval()

        # predicts with the model
        y_hat, g_hat = cnn_model(waveform.unsqueeze(dim=0))
        y_hat=y_hat.tolist()
        g_hat=g_hat.tolist()
        
        print_coefs(y_hat)
        print_geom(g_hat)
        

    else:
        # Predicts mean absorption coefficients (single task)
        cnn_model = conv_model.load_from_checkpoint(ckpt_path)
        cnn_model.eval()
        y_hat = cnn_model(waveform.unsqueeze(dim=0))
        y_hat=y_hat.tolist()
        
        print_coefs(y_hat)



if __name__ == "__main__":
    main()

