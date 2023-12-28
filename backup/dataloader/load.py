import pandas as pd
import torch
import torchaudio
import pickle
import sys
import yaml

from torch.utils.data import Dataset,DataLoader

class load_data(Dataset):
    def __init__(self, filename):
        self.filename = filename
        xy = pd.read_csv(filename)
        
        self.rir_file = xy.values[:,1]

        self.c_MAT = xy.values[:,2]
        self.c_125 = xy.values[:,3]
        self.c_250 = xy.values[:,4]
        self.c_500 = xy.values[:,5]
        self.c_1000 = xy.values[:,6]
        self.c_2000 = xy.values[:,7]
        self.c_4000 = xy.values[:,8]

        self.f_MAT = xy.values[:,9]
        self.f_125 = xy.values[:,10]
        self.f_250 = xy.values[:,11]
        self.f_500 = xy.values[:,12]
        self.f_1000 = xy.values[:,13]
        self.f_2000 = xy.values[:,14]
        self.f_4000 = xy.values[:,15]

        self.e_MAT = xy.values[:,16]
        self.e_125 = xy.values[:,17]
        self.e_250 = xy.values[:,18]
        self.e_500 = xy.values[:,19]
        self.e_1000 = xy.values[:,20]
        self.e_2000 = xy.values[:,21]
        self.e_4000 = xy.values[:,22]

        self.w_MAT = xy.values[:,23]
        self.w_125 = xy.values[:,24]
        self.w_250 = xy.values[:,25]
        self.w_500 = xy.values[:,26]
        self.w_1000 = xy.values[:,27]
        self.w_2000 = xy.values[:,28]
        self.w_4000 = xy.values[:,29]

        self.n_MAT = xy.values[:,30]
        self.n_125 = xy.values[:,31]
        self.n_250 = xy.values[:,32]
        self.n_500 = xy.values[:,33]
        self.n_1000 = xy.values[:,34]
        self.n_2000 = xy.values[:,35]
        self.n_4000 = xy.values[:,36]

        self.s_MAT = xy.values[:,37]
        self.s_125 = xy.values[:,38]
        self.s_250 = xy.values[:,39]
        self.s_500 = xy.values[:,40]
        self.s_1000 = xy.values[:,41]
        self.s_2000 = xy.values[:,42]
        self.s_4000 = xy.values[:,43]

        self.room_x = xy.values[:,44]
        self.room_y = xy.values[:,45]
        self.room_z = xy.values[:,46]

        self.source_x = xy.values[:,47]
        self.source_y = xy.values[:,48]
        self.source_z = xy.values[:,49]

        self.micro_x = xy.values[:,50]
        self.micro_y = xy.values[:,51]
        self.micro_z = xy.values[:,52]

        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        
        waveform, sample_rate = torchaudio.load(self.rir_file[idx])
         
        return waveform, self.room_x[idx],self.room_y[idx],self.room_z[idx]

        #return waveform, self.c_125[idx], self.c_250[idx], self.c_500[idx],self.c_1000[idx], self.c_2000[idx], self.c_4000[idx]




