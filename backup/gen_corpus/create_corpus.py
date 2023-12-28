import sys
import librosa
import random
import soundfile
import tqdm
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from tqdm import tqdm
from scipy.io import wavfile

def mix_audio(signal, noise, snr):

    if len(noise)<len(signal):
      # if the audio is longer than the noise
      # play the noise in repeat for the duration of the audio
      noise = noise[np.arange(len(signal)) % len(noise)]
    else:
      # if the audio is not longer than the noise
      # a random chunk of the noise is added instead of starting at point 0
      beg = random.randint(0, len(noise)-len(signal))
      end = beg + len(signal)
      noise = noise[beg:end]

    # if the audio is shorter than the noi
    # this is important if loading resulted in
    # uint8 or uint16 types, because it would cause overflow
    # when squaring and calculating mean
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)

    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)

    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    # print(g, a, b)
    # mix the signals
    return a * signal + b * noise



def rir_gen(data,columns,dims,audio,mats,nr,path_rir):
    
    # Creates room with specified dimensions
    x_dim = random.uniform(dims[0],dims[1]) 
    y_dim = random.uniform(dims[2],dims[3])
    z_dim = random.uniform(dims[4],dims[5])
    
    # adding source under the condition that it is 
    # at least 50cm spaced from the edges
    x_s = random.uniform(0.5, (x_dim-0.5))
    y_s = random.uniform(0.5, (y_dim-0.5))
    z_s = random.uniform(0.5, (z_dim-0.5))

    # adding microphone with the same conditions
    # aforementioned as well as a distance of 1m 
    # from the source
    cond=False
    while cond==False:

        x_m = random.uniform(0.5, (x_dim-0.5))
        y_m = random.uniform(0.5, (y_dim-0.5))
        z_m = random.uniform(0.5, (z_dim-0.5))

        ps = np.array([x_s,y_s,z_s])
        pm = np.array([x_m,y_m,z_m])

        dist = np.linalg.norm(ps - pm)
        if dist >= 1.0:
            cond=True

    # Samples random materials from the material list for each
    # wall, floor and ceiling
    ceiling_ = mats.sample()
    ceiling_mat = ceiling_["MATERIAL"].to_string(index=False) 
    floor_ = mats.sample()
    floor_mat = floor_["MATERIAL"].to_string(index=False)
    east_ = mats.sample()
    east_mat = east_["MATERIAL"].to_string(index=False)
    west_ = mats.sample()
    west_mat = west_["MATERIAL"].to_string(index=False)
    north_ = mats.sample()
    north_mat = north_["MATERIAL"].to_string(index=False)
    south_ = mats.sample()
    south_mat = south_["MATERIAL"].to_string(index=False)
    
    ceiling_.reset_index(drop=True, inplace=True) 
    floor_.reset_index(drop=True, inplace=True)
    east_.reset_index(drop=True, inplace=True)
    west_.reset_index(drop=True, inplace=True)
    north_.reset_index(drop=True, inplace=True)  
    south_.reset_index(drop=True, inplace=True) 


    m = pra.make_materials(
        ceiling=ceiling_mat,
        floor=floor_mat,
        east=east_mat,
        west=west_mat,
        north=north_mat,
        south=south_mat,
    ) 
    
    room = pra.ShoeBox(
        [x_dim, y_dim, z_dim],
        fs=16000,
        materials=m,
        air_absorption=True
    )

    # Adding the source
    room.add_source(np.array([x_s,y_s,z_s]), signal=audio)

    # Adding the microphone
    R = np.array([[x_m],[y_m],[z_m]])
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    room.compute_rir()

    # stores the RIR in an audio file
    filename =  path_rir + 'rir_room_' + 'train_'+str(nr) +'.wav'
    waveform = room.rir[0][0]
    
    # Padding to 0.5s (all files are greatly under this val...)
    if len(waveform)>8000:
        waveform = waveform[0:8000]
        
    if len(waveform)<8000:
        dif = 8000-len(waveform)
        pad = [0]*dif
        waveform = np.concatenate((waveform,pad))

    # generates white noise and adds it to the signal
    noise = np.random.normal(0, 1, size=8000)
    waveform = mix_audio(waveform, noise, snr=30)

    # saves rir as a .wav file
    wavfile.write(filename,room.fs,waveform)
    
    # Creates the reference dataframe
    filename =pd.DataFrame(list([filename]))
    
    dimensions = pd.DataFrame([[x_dim,y_dim,z_dim]],
            columns=['x_dim', 'y_dim', 'z_dim'])
    
    source = pd.DataFrame([[x_s,y_s,z_s]],
            columns=['x_s', 'y_s', 'z_s'])
    
    micro = pd.DataFrame([[x_m,y_m,z_m]],
            columns=['x_m', 'y_m', 'z_m'])

    coefs = pd.concat([filename,
                       ceiling_,
                       floor_,
                       east_,
                       west_,
                       north_,
                       south_,
                       dimensions,
                       source,
                       micro], axis=1)
    
    coefs = coefs.values.tolist()
    
    data = pd.concat([data, pd.DataFrame(coefs,
                        columns=columns)],ignore_index=True)    

    return data

def main():

    path_rir = '/home/squintas/CORAM/rir_estimation/data/rir_clips/'
    nr_samples = 100000
    sample_rate = 16000
    
    mats_index = './materials_index_pra.csv'
    mats = pd.read_csv(mats_index)

    # x_min,x_max,y_min,y_max,z_min,z_,max
    room_size=[2,10, 2,10, 2,7]

    dirac = './burst_balloon.wav'
    dirac = './speech.wav'
    audio, fs = librosa.load(dirac,sr=sample_rate)

    columns=['rir_file',
             'MAT_c','c_125Hz','c_250Hz','c_500Hz','c_1kHz','c_2kHz','c_4kHz',
             'MAT_f','f_125Hz','f_250Hz','f_500Hz','f_1kHz','f_2kHz','f_4kHz',
             'MAT_e','e_125Hz','e_250Hz','e_500Hz','e_1kHz','e_2kHz','e_4kHz',
             'MAT_w','w_125Hz','w_250Hz','w_500Hz','w_1kHz','w_2kHz','w_4kHz',
             'MAT_n','n_125Hz','n_250Hz','n_500Hz','n_1kHz','n_2kHz','n_4kHz',
             'MAT_s','s_125Hz','s_250Hz','s_500Hz','s_1kHz','s_2kHz','s_4kHz',
             'room_x','room_y','room_z',
             'source_x','source_y','source_z',
             'micro_x','micro_y','micro_z']
    
    data = pd.DataFrame([],columns=columns)

    for i in tqdm(range(0, nr_samples), desc ="Generating RIR corpus..."):
        data = rir_gen(data,columns,room_size,audio,mats,i,path_rir)

    data.to_csv('../data/val_rir_corpus_materials.csv')
    

if __name__ == "__main__":
    main()


