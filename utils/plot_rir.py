import pyroomacoustics as pra
import matplotlib.pyplot as plt
import librosa
import numpy as np
import sys

if len(sys.argv) != 2:
    print('usage: python plot_rir.py /path/to/rir/file.wav')
    sys.exit()

#path_rir = '../data/rir_clips/rir_room_29.wav'
path_rir = sys.argv[1]

# loads RIR file
y, sr = librosa.load(path_rir)

# plots RIR file
rir_time = np.arange(len(y)) / sr
fig = plt.figure()
plt.plot(rir_time, y)
fig.suptitle('RIR - file: '+path_rir,fontsize=45)
plt.xlabel('Time (seconds)', fontsize=40)
plt.ylabel('Amplitude',fontsize=40)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.show()
