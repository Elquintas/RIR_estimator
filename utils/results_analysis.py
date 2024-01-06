import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

def main():
    
    if len(sys.argv) != 3:
        print('usage: python results_analysis.py /path/to/results/file.csv /path/to/test.csv')
        sys.exit()
    
    res_filename = sys.argv[1]
    test_filename = sys.argv[2] 

    print(res_filename,test_filename)

    df = pd.read_csv(res_filename, sep=',')
    df_t = pd.read_csv(test_filename, sep=',')

        

    # Gets sabine coefficients from theoretical rt60.
    sab_coefs=[]
    eyr_coefs=[]
    for i in range(len(df_t)):
        sab = pra.inverse_sabine(df_t['rt60'][i],
                [df_t['room_x'][i],df_t['room_y'][i],df_t['room_z'][i]])
        sab_coefs.append(sab[0])
        eyr = -math.log(1-sab[0])
        eyr_coefs.append(eyr)

    sab_coefs = pd.DataFrame(sab_coefs)
    eyr_coefs = pd.DataFrame(eyr_coefs)


    # Absolute error
    band1 = (df['ref_125Hz']-df['pred_125Hz']).abs()
    band2 = (df['ref_250Hz']-df['pred_250Hz']).abs()
    band3 = (df['ref_500Hz']-df['pred_500Hz']).abs()
    band4 = (df['ref_1000Hz']-df['pred_1000Hz']).abs()
    band5 = (df['ref_2000Hz']-df['pred_2000Hz']).abs()
    band6 = (df['ref_4000Hz']-df['pred_4000Hz']).abs()

    # Full band (average) - REFERENCE
    ref_bands = (df['ref_125Hz']+df['ref_250Hz']+df['ref_500Hz']+
                df['ref_1000Hz']+df['ref_2000Hz']+df['ref_4000Hz'])/6

    pred_bands = (df['pred_125Hz']+df['pred_250Hz']+df['pred_500Hz']+
                df['pred_1000Hz']+df['pred_2000Hz']+df['pred_4000Hz'])/6

    avg_coefs_pred = (ref_bands - pred_bands).abs() 

    avg_coefs_sab = (ref_bands - sab_coefs[0]).abs()

    avg_coefs_eyr = (ref_bands - eyr_coefs[0]).abs()



    dist_x = (df['ref_x']-df['pred_x']).abs()
    dist_y = (df['ref_y']-df['pred_y']).abs()
    dist_z = (df['ref_z']-df['pred_z']).abs()




    # Plots error bars - absorption coefficients
    bands = ['125Hz','250Hz','500Hz','1000Hz','2000Hz','4000Hz',
             'All bands','Sabine','Eyring']
    x_pos = np.arange(len(bands))
    CTEs = [band1.mean(),band2.mean(),band3.mean(),
            band4.mean(),band5.mean(),band6.mean(),
            avg_coefs_pred.mean(),avg_coefs_sab.mean(),avg_coefs_eyr.mean()]
    error = [band1.std(),band2.std(),band3.std(),
            band4.std(),band5.std(),band6.std(),
            avg_coefs_pred.std(),avg_coefs_sab.std(),avg_coefs_eyr.std()]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, 
            ecolor='black', capsize=10)
    ax.set_ylabel('Absolute Error',fontsize=50)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bands)
    ax.set_title('Frequency bands absorption level coefficients',
            fontsize=50)
    ax.yaxis.grid(True)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.show()




    # Plots error bars - room measurements
    dist = ['Length (x)', 'Width (y)', 'Height (z)']
    x_pos = np.arange(len(dist))
    CTEs = [dist_x.mean(),dist_y.mean(),dist_z.mean()]
    error = [dist_x.std(),dist_y.std(),dist_z.std()]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5,
            ecolor='black', capsize=10)
    ax.set_ylabel('Absolute Error (meters)',fontsize=50)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dist)
    ax.set_title('Room measurements (x,y,z)',
            fontsize=50)
    ax.yaxis.grid(True)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.show()
    

if __name__ == "__main__":
    main()

