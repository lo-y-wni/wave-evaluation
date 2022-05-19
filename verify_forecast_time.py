#!/usr/bin/env python
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def readobs(filelist):
    time       = []
    waveheight = []
    waveperoid = []

    for file in filelist:
        with open(file, 'r', encoding='shift_jis') as f:
            First  = True
            for line in f:
            #   Skip fist line
                if First :
                    First = False
                    continue

            #   FORMAT(I4,4I2,2I6,4(F6.2,F6.1),I6)
            #   2021-11-01 06:00:00
                time.append("{yyyy:04d}-{mm:02d}-{dd:02d} {hh:02d}:{mn:02d}:00".format(
                                                                    yyyy=int(line[0:4]),
                                                                      mm=int(line[4:6]),
                                                                      dd=int(line[6:8]),
                                                                      hh=int(line[8:10]),
                                                                      mn=int(line[10:12])))
                waveheight.append(float(line[24:30]))
                waveperoid.append(float(line[30:36]))

    obsdata = {'time':np.array(time),
               'height':np.ma.masked_values(waveheight, 99.99),
               'peroid':np.ma.masked_values(waveperoid, 999.9)}

    return obsdata

def readfcst(filelist):

    fcst_wavedata_height = {}
    fcst_wavedata_peroid = {}
    fcst_wavedata_time   = {}
    
    for file in filelist:
        idx    = file.index('.csv')
        date   = file[idx-10:idx-2]
    
        with open(file, newline='') as csvfile:
            rows  = csv.reader(csvfile)
            fcst_time       = []
            fcst_waveheight = []
            fcst_waveperoid = []
    
            First = True
            for row in rows:
                if First :
                    First = False
                    continue
                
                fcst_time.append(row[1])
                fcst_waveheight.append(float(row[2]))
                fcst_waveperoid.append(float(row[3]))
    
        fcst_wavedata_height[date] = np.ma.masked_values(fcst_waveheight, 99.99)
        fcst_wavedata_peroid[date] = np.ma.masked_values(fcst_waveperoid, 999.9)
        fcst_wavedata_time[date]   = np.array(fcst_time)

        del fcst_time
        del fcst_waveheight
        del fcst_waveperoid

    fcstdata = {'time':fcst_wavedata_time,
                'height':fcst_wavedata_height,
                'peroid':fcst_wavedata_peroid}

    return fcstdata

def get_index(items, targets):
    idx = []
    for target in targets:
        tmp = np.nonzero(target == items)
        if tmp is not None:
            idx.append(tmp[0][0])

    return np.array(idx)

def MAE(arr1, arr2):
    diff   = np.abs(arr1 - arr2)

    return diff.mean()

def BIAS(arr1, arr2):
    diff   = arr1 - arr2

    return diff.mean()

def RMSE(arr1, arr2):
    diff = (arr1 - arr2) ** 2

    return np.sqrt(diff.mean())


if __name__ == "__main__":

    station_name = 'RUM'

    if station_name == 'ISHK':
        obs_id = 'H611e'
    elif station_name == 'RUM':
        obs_id = 'H604e'
    elif station_name == 'KAS':
        obs_id = 'H207e'
        
    #obs files
    #change directory as need
    files = ['obs/OCT/'+obs_id+'.s2110.txt',
             'obs/NOV/'+obs_id+'.s2111.txt',
             'obs/DEC/'+obs_id+'.s2112.txt']
    obsdata = readobs(files)

    #fcst files
    #change directory as need
    files = glob.iglob("fcst/NPHAS_"+station_name+"-*.csv")
    fcstdata = readfcst(files)

    #give a specific initial time
    timelist     = ['20211001', '20211002', '20211003', '20211004', '20211005',
                    '20211006', '20211007', '20211008', '20211009', '20211010',
                    '20211011', '20211012', '20211013', '20211014', '20211015',
                    '20211016', '20211017', '20211018', '20211019', '20211020',
                    '20211021', '20211022', '20211023', '20211024', '20211025',
                    '20211026', '20211027', '20211028', '20211029', '20211030', '20211031',
                    '20211101', '20211102', '20211103', '20211104', '20211105',
                    '20211106', '20211107', '20211108', '20211109', '20211110',
                    '20211111', '20211112', '20211113', '20211114', '20211115',
                    '20211116', '20211117', '20211118', '20211119', '20211120',
                    '20211121', '20211122', '20211123', '20211124', '20211125',
                    '20211126', '20211127', '20211128', '20211129', '20211130',
                    '20211201', '20211202', '20211203', '20211204', '20211205',
                    '20211206', '20211207', '20211208', '20211209', '20211210',
                    '20211211', '20211212', '20211213', '20211214', '20211215',
                    '20211216', '20211217', '20211218', '20211219', '20211220',
                    '20211221', '20211222']

    nfcst      = fcstdata['time']['20211001'].size
    mae_array  = np.zeros(nfcst)
    rmse_array = np.zeros(nfcst)
    bias_array = np.zeros(nfcst)

    for fcst_time in range(nfcst):
        times       = []
        height_data = []

        for init_time in timelist:
            if fcst_time <= fcstdata['time'][init_time].size - 1:
                times.append(fcstdata['time'][init_time][fcst_time])
                height_data.append(fcstdata['height'][init_time][fcst_time])

        times       = np.array(times)
        height_data = np.ma.masked_values(height_data, 99.99)
        idx         = get_index(obsdata['time'], times)

        rmse_array[fcst_time] = RMSE(height_data, obsdata['height'][idx])
        bias_array[fcst_time] = BIAS(height_data, obsdata['height'][idx])
        mae_array[fcst_time]  =  MAE(height_data, obsdata['height'][idx])

        ## save obs and fcst data to a big array
        ## plot them in plt.his2d
        if fcst_time == 0:
            obs_his2d  = obsdata['height'][idx]
            fcst_his2d = height_data
        else:
            obs_his2d  = np.append(obs_his2d, obsdata['height'][idx])
            fcst_his2d = np.append(fcst_his2d, height_data)

        del times
        del height_data

    masked = obs_his2d == 99.99
    obs_his2d  =  obs_his2d[ np.logical_not(masked) ]
    fcst_his2d = fcst_his2d[ np.logical_not(masked) ]

    ## plot
    xvalues = np.arange(nfcst) * 3
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot(xvalues, mae_array,  label='MAE')
    ax1.plot(xvalues, bias_array, label='BIAS')
    ax1.plot(xvalues, rmse_array, label='RMSE')
    ax1.set_xlabel('Forecast hours')
    ax1.set_ylabel('wave height [m]')
    ax1.set_title('Station: {station}  Data from {begin} to {end}'.format(station=station_name, 
                                                                         begin=timelist[0],
                                                                         end=timelist[-1]))
    ax1.legend()


    # Calculate the point density
    xy = np.vstack([obs_his2d, fcst_his2d])
    z  = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    obs_his2d, fcst_his2d, z = obs_his2d[idx], fcst_his2d[idx], z[idx]

    ax2     = fig.add_subplot(1, 2, 2)
    density = ax2.scatter(obs_his2d, fcst_his2d, c=z, s=30)
    ax2.set_xlabel('observation')
    ax2.set_ylabel('forecast')
    ax2.set_xlim(0, 4.0)
    ax2.set_ylim(0, 4.0)
    ax2.set_title('Scatter-density Data from {begin} to {end}'.format(station=station_name, 
                                                                   begin=timelist[0],
                                                                     end=timelist[-1]))
    ax2.plot([0, 4.0], [0, 4.0], ls="--", c=".3")
    fig.colorbar(density)

   # plt.show()
    plt.savefig(station_name+'_verify_forecast_time.png', dpi=200)
        
