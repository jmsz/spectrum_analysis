import sys
import os.path
print sys.path

sys.path.append('/Users/Asia/becquerel-master/')
sys.path.append('/Users/Asia/becquerel-master/becquerel')
sys.path.append('/Users/Asia/becquerel-master/becquerel/core')

#sys.path.remove('/Users/Asia/becquerel-master/')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel/core')

import matplotlib.pyplot as plt
from becquerel import Spectrum
from becquerel import EnergyCalBase
import numpy as np
import lmfit as lmfit


def plot_spectrum_raw(channel_array, count_array):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    plt.plot(channel_array, count_array)
    plt.title('raw data: channels vs counts')
    plt.xlabel('channel')
    plt.ylabel('counts')
    plt.show()

def plot_spectrum_calibrated(channel_array, count_array):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    plt.plot(channel_array, count_array)
    plt.title('raw data: energy vs counts')
    plt.xlabel('energy')
    plt.ylabel('counts')
    plt.show()

def plot_spectrum_zoom(channel_array, count_array, lowindex, highindex):
    """Plot spectrum data"""
    plt.figure(figsize=(10,5))
    plt.axis([lowindex, highindex, 0, max(count_array)])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.plot(channel_array, count_array)
    plt.xlabel('energy/channels')
    plt.ylabel('counts')
    plt.show()

def plotfwhmrisetime(x_vals, y_vals, y_err):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, 'bo')
    plt.errorbar(x_vals, y_vals, xerr=0, yerr=y_err)
    plt.title('FWHM (kev) vs risetime (us)')
    plt.xlabel('time')
    plt.ylabel('')
    plt.show()

def plotfwhmgaptime(x_vals, y_vals, y_err):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    x_vals_1 = []
    x_vals_2 = []
    y_vals_1 = []
    y_vals_2 = []
    
    #y_err = int(y_err)
    plt.plot(x_vals, y_vals)
    plt.errorbar(x_vals, y_vals, xerr=0, yerr=y_err)
    plt.title('FWHM (kev) vs gaptime (us)')
    plt.xlabel('time')
    plt.ylabel('')
    plt.show()

def plotfwhm(outputfilename):
    
    #plot FWHM vs rise time
    plt.figure(figsize=(10, 5))
    
    datarisetime = np.array([], dtype=np.float)
    dataenergy = np.array([], dtype=np.float)
    dataFWHM = np.array([], dtype=np.float)
    dataerror = np.array([], dtype=np.float)
    datafile = str(outputfilename)
    labels = ['risetime', 'center', 'FWHM', 'FWHM error']
    df = pd.read_csv(datafile, header = None, usecols = [2,3,7,8], dtype = float, skiprows = 1, nrows = 14)
    datarisetime = df[2]
    dataenergy = df[3]
    dataFWHM = df[7]
    dataerror = df[8]
    plt.plot(datarisetime, dataFWHM, 'bo')
    plt.errorbar(datarisetime, dataFWHM, xerr=0, yerr=dataerror)
    
    datarisetime2 = np.array([], dtype=np.float)
    dataenergy2 = np.array([], dtype=np.float)
    dataFWHM2 = np.array([], dtype=np.float)
    dataerror2 = np.array([], dtype=np.float)
    datafile2 = str(outputfilename)
    labels = ['risetime', 'center', 'FWHM', 'FWHM error']
    df2 = pd.read_csv(datafile, header = None, usecols = [2,3,7,8], dtype = float, skiprows = 15, nrows = 14)
    datarisetime2 = df2[2]
    dataenergy2 = df2[3]
    dataFWHM2 = df2[7]
    dataerror2 = df2[8]
    plt.plot(datarisetime2, dataFWHM2, 'bo')
    plt.errorbar(datarisetime2, dataFWHM2, xerr=0, yerr=dataerror2)
    ax = plt.gca()
    ax.set_xscale('log')
    
    plt.show()
    
    outputfilename.close()

def main():
    print ("PlottingFunctions")

if __name__ == "__main__":
    main()