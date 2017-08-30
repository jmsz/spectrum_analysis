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
#import becquerel
import numpy as np
#from scipy import optimize
from lmfit.models import GaussianModel
from lmfit.models import LinearModel

####functions

def PlotSpectrumRaw(channel_array, count_array):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    plt.plot(channel_array, count_array)
    plt.title('raw data: channels vs counts')
    plt.xlabel('channel')
    plt.ylabel('counts')
    plt.show()

def PlotSpectrumCalibrated(channel_array, count_array):
    """Plot spectrum data"""
    plt.figure(figsize=(10, 5))
    plt.plot(channel_array, count_array)
    #plt.legend()
    plt.title('data: energy vs counts')
    plt.xlabel('energy (kev)')
    plt.ylabel('counts')
    plt.show()

def PlotSpectrumZoom(channel_array, count_array):
    """Plot spectrum data"""
    plt.figure(figsize=(10,5))
    lowindex = int(raw_input('low index (channel number) for plot'))
    highindex = int(raw_input('high index (channel number) for plot'))
    
    plt.axis([lowindex, highindex, 0, max(count_array)])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    plt.plot(channel_array, count_array)
    plt.title('raw data: energy vs counts')
    plt.xlabel('energy')
    plt.ylabel('counts')
    plt.show()

def SelectROI():
    lowindex = raw_input("ROI low index")
    #lowindex = 8000
    if lowindex == '':
        lowindex = 0
    highindex = raw_input("ROI high index")
    #highindex = 11000
    if highindex == '':
        highindex = spec.channels(max)
    lowindex = int(lowindex)
    highindex = int(highindex)
    return lowindex, highindex

def FindPeaks(data_x, data_y, low_index, high_index):
    #data_x = data_x[low_index:high_index]
    #print ("data_x")
    #print data_x
    cut_data_y = data_y[low_index:high_index]
    #print ("data_y")
    #print data_y
    peak_y = max(data_y[low_index:high_index])
    for i in range (low_index,high_index):
        if data_y[i] == max(cut_data_y):
            #print("HERE!")
            #print (i)
            minindex=int(i-200)
            maxindex=int(i+200)
    #peak_x = data_x[i]
    ROIrange = [minindex, maxindex]
    return minindex, maxindex

def LinearBackgroundSubtractedCounts(x_val, x_energies, y_val, ROI_low, ROI_high):
    ch_bkg_low = (x_val> (ROI_low -100)) & (x_val<ROI_low)
    ch_bkg_high = (x_val> ROI_high) & (x_val < (ROI_high + 100))
    
    #x_val = x_energies
    
    x_bkg = np.concatenate((x_val[ch_bkg_low], x_val[ch_bkg_high]))
    y_bkg = np.concatenate((y_val[ch_bkg_low], y_val[ch_bkg_high]))
    
    plt.plot(x_bkg, y_bkg, 'o')
    
    fit = np.polyfit(x_bkg, y_bkg, 1)
    m = fit[0]
    c = fit[1]
    
    plt.axis([(ROI_low -100), (ROI_high + 100), 0, np.amax(y_val)])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    background = m*x_val + c
    y_bkg_corr = y_val- background
    #plt.plot(x_val, y_val)
    plt.plot(x_val, y_bkg_corr)
    plt.show()
    
    return y_bkg_corr

def FitGaussian(x,y):
    plt.figure(figsize=(20,10))
    mod = GaussianModel() #+ LinearModel()
    
    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))
    #print(out.sigma)
    
    #q = out.var_names()
    #print(q)
    #mod.best_values(fwhm)
    
    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()
    
    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    #plt.plot (xmaxima, ymaxima, 'o')
    plt.show()

#FitGaussian(spec.energies_kev[9750:10000], spec.counts_vals[9750:10000])

#spec = Spectrum.from_file('/Users/Asia/Desktop/HRGe/nanoMCA_strip6/3usrisetime.csv')
#plot_spectrum_raw(spec.channels, spec.counts_vals)

#FitGaussian(spec.energies_kev[9800:10100], spec.counts_vals[9800:10100])
#FitGaussian(spec.energies_kev[10800:11100], spec.counts_vals[10800:11100])

##IMPLEMENT INTERACTIVE PLOTLY PLOT

#import plotly.plotly as py
#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")
#table = ff.create_table(df)
#py.iplot(table, filename='jupyter/table1')

######main######

spec = Spectrum.from_file('/Users/Asia/Desktop/HRGe/nanoMCA_strip6/4usrisetime.csv')


spec = Spectrum.from_file('/Users/Asia/Desktop/HRGe/nanoMCA_strip6/8usrisetime.csv')

#add_calpoint = EnergyCalBase.add_calpoint(EnergyCalBase, 200, 1000)


plot_spectrum_raw(spec.channels, spec.counts_vals)


#linearbackgroundsubtractedcounts
#findpeaks

ROI_low, ROI_high = selectROI()
ROIrangemin, ROIrangemax = findpeaks(spec.channels, spec.counts_vals, ROI_low, ROI_high)
counts_vals_corrected = linearbackgroundsubtractedcounts(spec.channels, spec.energies_kev, spec.counts_vals, ROI_low, ROI_high)

FitGaussian(spec.energies_kev[ROIrangemin:ROIrangemax], counts_vals_corrected[ROIrangemin:ROIrangemax])