#!/usr/bin/python

#----------------------------------------------------------------------------------------
# Title: Spectrum Analysis Script
# Institution: Lawrence Berkeley National Laboratory
# Author: J. Szornel
# Version: 1.0
# Created: 08-15-17
# Last edited: 08-15-17
# Description: Analyzes spectra from labzy spectrum files (currently implementing multiple files,
#              streamlined process. Extracts fwhm values.
#----------------------------------------------------------------------------------------

import os
import sys
import os.path
print sys.path

sys.path.append('/Users/Asia/becquerel-master/')
sys.path.append('/Users/Asia/becquerel-master/becquerel')
sys.path.append('/Users/Asia/becquerel-master/becquerel/core')

#sys.path.remove('/Users/Asia/becquerel-master/')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel/core')

from SpectrumAnalysisFunctions import *
from PlottingFunctions import *
from scipy.ndimage import filters

def main():
    
    directory = '/Users/Asia/Desktop/HRGe/nanoMCA_strip6/risetime/' #data file locaton
    outputfilename = '../strip64file.csv'
    
    #initalize output file
    outputfile = open('../' + str(outputfilename), 'a')
    outputfile.write('filename' + ', ')
    outputfile.write('livetime' + ', ')
    outputfile.write('risetime' + ', ')
    #outputfile.write('gaptime' + ', ')
    outputfile.write('center' + ', ')
    outputfile.write('sigma' + ', ')
    outputfile.write('height' + ', ')
    outputfile.write('amplitude (# counts)' + ', ')
    outputfile.write('FWHM' + ', ')
    outputfile.write('FWHM error' + '\n')
    outputfile.close()

    filelist = []

    for filename in os.listdir(directory):
        if filename.endswith(".lzs"):
            filelist.append(os.path.join(directory, filename))
            continue
        else:
            continue
    if len(filelist) == 0:
        print("no files found")
        sys.exit()
    else:
        print('\n' + 'files to analyze:')
        for i in filelist:
            print i
        raw_input('continue')

    for filefromlist in filelist:
        
        spectrumlivetime = 0.0
        spectrumrisetime = 0.0
        spectrumgaptime = 0.0
        peakcenter = 0.0
        peaksigma = 0.0
        peakheight = 0.0
        peakamplitude = 0.0
        peakfwhm = 0.0
        peakfwhmerr = 0.0
        
        spec = Spectrum.from_file(filefromlist)
        spectrumrisetime = raw_input('risetime for spectrum in us (check filename):')
        if float(spectrumrisetime) < 0:
            print('invalid entry')
            sys.exit()
        spectrumgaptime = 0.35 ###TODO
        spectrumlivetime = spec.livetime
        xvalue = spec.channels
        #print(xvalue)
        #convolvingfunction = gaussianfunction(700, xvalue, 2)
        #yvalue = convolvingfunction(xvalue)
        #print(yvalue)
        yvalue = filters.gaussian_filter(spec.counts_vals, 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        print('plots displayed. Please note indices of ROIs')
        plot_spectrum_raw(spec.channels, spec.counts_vals)
        #plot_spectrum_calibrated(spec.energies_kev, spec.counts_vals)
        numberofROIs = int(raw_input('how many ROIs?'))
        print (numberofROIs)
        if numberofROIs < 1:
            print('no ROIs. Exiting...')
            sys.exit()
        for i in range (0, (numberofROIs), 1):
            ####plot_spectrum_raw(spec.channels, spec.counts_vals)
            #ROI_low, ROI_high = selectROI()
        
        #plot_spectrum_zoom(xvalue, yvaluegradient)
        #yvalue2gradient = np.gradient(yvaluegradient)
        #plot_spectrum_zoom(xvalue, yvalue2gradient)

        #filters.gaussian_filter(spec.counts_vals, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        
        #dx = float(len(self.channels))/self.num_channels
        #gx = np.arange(-3*sigma, 3*sigma, dx)
        #gaussian = np.exp(-(x/sigma)**2/2)
        #result = np.convolve(original_curve, gaussian, mode="full")
        
        #print("HERE")
        #smoothed = np.convolve(spec.counts_vals, convolvingfunction)
        #plt.plot(xvalue, convolvingfunction)
        #plt.plot(xvalue, smoothed)
        #plt.show()

            ROI_low, ROI_high = selectROI()
            centroidguess, amplitudeguess, ROIrangemin, ROIrangemax = findpeaks(spec.energies_kev, spec.counts_vals, ROI_low, ROI_high)

            plot_spectrum_zoom(spec.channels, spec.counts_vals, ROIrangemin, ROIrangemax)
            plot_spectrum_zoom(spec.channels, yvalue, ROIrangemin, ROIrangemax)
            plot_spectrum_raw(spec.energies_kev[ROI_low:ROI_high], spec.counts_vals[ROI_low:ROI_high])
#plot_spectrum_zoom(xvalue, yvalue, ROIrangemin, ROIrangemax)
        
            print("centroidguess: ", centroidguess, "amplitudeguess: ", amplitudeguess, "ROIrangemin: ",  ROIrangemin, "ROIrangemax: ",  ROIrangemax) #TODO fix changing params!
            userinputparameters= raw_input('edit these parameters? y/n')
            if userinputparameters == 'y':
                centroidguess = float(raw_input('centroidguess'))
                amplitudeguess = float(raw_input('amplitudeguess'))
                ROIrangemin = float(raw_input('ROIrangemin'))
                ROIrangemax = float(raw_input('ROIrangemax'))
                print("new region plotted")
                plot_spectrum_raw(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax])
                print("continuing")
            else:
                print("continuing")
#print("which fitting function?") TODO

#  peakfwhm, peakfwhmerr, peakcenter, peaksigma, peakheight, peakamplitude = FitGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, (1))
#peakfwhm, peakfwhmerr, peakcenter, peaksigma, peakheight, peakamplitude =
            FitGaussianLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)
            spectrumrisetime = float(spectrumrisetime)
            spectrumgaptime =float(spectrumgaptime)
            peakfwhm = float(peakfwhm)
            peakfwhmerr = float(peakfwhmerr)

#plotfwhmrisetime(risetimes,fwhmvalues, fwhmerrs)

            outputfile = open('../' + str(outputfilename), 'a')
            outputfile.write(filefromlist + ', ')
            outputfile.write(str(spectrumlivetime))
            outputfile.write(', ')
            outputfile.write(str(spectrumrisetime))
            outputfile.write(', ')
        #ouputfile.write(str(spectrumgaptime))
        #otputfile.write(', ')
            outputfile.write(str(peakcenter))
            outputfile.write(', ')
            outputfile.write(str(peaksigma))
            outputfile.write(', ')
            outputfile.write(str(peakheight))
            outputfile.write(', ')
            outputfile.write(str(peakamplitude))
            outputfile.write(', ')
            outputfile.write(str(peakfwhm))
            outputfile.write(', ')
            outputfile.write(str(peakfwhmerr))
            outputfile.write('\n')
            outputfile.close()
            print("-----PEAK FINISHED-----")
        print("-----SPECTRUM FINISHED-----")
    print("-----ANALYSIS FINISHED-----")


#open file and copy info + values there: filename, risetime, FWHM, hardware values?
#write a plotter for that data that takes from that file.

if __name__ == "__main__":
    main()