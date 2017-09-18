#!/usr/bin/python

# ----------------------------------------------------------------------------------------
# Title: Spectrum Analysis Script
# Institution: Lawrence Berkeley National Laboratory
# Author: J. Szornel
# Version: 1.0
# Created: 08-15-17
# Last edited: 09-01-17
# Description: Analyzes spectra from labzy spectrum files (currently implementing multiple files,
#              streamlined process. Extracts fwhm values.
# ----------------------------------------------------------------------------------------

from __future__ import print_function
from SpectrumAnalysisFunctions import *
from PlottingFunctions import *
from scipy.ndimage import filters
import os
import sys

def main():

    # initialize output file
    directory = '/home/anp/Desktop/nanoMCA_strip6/risetimetest' #data file locaton
    filenumber = 0
    outputfilename = "/home/anp/Desktop/nanoMCA_strip6/summaryfiles/output" + str(filenumber) + '.csv'

    if os.path.exists(outputfilename):
        outputfile = open(outputfilename, "a+")
    else:
        outputfile = open(outputfilename, "w")
    outputfile.write('filename' + ', ')
    outputfile.write('livetime' + ', ')
    outputfile.write('risetime' + ', ')
    outputfile.write('gaptime' + ', ')
    outputfile.write('ROIrangemin' + ', ')
    outputfile.write('ROIrangemax' + ', ')
    outputfile.write('Chi Sqr' + ', ')
    outputfile.write('Reduced Chi Sqr' + ', ')
    outputfile.write('center' + ', ')
    outputfile.write('sigma' + ', ')
    outputfile.write('height' + ', ')
    outputfile.write('amplitude (# counts)' + ', ')
    outputfile.write('FWHM' + ', ')
    outputfile.write('FWHM error' + ', ')
    outputfile.write('gamma' + '\n ')
    outputfile.close()

    filelist = []

    for filename in os.listdir(directory):
        if filename.endswith("8_us.lzs"):
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
            print(i)
        input('continue')

    for filefromlist in filelist:

        spec = Spectrum.from_file(filefromlist)

        spectrumrisetime = input('risetime for spectrum in us (check filename):')
#        spectrumrisetime = .2
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
#        plot_spectrum_raw(spec.channels, spec.counts_vals)
        print('plots displayed. Please note indices of ROIs')

        plot_spectrum_raw(spec.channels, spec.counts_vals)
        #plot_spectrum_calibrated(spec.energies_kev, spec.counts_vals)
######ASIA
#        numberofROIs = int(input('how many ROIs?'))
        numberofROIs = 2
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
            #plot_spectrum_zoom(spec.channels, yvalue, ROIrangemin -100, ROIrangemax+100)
            #plot_spectrum_raw(spec.energies_kev[ROI_low:ROI_high], spec.counts_vals[ROI_low:ROI_high])
#plot_spectrum_zoom(xvalue, yvalue, ROIrangemin, ROIrangemax)

            print("centroidguess: ", centroidguess, "amplitudeguess: ", amplitudeguess, "ROIrangemin: ",  ROIrangemin, "ROIrangemax: ",  ROIrangemax) #TODO fix changing params!
            userinputparameters= input('edit these parameters? y/n')
            if userinputparameters == 'y':
                centroidguess = float(input('centroidguess'))
                amplitudeguess = float(input('amplitudeguess'))
                ROIrangemin = int(input('ROIrangemin'))
                ROIrangemax = int(input('ROIrangemax'))
                print("new region plotted")
                plot_spectrum_raw(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax])
                print("continuing")
            else:
                print("continuing")
#print("which fitting function?") TODO

#peakfwhm, peakfwhmerr, peakcenter, peaksigma, peakheight, peakamplitude =

#FitGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)

#FitLorentzianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)

# FitVoigtPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, 0)

# FitPseudoVoigtPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, .5)

# FitExponentialGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, 676, 1, -3)

#  FitGaussianSkewedGaussianLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, amplitudeguess, centroidguess, 1, 0)
            fittypes = [FitGaussianPeakLinearBackground, FitSkewedGaussianPeakLinearBackground]

            peakcenter = 0.0
            peaksigma = 0.0
            peakgamma = 0.0
            peakheight = 0.0
            peakamplitude = 0.0
            peakfwhm = 0.0
            peakfwhmerr = 0.0

            peakfwhm, peakfwhmerr, peakcenter, peaksigma, peakheight, peakamplitude, chisqr, redchi = FitGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, (1))

            spectrumrisetime = float(spectrumrisetime)
            spectrumgaptime = float(spectrumgaptime)
            peakfwhm = float(peakfwhm)
            peakfwhmerr = float(peakfwhmerr)

            outputfile = open(outputfilename, "a+")
            outputfile.write(filefromlist + ', ')
            outputfile.write(str(spectrumlivetime))
            outputfile.write(', ')
            outputfile.write(str(spectrumrisetime))
            outputfile.write(', ')
            outputfile.write(str(spectrumgaptime))
            outputfile.write(', ')
            outputfile.write(str(ROIrangemin))
            outputfile.write(', ')
            outputfile.write(str(ROIrangemax))
            outputfile.write(', ')
            outputfile.write(str(chisqr))
            outputfile.write(', ')
            outputfile.write(str(redchi))
            outputfile.write(', ')
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
            outputfile.write(', ')
            outputfile.write(str(peakgamma))
            outputfile.write('\n')

            peakcenter = 0.0
            peaksigma = 0.0
            peakgamma = 0.0
            peakheight = 0.0
            peakamplitude = 0.0
            peakfwhm = 0.0
            peakfwhmerr = 0.0
            chisqr = 0.0
            redchi = 0.0

            peakfwhm, peakgamma, peakcenter, peaksigma, peakamplitude, chisqr, redchi = FitSkewedGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, -3)

            spectrumrisetime = float(spectrumrisetime)
            spectrumgaptime =float(spectrumgaptime)
            peakfwhm = float(peakfwhm)
            peakfwhmerr = float(peakfwhmerr)

            outputfile = open(outputfilename, "a+")
            outputfile.write(filefromlist + ', ')
            outputfile.write(str(spectrumlivetime))
            outputfile.write(', ')
            outputfile.write(str(spectrumrisetime))
            outputfile.write(', ')
            outputfile.write(str(spectrumgaptime))
            outputfile.write(', ')
            outputfile.write(str(ROIrangemin))
            outputfile.write(', ')
            outputfile.write(str(ROIrangemax))
            outputfile.write(', ')
            outputfile.write(str(chisqr))
            outputfile.write(', ')
            outputfile.write(str(redchi))
            outputfile.write(', ')
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
            outputfile.write(', ')
            outputfile.write(str(peakgamma))
            outputfile.write('\n')
            outputfile.close()
            print("-----PEAK FINISHED-----")
        print("-----SPECTRUM FINISHED-----")
    print("-----ANALYSIS FINISHED-----")

    #plotfwhmrisetime(risetimes,fwhmvalues, fwhmerrs)

if __name__ == "__main__":
    main()
