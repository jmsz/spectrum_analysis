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
import glob
# from scipy.ndimage import filters
import os
import sys


def main():

    # initialize output file
    directory = '/home/anp/Desktop/uhrge_measurements/nanoMCAII/sept29/am241/strip8' # data file locaton
    outputfilename = "/home/anp/Desktop/uhrge_measurements/nanoMCAII/sept29/am241/redo_summary.csv" # = directory + ".csv"

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
        if filename.endswith("6.lzs"):
            filelist.append(os.path.join(directory, filename))
            continue
        #if filename.endswith("10.lzs"):
        #    filelist.append(os.path.join(directory, filename))
        #    continue
        #if filename.endswith("8.lzs"):
        #    filelist.append(os.path.join(directory, filename))
        #    continue
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
        print(filefromlist)
        #skipflag = input('analyze(y) or skip(n)? enter y/n')
        #if skipflag == 'y':
        #    print("proceeding...")
        #elif skipflag == 'n':
    #        print("skipping...")
#            pass
#        else:
#            print("proceeding...")
        spec = Spectrum.from_file(filefromlist)
        spectrumlivetime = spec.livetime
        spectrumgaptime = spec.gaptime
        spectrumrisetime = spec.risetime
        print(spectrumgaptime)
        print(spectrumrisetime)
        print("***")
        # xvalue = spec.channels
        # print(xvalue)
        # convolvingfunction = gaussianfunction(700, xvalue, 2)
        # yvalue = convolvingfunction(xvalue)
        # print(yvalue)
        # yvalue = filters.gaussian_filter(spec.counts_vals, 2, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
        print('plots displayed. Please note indices of ROIs')
        plot_spectrum_raw(spec.channels, spec.counts_vals)
        numberofROIs = 2
        print(numberofROIs)
        if numberofROIs < 1:
            print('no ROIs. Exiting...')
            sys.exit()
        for i in range(0, (numberofROIs), 1):
        #    if i == 0:
        #        ROI_low = 7000
        #        ROI_high = 12000
        #    elif i == 1:
        #        ROI_low = 14000
        #        ROI_high = 16282
            #else:
        #        print ("error on number of ROIs...")

            ROI_low, ROI_high = selectROI()
            #if i ==0:strip6
            #    ROI_low = 100
            #    ROI_high = 8000
            #elif i ==1:
            #    ROI_low = 8000
            #    ROI_high = 16000

        #    ROIrangemin = ROI_low - 1000
        #    ROIrangemax = ROI_high + 1000

            centroidguess, amplitudeguess, ROIrangemin, ROIrangemax = findpeaks(spec.energies_kev, spec.counts_vals, ROI_low, ROI_high)
            #plot_spectrum_zoom(spec.channels, spec.counts_vals, ROIrangemin, ROIrangemax)

            print("centroidguess: ", centroidguess, "amplitudeguess: ", amplitudeguess, "ROIrangemin: ",  ROIrangemin, "ROIrangemax: ",  ROIrangemax) #TODO fix changing params!
            #userinputparameters = input('edit these parameters? y/n')
            userinputparameters = 'n'
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
# print("which fitting function?") TODO
# peakfwhm, peakfwhmerr, peakcenter, peaksigma, peakheight, peakamplitude =
# FitGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)
# FitLorentzianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1)
# FitVoigtPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, 0)
# FitPseudoVoigtPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, .5)
# FitExponentialGaussianPeakLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, 676, 1, -3)
#  FitGaussianSkewedGaussianLinearBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, amplitudeguess, centroidguess, 1, 0)
#  fittypes = [FitGaussianPeakLinearBackground, FitSkewedGaussianPeakLinearBackground]

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

            peakfwhm, peakgamma, peakcenter, peaksigma, peakamplitude, chisqr, redchi = FitSkewedGaussianPeakStepBackground(spec.energies_kev[ROIrangemin:ROIrangemax], spec.counts_vals[ROIrangemin:ROIrangemax], amplitudeguess, centroidguess, 1, -3)
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
            outputfile.close()
            print("-----PEAK FINISHED-----")
        print("-----SPECTRUM FINISHED-----")
    print("-----ANALYSIS FINISHED-----")

    # plotfwhmrisetime(risetimes,fwhmvalues, fwhmerrs)


if __name__ == "__main__":
    main()
