import sys
import os.path
print (sys.path)

#sys.path.append('/Users/Asia/becquerel-master/')
#sys.path.append('/Users/Asia/becquerel-master/becquerel')
#sys.path.append('/Users/Asia/becquerel-master/becquerel/core')

#sys.path.remove('/Users/Asia/becquerel-master/')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel')
#sys.path.remove('/Users/Asia/becquerel-master/becquerel/core')

import matplotlib.pyplot as plt
from becquerel import Spectrum
###from becquerel import EnergyCalBase
import numpy as np
import lmfit as lmfit

def selectROI():
    lowindex = raw_input("ROI low index (channel #)")
    #lowindex = 8000
    if lowindex == '':
        lowindex = 0
    highindex = raw_input("ROI high index (channel #)")
    #highindex = 11000
    if highindex == '':
        highindex = spec.channels(max)
    lowindex = int(lowindex)
    highindex = int(highindex)
    return lowindex, highindex

def findpeaks(data_x, data_y, low_index, high_index):
    cut_data_y = data_y[low_index:high_index]
    for i in range (low_index,high_index):
        if data_y[i] == max(cut_data_y):
            centerindex =  data_x[i]
            minindex=low_index
            maxindex=high_index
            amplitude = max(cut_data_y)
    ROIrange = [minindex, maxindex]
    return centerindex, amplitude, minindex, maxindex

def linearbackgroundsubtractedcounts(x_val, x_energies, y_val, ROI_low, ROI_high):
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

def fitgaussian(data):
    """Returns (height, x, width_x) the gaussian parameters of a 1D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -data)
    p, success = optimize.leastsq(errorfunction, params)
    print (p)
    return p

def moments(data):
    total = data.sum()
    X = spec.energies_kev[9690: 10000]
    x = (X*data).sum()/total
    Width_x = np.sqrt(np.abs(np.sum((X-x)**2*data)/np.sum(data)))
    height = data.max()
    return height, x, Width_x

def gaussianfunction(height, center_x, width_x):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    gaussianfunc = lambda x: height*np.exp(-(((center_x-x)**2/width_x)))
    return gaussianfunc

def calculatefwhm(x, y, peak_centroid):
    peak_height = max(y)
    peak_height = float(peak_height)
    peak_centroid = int(peak_centroid)
    lower_x = x[0:peak_centroid]
    lower_x = lower_x[::-1]
    higher_x = x[peak_centroid:-1]
    lower_x_hm = 0.0
    high_x_hm = 0.0
    fwhm = 0.0
    for i in y:
        if y[i]<= peak_height/2:
            # lower_x_hm = x[i]
            #print(lower_x_hm)
            print(y[i])
            break
    for i in y:
        if y[i]<= peak_height/2:
            higher_x_hm = x[i]
            print(higher_x_hm)
            print(y[i])
            break
    fwhm = higher_x_hm - lower_x_hm
    print(fwhm)
    return fwhm

def FitGaussian(x,y):
    plt.figure(figsize=(20,10))

    mod = lmfit.models.GaussianModel()

    pars = mod.guess(y, x=x)
    out  = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.show()

def FitGaussianPeakLinearBackground(x,y,peak_amplitude, peak_centroid,peak_sigma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1  = lmfit.models.GaussianModel(prefix='g1_')
    pars.update( gauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-200), max=(peak_centroid+200))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    #gauss2  = GaussianModel(prefix='g2_')
    #pars.update(gauss2.make_params())
    #pars['g2_center'].set(155, min=125, max=175)
    #pars['g2_sigma'].set(15, min=3)
    #pars['g2_amplitude'].set(2000, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    #print("fwhm:")
    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_fwhm_err = (out.params['g1_fwhm'].stderr )
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    fit_height = (out.params['g1_height'].value)
    fit_amplitude = (out.params['g1_amplitude'].value)
    #print(fit_fwhm)
    #print(fit_fwhm_err)
    # print(out.params)

    # result1.params   :  the final Parameters
    # result1.params['amp']  : the 'amp' Parameter
    # result1.params['amp'].value   : best fit value for 'amp'
    # result1.params['amp'].stderr  : the uncertainty for 'amp'
    # result1.params['amp'].correl  : dict of correlation values with other the other parameters

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    #plt.plot(x, comps['g2_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()

    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'k--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

    return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitLorentzianPeakLinearBackground(x,y,peak_amplitude, peak_centroid, peak_sigma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1  = lmfit.models.LorentzianModel(prefix='g1_')
    pars.update( gauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-200), max=(peak_centroid+200))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_fwhm_err = (out.params['g1_fwhm'].stderr )
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    fit_height = (out.params['g1_height'].value)
    fit_amplitude = (out.params['g1_amplitude'].value)
    print(out.params)

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

    plt.show()

    return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitVoigtPeakLinearBackground(x,y,peak_amplitude, peak_centroid, peak_sigma, peak_gamma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    peak_gamma = int(peak_gamma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1  = lmfit.models.VoigtModel(prefix='g1_')
    pars.update( gauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-200), max=(peak_centroid+200))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)
    pars['g1_gamma'].set(peak_gamma)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_fwhm_err = (out.params['g1_fwhm'].stderr )
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    fit_height = (out.params['g1_height'].value)
    fit_amplitude = (out.params['g1_amplitude'].value)
    print(out.params)

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()

    plt.show()

    return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitPseudoVoigtPeakLinearBackground(x,y,peak_amplitude, peak_centroid, peak_sigma, peak_beta):

    gaussian_amplitude = int(peak_amplitude)
    gaussian_centroid = int(peak_centroid)
    gaussian_sigma = int(peak_sigma)
    gaussian_beta = int(peak_beta)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1  = lmfit.models.PseudoVoigtModel(prefix='g1_')
    pars.update( gauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-200), max=(peak_centroid+200))
    pars['g1_sigma'].set(peak_sigma, min=0.1)
    pars['g1_amplitude'].set(peak_amplitude, min=10)

    mod = gauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    print("fwhm:")
    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_fwhm_err = (out.params['g1_fwhm'].stderr )
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    #fit_height = (out.params['g1_height'].value)
    fit_amplitude = (out.params['g1_amplitude'].value)
    print(out.params)

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    #plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()

    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'g--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

#return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitSkewedGaussianPeakLinearBackground(x,y,peak_amplitude, peak_centroid, peak_sigma, peak_gamma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    peak_gamma = int(peak_gamma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    skewedgauss1  = lmfit.models.SkewedGaussianModel(prefix='g1_')
    pars.update(skewedgauss1.make_params())

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-200), max=(peak_centroid+200))
    pars['g1_sigma'].set(peak_sigma, min=0)
    pars['g1_amplitude'].set(peak_amplitude, min=1)
    pars['g1_gamma'].set(peak_gamma)

    mod = skewedgauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    print(out.params)
    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    #plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, comps['g1_'], 'b--')
    #plt.plot(x, comps['g2_'], 'b--')
    plt.plot(x, comps['lin_'], 'g--')

#plt.plot(x, out.best_fit, 'ro')
#    plt.show()
    calculatefwhm(x, out.best_fit, (out.params['g1_center'].value))

    print("HEREEEEE!")
    print("FWHM")
    print(fwhm)
    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'k--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

#return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitExponentialGaussianPeakLinearBackground(x,y,peak_amplitude, peak_centroid, peak_sigma, peak_gamma):

    peak_amplitude = int(peak_amplitude)
    peak_centroid = int(peak_centroid)
    peak_sigma = int(peak_sigma)
    peak_gamma = int(peak_gamma)

    #bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    #pars = bkg_mod.guess(y, x=x)

    skewedgauss1  = lmfit.models.ExponentialGaussianModel(prefix='g1_')
    #pars.update( skewedgauss1.make_params())
    mod = skewedgauss1
    pars = mod.guess(y, x=x)

    pars['g1_center'].set((peak_centroid), min=(peak_centroid-20), max=(peak_centroid+20))
    pars['g1_sigma'].set(peak_sigma, min=0)
    pars['g1_amplitude'].set(peak_amplitude, min=1)
    pars['g1_gamma'].set(peak_gamma)

    mod = skewedgauss1 #+ bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    print(out.params)

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    #plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, comps['g1_'], 'b--')
    #plt.plot(x, comps['g2_'], 'b--')
    #plt.plot(x, comps['lin_'], 'k--')

    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()

    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'k--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

#return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude

def FitGaussianSkewedGaussianLinearBackground(x,y,gaussian_amplitude, gaussian_centroid, gaussian_sigma, skewed_gaussian_amplitude, skewed_gaussian_centroid, skewed_gaussian_sigma, skewed_gaussian_gamma):

    gaussian_amplitude = int(gaussian_amplitude)
    gaussian_centroid = int(gaussian_centroid)
    gaussian_sigma = int(gaussian_sigma)
    skewed_gaussian_amplitude = int(gaussian_amplitude)
    skewed_gaussian_centroid = int(gaussian_centroid)
    skewed_gaussian_sigma = int(gaussian_sigma)
    skewed_gaussian_gamma = int(skewed_gaussian_gamma)

    bkg_mod = lmfit.models.LinearModel(prefix='lin_')
    pars = bkg_mod.guess(y, x=x)

    gauss1  = lmfit.models.GaussianModel(prefix='g1_')
    pars.update( gauss1.make_params())

    pars['g1_center'].set((gaussian_centroid), min=(gaussian_centroid-200), max=(gaussian_centroid+200))
    pars['g1_sigma'].set(gaussian_sigma, min=0.1)
    pars['g1_amplitude'].set(gaussian_amplitude, min=10)

    skewedgauss1  = lmfit.models.SkewedGaussianModel(prefix='sg1_')
    pars.update( skewedgauss1.make_params())

    pars['sg1_center'].set((skewed_gaussian_centroid), min=(skewed_gaussian_centroid-200), max=(skewed_gaussian_centroid+200))
    pars['sg1_sigma'].set(skewed_gaussian_sigma, min=0)
    pars['sg1_amplitude'].set(skewed_gaussian_amplitude, min=1)
    pars['sg1_gamma'].set(skewed_gaussian_gamma)


    mod = gauss1 + skewedgauss1 + bkg_mod
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    comps = out.eval_components(x=x)

    fit_fwhm = (out.params['g1_fwhm'].value )
    fit_fwhm_err = (out.params['g1_fwhm'].stderr )
    fit_center = (out.params['g1_center'].value)
    fit_sigma = (out.params['g1_sigma'].value)
    fit_height = (out.params['g1_height'].value)
    fit_amplitude = (out.params['g1_amplitude'].value)
    print(out.params)

    print(out.fit_report(min_correl=0.5))

    plt.figure(figsize=(10,5))

    plt.axis([(out.params['g1_center'].value - 2 * out.params['g1_fwhm'].value), (out.params['g1_center'].value + 2* out.params['g1_fwhm'].value), 0, (1.2* out.params['g1_height'].value)])
    ax = plt.gca()
    ax.set_autoscale_on(False)

    plt.plot(x, y, 'bo')
    plt.plot(x, out.best_fit, 'r-')
    plt.plot(x, out.init_fit, 'k--')
    plt.plot(x, comps['g1_'], 'b--')
    plt.plot(x, comps['sg1_'], 'g--')
    #plt.plot(x, comps['g2_'], 'b--')
    plt.plot(x, comps['lin_'], 'k--')

    myfile = open('testfile.txt','w')
    myfile.write(out.fit_report())
    myfile.close()

    #   plt.plot(x, y, 'bo')
    #   plt.plot(x, out.init_fit, 'k--')
    #   plt.plot(x, out.best_fit, 'r-')
    plt.show()

#return fit_fwhm, fit_fwhm_err, fit_center, fit_sigma, fit_height, fit_amplitude
def main():
    print ("SpectrumAnalysisFunctions")

if __name__ == "__main__":
    main()
