from random import sample
import matplotlib as mpl
#mpl.use('Agg')
from itertools import combinations_with_replacement
from itertools import product
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm
import os
import pandas
from numpy import pi, log, sqrt
import pix_by_pix_mk24_final2 as likelihood

import scipy.optimize as so
import math
import scipy

import matplotlib.cm as cm
from scipy.optimize import minimize

try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    from pypolychord.priors import UniformPrior
    from pypolychord.priors import GaussianPrior
except:
    pass
try:
    from anesthetic import NestedSamples
except ImportError:
    pass
try:
    from anesthetic.weighted_pandas import WeightedDataFrame
except ImportError:
    pass
from scipy.optimize import minimize
#import gen_beams_and_T_vs_LST as gen_beams_and_T_vs_LST_v2
from line_profiler import LineProfiler
import matplotlib
from astropy.modeling.powerlaws import LogParabola1D
from csv import writer

calc_mean_comps = False
calc_skys = False#True

#test_freqs = np.array([45.0,50,60,70,74])
test_freqs_indexes = np.array([False,False,False,False,False,False,False,False,True,True])#np.array([True,True,True,False,False,False,False,False,False])#np.array([False,False,False,False,False,False,False,True,False])

#declare a random seed
np.random.seed(0)
use_perturbed_dataset = True #do we want the input dataset to have calibration errors

#===================================================================#
#| SET PARAMS FOR THE SIMULATED DATA AND NESTED SAMPLING
Max_Nside=32 #the Nside at which to generate the set of maps
Max_m = hp.nside2npix(Max_Nside)
no_of_comps = 2
fit_curved=[True,True]


no_to_fit_curve = np.sum(fit_curved)
rezero_prior_std=2000

#params for the prior on the spectra
spec_min, spec_max = -3.5,1 #the range for the prior on the spectral indexes
curvature_mean, curvature_std = 0,2 #the range for the prior on the spectral index curvature parameter

#params for fitting the reference frequency 

fixed_f0 = 120 #if you dont fit a seperate reference freq for each comp then we fix f0 to this value


#params for the prior on the true maps
map_prior_variance_spec_index = -2.6
map_prior_variance_f0 = 408#fixed_f0
map_prior_std=400

calibrate = True
calibrate_all_but_45_150 = False#True #calibrate all maps in the dataset but the 45 and 150 MHz maps
calibrate_all = True#False #calibrate every map in the dataset


use_equal_spaced_LSTs = True
fit_haslam_noise = False
subtract_CMB = -2.726
print_vals_as_calc = False



reject_criterion = 1e-3#None #how close can two spectral idexes be in value before being rejected
cond_no_threshold =1e+9




test_LSTs = np.linspace(0,24,73)[:-1]#np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])#np.array([0,2,4,6,8,10,12,14,16,18,20,22]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
#test_freqs = [47.5,75]#,250,300,350]#[45,50.005,59.985,70.007,73.931,79.960,100,125,150,159,200,408]
#test_freqs = [100,140,200]
#test_freqs = [250,300,350]

#np.array([45.0,50.0,60.0,70,74,80,150,159,408])
#test_freqs = [70.0,74.0,80.0]
#test_freqs = [150.0,159.0,408.0]



unobs_marker = -32768

nlive = 500#2500*no_of_comps

precision_criterion = 1e-3


#specify what to fit for each map
if calibrate == True:
    if calibrate_all == True:
        freqs_to_calibrate = np.array([True,True,True,True,True,True,True,True,True,True]) #calibrate all the maps
        freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])
    if calibrate_all_but_45_150 == True:
        freqs_to_calibrate = np.array([True,False,True,True,True,True,True,False,True,False]) #calibrate all the maps except 45 and 150
        freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])

else:
    freqs_to_calibrate = np.array([False,False,False,False,False,False,False,False,False,False])#np.array([True,True,True,True,True,True,False,False])
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])


no_to_cal = np.sum(freqs_to_calibrate)
no_to_fit_noise = np.sum(freqs_to_fit_noise)


main_label = "_"+str(no_of_comps)+"_comp_cal:"+str(calibrate)+"_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres="+str(np.round(np.log10(cond_no_threshold),1))+"_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if use_equal_spaced_LSTs==True:
    if calibrate_all==True:
        root="inc_LWA1_45_mk24_no_curve:"+str(no_to_fit_curve)+"_cal_all_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
    else:
        if calibrate_all_but_45_150==True:
            root="fin_mk24_no_curve:"+str(no_to_fit_curve)+"_no_cal_45_150_408_f0="+str(fixed_f0)+main_label
else:
    
    
    root="mk24_extra_uneq_LSTs:"+str(len(LSTs_for_comparison))+"_fixed_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label

p=os.getcwd()+"/"
path = p+root+"/"

#load the marginal samples and comp map samples

marg_samps = np.loadtxt(path+"post_samples_marginal.csv",delimiter=",")
weights = np.loadtxt(path+"post_samples_marginal_weights.csv",delimiter=",")

no_of_samps = marg_samps.shape[0]

freqs_real = np.array([44.933,45,50.005,59.985,70.007,73.931,79.96,150,159,408])
nv = len(freqs_real)

calibration_params = marg_samps[:,3*no_of_comps+no_to_fit_noise:]
scale_params = np.ones((no_of_samps,nv))
zero_params = np.zeros((no_of_samps,nv))

scale_params[:,freqs_to_calibrate] = calibration_params[:,no_to_cal:]
zero_params[:,freqs_to_calibrate] = calibration_params[:,:no_to_cal]


print ("scale params")
print (scale_params)
print ("zero params")
print (zero_params)
freqs2 = np.array([44.933,45,50,60,70,74,80,150,159,408])
coords = ["G","C","C","C","C","C","G","G","G"]
print (test_freqs_indexes)
test_freqs = freqs2[test_freqs_indexes]
test_freqs_real = freqs_real[test_freqs_indexes]
scale_params_for_test = scale_params[:,test_freqs_indexes]
zero_params_for_test = zero_params[:,test_freqs_indexes]

print ("are testing the maps at v=",test_freqs)
import gzip

for i in range(1,12):
    name = path+"post_samples_chunk_"+str(int(i))+"_comp_maps.npy.gz"

    try:
        print ("opening chunk:",i)
        f = gzip.GzipFile(name,"r")

        chunk = np.load(f)
        f.close()

        if i ==1:
            maps = chunk
        else:
            maps = np.append(maps,chunk,axis=0)
    except:
        print ("no file for this chunk")
    print (maps.shape)

no_of_pixels = int(maps.shape[1]/no_of_comps)


if calc_mean_comps==True:
    samples_of_maps_DF = WeightedDataFrame(maps,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()
    #print (posterior_mean.shape)

    recovered_map = np.empty((no_of_pixels,no_of_comps))
    err_map = np.empty((no_of_pixels,no_of_comps))
    bool_arr_template = np.zeros(no_of_comps,dtype=bool)
    #comps_arr = np.empty((no_of_comps,no_of_comps,no_of_pixels))
    for c in range(no_of_comps):
        bool_arr_temp = np.copy(bool_arr_template)
        bool_arr_temp[c] = 1

        bool_arr = np.tile(bool_arr_temp,no_of_pixels)

        recovered_map[:,c]=posterior_mean[bool_arr]
        err_map[:,c]=posterior_std[bool_arr]#[c*no_of_pixels:(c+1)*no_of_pixels]
    #    print (recovered_map)

#    comps_arr[:,c,:] = maps[:,bool_arr]

    print (recovered_map)
    #save the mean componetent maps and their errors
    for i in range(no_of_comps):
        rec_comp, rec_comp_errs = recovered_map[:,i], err_map[:,i]

        np.savetxt(path+'posterior_mean_for_comp_'+str(i+1),rec_comp,delimiter=",")
        np.savetxt(path+'posterior_std_for_comp_'+str(i+1),rec_comp_errs,delimiter=",")
    #plot the component maps
    fig = plt.figure(figsize=(15,10))

    for i in range(no_of_comps):
        ax = plt.subplot2grid((2,no_of_comps),(0,i))

        plt.axes(ax)
        hp.mollview(recovered_map[:,i],title="recovered mean comp "+str(i+1),hold=True,notext=True)#,min=-10,max=10)#np.max(c_map_1))

    for i in range(no_of_comps):
        ax = plt.subplot2grid((2,no_of_comps),(1,i))

        plt.axes(ax)
        hp.mollview(err_map[:,i],title="errs mean comp "+str(i+1),hold=True)#,notext=True,min=-10,max=10)#np.max(c_map_1))

    plt.savefig(path+"recovered_comp_comparison.png")

#gen the skys
bool_arr_template = np.zeros(no_of_comps,dtype=bool)
comps_arr = np.empty((no_of_samps,no_of_comps,no_of_pixels))
for c in range(no_of_comps):
    bool_arr_temp = np.copy(bool_arr_template)
    bool_arr_temp[c] = 1

    bool_arr = np.tile(bool_arr_temp,no_of_pixels)
    comps_arr[:,c,:] = maps[:,bool_arr]

def gen_samp_sky(index):
     
    spec_params = marg_samps[index,:3*no_of_comps]

    comp_maps_samp = comps_arr[index]

    specs = np.empty((len(test_freqs),no_of_comps))

    
    for i in range(no_of_comps):
        sp = spec_params[3*i:3*(i+1)]
    #    print (sp)
        spec = LogParabola1D(1,sp[0],-sp[1],-sp[2])(test_freqs)
    #    print (spec)
        specs[:,i]=spec
    specs = np.array(specs)
    #print (specs)
    #compute the sky prediction for these freqs (this is the actual predicted sky)
    s1 = specs @ comp_maps_samp

    #"un calibrate" the prediction to make it comparable to the
    a_inv = np.diag(1/scale_params_for_test[index])
    #print (a_inv)
    s1_uncal = a_inv @ (s1 - np.reshape(zero_params_for_test[index],(len(test_freqs),1)))
    #print (s1_uncal)
    #print (s1_uncal.flatten())
    return s1.flatten(),s1_uncal.flatten()

def comp_mean_sky(sky_samps_array):
    samples_of_maps_DF = WeightedDataFrame(sky_samps_array,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()

    return posterior_mean, posterior_std
cal_pars = [r"$a_{44.9}=$ 1.165  "+r"$b_{44.9}=$ -286 K",r"$a_{45}=$ 1.034  "+r"$b_{45}=$ 163 K",r"$a_{50}=$ 1.139  " +r"$b_{50}=$ -465 K",r"$a_{60}=$ 1.142  "+r"$b_{60}=$ -651 K",r"$a_{70}=$ 1.256  "+r"$b_{70}=$ -673 K",r"$a_{74}=$ 1.229  "+r"$b_{74}=$ -639 K",r"$a_{80}=$ 1.250  "+r"$b_{80}=$ -600 K",r"$a_{150}=$ 0.998  "+r"$b_{150}=$ 16.6 K",r"$a_{159}=$ 1.285  "+r"$b_{159}=$ -66.3 K",r"$a_{408}=$ 1.029  "+r"$b_{408}=$ 0.91 K"]
cal_pars_for_plt = [cal_pars[i] for i in np.nonzero(test_freqs_indexes)[0]]

if calc_skys == True:
    print ("generating sky samples")
    sky_samps = []
    uncal_sky_samps = []

    for i in range(no_of_samps):
        scal, suncal = gen_samp_sky(i)
        sky_samps.append(scal)
        uncal_sky_samps.append(suncal)

    sky_samps = np.array(sky_samps)
    uncal_sky_samps = np.array(uncal_sky_samps)

    print ("computing mean sky (actual predictions)")
    skys, skys_err = comp_mean_sky(sky_samps)
    print ("computing mean sky (after un calibrating the sky)")
    skys_uncal, skys_uncal_err = comp_mean_sky(uncal_sky_samps)

    for i in range(len(test_freqs)):
        f=test_freqs[i]
        sky = skys[i*no_of_pixels:(i+1)*no_of_pixels]
        sky_err = skys_err[i*no_of_pixels:(i+1)*no_of_pixels]

        sky_uncal = skys_uncal[i*no_of_pixels:(i+1)*no_of_pixels]
        sky_uncal_err = skys_uncal_err[i*no_of_pixels:(i+1)*no_of_pixels]

        np.savetxt(path+"bayesian_pred_"+str(f)+"MHz",sky,delimiter=",")
        np.savetxt(path+"bayesian_errs_"+str(f)+"MHz",sky_err,delimiter=",")

        np.savetxt(path+"uncal_bayesian_pred_"+str(f)+"MHz",sky_uncal,delimiter=",")
        np.savetxt(path+"uncal_bayesian_errs_"+str(f)+"MHz",sky_uncal_err,delimiter=",")

        max1,min1 = np.round(np.nanmax(sky),0),np.round(np.nanmin(sky),0)
        max2,min2 = np.round(np.nanmax(sky_uncal),0),np.round(np.nanmin(sky_uncal),0)
        max3,min3 = np.round(np.nanmax(sky_err),0),np.round(np.nanmin(sky_err),0)
        max4,min4 = np.round(np.nanmax(sky_uncal_err),0),np.round(np.nanmin(sky_uncal_err),0)
        fig=plt.figure(figsize=(6,5))
        plt.subplot(2,2,2)
        hp.mollview(sky_uncal,title="calibration removed\n"+"posterior mean",hold=True,max=max2,min=min2,norm="log")
        
        plt.subplot(2,2,1)
        hp.mollview(sky,title="original posterior mean\n"+str(f)+"MHz",hold=True,max=max1,min=min1,norm="log")

        plt.subplot(2,2,4)
        hp.mollview(sky_uncal_err,title="calibration removed\n"+r"$1\sigma$"+" uncertainty",hold=True,max=max4,min=min4,norm="log")

        plt.subplot(2,2,3)
        hp.mollview(sky_err,title="original posterior\n"+r"$1\sigma$"+" uncertainty",hold=True,max=max3,min=min3,norm="log")

        plt.suptitle(cal_pars_for_plt[i])
        plt.rcParams.update({'font.size':15})
        plt.savefig(path+"pred_at_"+str(f)+".png")


def apply_calibrations(freq_index):
    nside_for_output = 64
    f_for_real = freqs_real[freq_index]
    f2 = freqs2[freq_index]

    print ("applying calibrations to the map at ",f_for_real,"MHz")

    true_sky = hp.read_map(p+"/dataset_trimmed/map_"+str(f_for_real)+"MHz_FWHM=5_nside=32.fits")
    true_sky_original = hp.read_map(p+"/data_at_original_resolution/"+str(f2)+"_map.fits")
    r=hp.Rotator(coord=[coords[freq_index],"G"])
    true_sky_original = r.rotate_map_pixel(true_sky_original)
    true_sky_original = hp.ud_grade(true_sky_original,nside_out=nside_for_output)


    true_sky[(true_sky<=0)]=float("NaN")
    true_sky_original[(true_sky_original<=0)] = float("NaN")

    try:
        true_errs = hp.read_map(p+"/dataset_trimmed/map_errs_"+str(f_for_real)+"MHz_FWHM=5_nside=32.fits")
        true_errs[np.isnan(true_sky)]=float("NaN")
        
    except:
        print ("no error map")
        true_errs = 0.1*true_sky
        #true_errs = np.ones(len(true_sky))
        #true_errs[:] = float("NaN")
    try:
        true_errs_original = hp.read_map(p+"/data_at_original_resolution/"+str(f2)+"_err_map.fits")
        true_errs_original = r.rotate_map_pixel(true_errs_original)
        true_errs_original[(true_errs_original<0)]=0
        true_errs_original = np.sqrt(hp.ud_grade(true_errs_original**2,nside_out=nside_for_output))
        true_errs_original[np.isnan(true_sky_original)]=float("NaN")
        #true_errs_original[(true_errs_original)]=float("NaN")
    except:
        print ("no error map")
        true_errs_original = 0.1*true_sky_original

    hp.mollview(true_errs_original)
    plt.savefig(path+"errs_orig.png")

    #loop over all samples and generate a posterior set of corrected skys
    sky_samps = []
    sky_samps_original = []

    print ("generating samples")
    for i in range(no_of_samps):
        #apply the calibration for this posterior sample
        corrected_sky_samp = (true_sky*scale_params[i,freq_index]) + zero_params[i,freq_index]
        corrected_original_sky_samp = (true_sky_original*scale_params[i,freq_index]) + zero_params[i,freq_index]
        
        #rescale the errs
        true_errs_samp = true_errs*scale_params[i,freq_index]
        true_errs_original_samp = true_errs_original*scale_params[i,freq_index]

        #gen random noise acording to these error maps
        rand_noise = np.random.normal(0,true_errs_samp)
        rand_noise_original = np.random.normal(0,true_errs_original_samp)

        corrected_sky_samp += rand_noise
        corrected_original_sky_samp += rand_noise_original

        sky_samps.append(corrected_sky_samp)
        sky_samps_original.append(corrected_original_sky_samp)

    sky_samps = np.array(sky_samps)
    sky_samps_original = np.array(sky_samps_original)

    #compute mean for the low res maps
    print ("computing mean for low res maps")
    samples_of_maps_DF = WeightedDataFrame(sky_samps,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()

    np.savetxt(path+"calibrated_sky_"+str(f_for_real)+"MHz",posterior_mean,delimiter=",")
    np.savetxt(path+"calibrated_sky_errs_"+str(f_for_real)+"MHz",posterior_std,delimiter=",")

    hp.mollview(posterior_mean,title=str(f_for_real))
    plt.savefig(path+"calbrated_"+str(f_for_real)+".png")

    hp.mollview(posterior_std,title=str(f_for_real))
    plt.savefig(path+"calbrated_"+str(f_for_real)+"_err.png")

    #compute mean for the high res maps
    print ("computing mean for high res maps")
    samples_of_maps_DF = WeightedDataFrame(sky_samps_original,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()

    np.savetxt(path+"calibrated_sky_original_res_"+str(f_for_real)+"MHz",posterior_mean,delimiter=",")
    np.savetxt(path+"calibrated_sky_errs_original_res_"+str(f_for_real)+"MHz",posterior_std,delimiter=",")

    hp.mollview(posterior_mean,title=str(f_for_real))
    plt.savefig(path+"calbrated_original_res_"+str(f_for_real)+".png")

    hp.mollview(posterior_std,title=str(f_for_real))
    plt.savefig(path+"calbrated_original_res_"+str(f_for_real)+"_err.png")
calibrate_diffuse_maps = False
if calibrate_diffuse_maps == True:

    for freq_index in range(5,9):#range(nv):
        apply_calibrations(freq_index)

#==========================================================================================================================================#
#generate sky predictions for 45,55,65,75,85,95 MHz
calc_skys_no_calibration = True
#test_freqs = np.array([120,150,160,200])
#test_freqs = np.array([45,50,60,70])
test_freqs = np.array([80,250,300,350])
#test_freqs = np.array([70,80,90])
#test_freqs = np.array([100,110,120])
#test_freqs = np.array([130,140])
#test_freqs = np.array([150,160,170])
#test_freqs = np.array([180,190,250])#np.array([45,65,85])
#test_freqs = np.array([300,350,400])
#test_freqs = np.array([150,160,170,180,190])
#test_freqs = np.array([200,210])
def gen_samp_sky2(index):
     
    spec_params = marg_samps[index,:3*no_of_comps]

    comp_maps_samp = comps_arr[index]

    specs = np.empty((len(test_freqs),no_of_comps))

    
    for i in range(no_of_comps):
        sp = spec_params[3*i:3*(i+1)]
    #    print (sp)
        spec = LogParabola1D(1,sp[0],-sp[1],-sp[2])(test_freqs)
    #    print (spec)
        specs[:,i]=spec
    specs = np.array(specs)
    #print (specs)
    #compute the sky prediction for these freqs (this is the actual predicted sky)
    s1 = specs @ comp_maps_samp

    return s1.flatten()



if calc_skys_no_calibration == True:
    print ("generating sky samples")
    sky_samps = []


    for i in range(no_of_samps):
        scal = gen_samp_sky2(i)
        sky_samps.append(scal)


    sky_samps = np.array(sky_samps)


    print ("computing mean sky (actual predictions)")
    skys, skys_err = comp_mean_sky(sky_samps)


    for i in range(len(test_freqs)):
        f=test_freqs[i]
        sky = skys[i*no_of_pixels:(i+1)*no_of_pixels]
        sky_err = skys_err[i*no_of_pixels:(i+1)*no_of_pixels]

        #sky_uncal = skys_uncal[i*no_of_pixels:(i+1)*no_of_pixels]
        #sky_uncal_err = skys_uncal_err[i*no_of_pixels:(i+1)*no_of_pixels]

        np.savetxt(path+"test_bayesian_pred_"+str(f)+"MHz",sky,delimiter=",")
        np.savetxt(path+"test_bayesian_errs_"+str(f)+"MHz",sky_err,delimiter=",")

        