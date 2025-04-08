from random import sample
import matplotlib as mpl
mpl.use('Agg')
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
#import gen_beams_and_T_vs_LST_for_plt as gen_beams_and_T_vs_LST_v2

#declare a random seed
np.random.seed(0)



#===================================================================#
#| SET PARAMS FOR THE SIMULATED DATA AND NESTED SAMPLING
Max_Nside=32 #the Nside at which to generate the set of maps
Max_m = hp.nside2npix(Max_Nside)
no_of_comps = 2#3
#which components will have curved powerlaws
fit_curved= [True,True]#[True,True,True]



no_to_fit_curve = np.sum(fit_curved)
rezero_prior_std=2000

#params for the prior on the spectra
spec_min, spec_max = -3.5,1 #the range for the prior on the spectral indexes
curvature_mean, curvature_std = 0,2 #the range for the prior on the spectral index curvature parameter

fixed_f0 = 180 #if you dont fit a seperate reference freq for each comp then we fix f0 to this value


#params for the prior on the true maps
map_prior_variance_spec_index = -2.7#-4
map_prior_variance_f0 = 408#fixed_f0
map_prior_std=400#1000

#misalanious params
calibrate = True
calibrate_all_but_45_150 = False #calibrate all maps in the dataset but the 45 and 150 MHz maps
calibrate_all = True #calibrate every map in the dataset

use_equal_spaced_LSTs = True
fit_haslam_noise = False
subtract_CMB = -2.726
print_vals_as_calc = False#True



reject_criterion = 1e-3#None #how close can two spectral idexes be in value before being rejected
cond_no_threshold =1e+9




unobs_marker = -32768



nlive = 500#2500*no_of_comps

precision_criterion = 1e-3


f0=150 #ref freq used for some fitting of spec indexes for plots (not used during any model fitting)


freqs = [44.933,45,50.005,59.985,70.007,73.931,79.96,150,159,408] #the frequencies in MHz of maps used to generate the model
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

#freqs_for_T_v_LST_comp = np.linspace(40,190,76,dtype="int")
#freqs_for_T_v_LST_comp = np.array([40,42.5,45,47.5,50,52.5,55,57.5,60,62.5,65,67.5,70,72.5,75,77.5,80,82.5,85,87.5,90,92.5,95,97.5,100,102.5,105,107.5,110,112.5,115,117.5,120,122.5,125,127.5,130,132.5,135,137.5,140,142.5,145,147.5,150,152.5,155,157.5,160,162.5,165,167.5,170,172.5,175,177.5,180,182.5,185,187.5,190,192.5,195,197.5,200])
freqs_for_T_v_LST_comp = np.array([40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,105.0,110.0,115.0,120.0,125.0,130.0,135.0,140.0,145.0,150.0,155.0,160.0,165.0,170.0,175.0,180.0,185.0,190.0,195.0,200.0])
n_spec_pars = 3*no_of_comps #the number of parameters for the spectra (for each comp we have: break_freq, spec_index1, spec_index2)


nv=len(freqs) #the number of freqs that have maps
no_of_fitted_noise = np.sum(freqs_to_fit_noise) #the number of freqs at which we fit noise level
no_of_calibrated = np.sum(freqs_to_calibrate) #the number of freqs at which we fit the calibration



#set the prior for the noise (on the Haslam map)
noise_prior_lower, noise_prior_upper = np.array([0.01]),np.array([50])

print ("noise prior is from:",noise_prior_lower,"to",noise_prior_upper,"Kelvin")




if np.sum(freqs_to_calibrate)!=0:
    #set the prior params for the zero levels
    zero_lev_prior_std = rezero_prior_std*np.ones(np.sum(freqs_to_calibrate))#200*((np.array(freqs)[freqs_to_calibrate]/100)**-2.5)
    zero_lev_prior_means = np.zeros(np.sum(freqs_to_calibrate))


    #set the prior params for the scale corrections
    scale_prior_lower=0.5
    scale_prior_upper=1.5
    

    print ("zero level prior is gauss with mean 0K, stds (Kelvin):")
    print (zero_lev_prior_std)
    print ("temp scale prior is uniform from:",scale_prior_lower,"to",scale_prior_upper)





#true_spec_indexes = [-2.13,-1.86,-1.46]
#CREATE A DIR TO STORE RESULTS
#====================================================================#
#set the file root

main_label = "_"+str(no_of_comps)+"_comp_cal:"+str(calibrate)+"_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres="+str(np.round(np.log10(cond_no_threshold),1))+"_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if use_equal_spaced_LSTs==True:
    #LSTs_for_comparison = np.array([2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])#np.array([0,2,4,6,8,10,12,14,16,18,20,22])#np.array([2.5,18]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
    LSTs_for_comparison = np.linspace(0,24,73)#[:-1]
    print (LSTs_for_comparison)
    print ("no of LSTs is:",len(LSTs_for_comparison))
    if calibrate_all==True:
        root="inc_LWA1_45_mk24_no_curve:"+str(no_to_fit_curve)+"_cal_all_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
    else:
        if calibrate_all_but_45_150==True:
            root="inc_LWA1_45_mk24_no_curve:"+str(no_to_fit_curve)+"_no_cal_45_150_408_f0="+str(fixed_f0)+main_label
else:
    #LSTs_for_comparison = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11.25,11.5,11.75,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,15.5,15.75,16,16.25,16.5,16.75,17,17.1,17.2,17.3,17.4,17.5,17.6,17.7,17.8,17.9,18,18.25,18.5,18.75,19,19.25,19.5,19.75,20,20.25,20.5,20.75,21,21.25,21.5,21.75,22,22.25,22.5,22.75,23,23.25,23.5,23.75])
    LSTs_for_comparison = np.array([0,2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])
    print (LSTs_for_comparison)
    print ("no of LSTs is:",len(LSTs_for_comparison))
    
    root="mk24_extra_uneq_LSTs:"+str(len(LSTs_for_comparison))+"_fixed_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
    
print (LSTs_for_comparison)
print ("no of LSTs is:",len(LSTs_for_comparison))
#make a dir to store the results
p=os.getcwd()+"/"
path = p+root+"/"
try:
    os.mkdir(root)
except:
    pass
#make a dir to store the results as we run
root2 = path+"running_results/"
try:
    os.mkdir(root2)
except:
    pass

#write a text file containing the key results
with open(path+"run_details.txt","w") as f:
    f.write("\n")
    f.write("a sumary file for the results of the nested sampling run.\n")
    f.write("=========================================================\n")
    f.write("reference freq for model, f0 = "+str(fixed_f0)+"\n")
    f.write("map prior spec index is:"+str(map_prior_variance_spec_index)+"\n")
    f.write("map prior std is: "+str(map_prior_std)+" Kelvin\n")
    f.write("reference freq for map prior, map_prior_f0 = "+str(map_prior_variance_f0))
    f.write("freqs with maps in input dataset \n")
    f.write(str(freqs)+"\n")
    f.write("calibration freqs\n")
    f.write(str(freqs_to_calibrate)+"\n")
    f.write("fit noise at freqs\n")
    f.write(str(freqs_to_fit_noise)+"\n")
    f.write("freqs for comparison of T vs LST: \n")
    f.write(str(freqs_for_T_v_LST_comp)+"\n")
    f.write("LSTs for comparison: \n")
    f.write(str(LSTs_for_comparison)+"\n")
    f.close()

#LOAD THE DATASET AND THE ERROR MAPS
#====================================================================#

obs_maps = []
inv_err_maps = []
data_err_maps = []
load_path = p+"dataset_trimmed/"#"dataset_Haslam_LWA1_uncal_Guz_LW_Monsalve_cal_smoothed_FWHM=5_at_Nside=32/"
for i in range(len(freqs)):
    f=freqs[i]

    print (f)
    fname1 = "map_"+str(f)+"MHz_FWHM=5_nside=32.fits"
    
    
    #fname2 = "noise_"+str(f)

    if freqs_to_fit_noise[i]==False:
        try:
            #m1, err_m = np.loadtxt(load_path+fname1), np.loadtxt(load_path+fname2)
            m1 = hp.read_map(load_path+fname1)
            try:
                err_m = hp.read_map(load_path+"map_errs_"+str(f)+"MHz_FWHM=5_nside=32.fits")
            except:
                print ("using aprox errors for:",f)
                err_m = 0.1*m1
        except:
            print ("cant find the files for freq:",f)

        
        #mask out any pixels with negative temps
        bool_arr = m1<=0
        err_m[bool_arr] = unobs_marker
        m1[bool_arr] = unobs_marker

        

        inv_err_m = 1/err_m
        inv_err_m[(err_m==unobs_marker)] = 0

        m1[m1!=unobs_marker] = m1[m1!=unobs_marker]+subtract_CMB
        obs_maps.append(m1)
    
        inv_err_maps.append(inv_err_m)

        data_err_maps.append(err_m)
    else:
        m1 = np.loadtxt(load_path+fname1)
        m1[m1!=unobs_marker] = m1[m1!=unobs_marker]+subtract_CMB
        obs_maps.append(m1)
    

obs_maps=np.array(obs_maps)
inv_err_maps=np.array(inv_err_maps)
print ("dataset loaded")

#CREATE THE INVERSE NOISE MATRICES
#====================================================================#
#generate the inverse noise covariance matrix for each pixel
inv_noise_mats = np.empty(shape=(Max_m,len(freqs),len(freqs)))
for p in range(0,Max_m):
    inv_stds_for_pixel = np.zeros(len(freqs))
    inv_stds_for_pixel[~freqs_to_fit_noise] = inv_err_maps[:,p]
    #print (inv_stds_for_pixel)

    Np_inv = np.diag(inv_stds_for_pixel**2)
    #print (Np_inv)
    inv_noise_mats[p,:,:] = Np_inv
print ("==============================================================")
print ("the mean inv_noise mat has diagonal elements of:")
print (np.diag(np.nanmean(inv_noise_mats,axis=0)))
print ("max and min for the inv noise mats: ",np.max(inv_noise_mats),np.min(inv_noise_mats[(inv_noise_mats!=0)]))

print ("fiting a power law to these matrix diagonal elements, gives:")
logs = np.log(np.diag(np.nanmean(inv_noise_mats,axis=0)))

log_freqs = np.log(np.array(freqs)/map_prior_variance_f0)
fun = lambda x: np.nansum((logs - x[0]*log_freqs -x[1])**2)
res = minimize(fun,[-2.5,1])
print (res)

print ("inverse noise matrices created")
#PLOT THE DATASET
#====================================================================#
fig = plt.figure(figsize=(12,16))
for i in range(len(freqs)):
    map_i = np.copy(obs_maps[i])
    ax = plt.subplot(5,3,int(i+1))
    map_i[(map_i==unobs_marker)]=float("NaN")
    plt.axes(ax)
    hp.mollview(map_i,title="Data Freq="+str(freqs[i]),hold=True,notext=True,norm="log")



plt.savefig(path+"/sky_maps_for_dataset")
#plt.show()
fig = plt.figure(figsize=(12,16))
for i in range(len(freqs)):
    map_i = np.copy(inv_err_maps[i])
    ax = plt.subplot(5,3,int(i+1))
    map_i[(map_i==0)]=float("NaN")
    plt.axes(ax)
    hp.mollview(1/map_i,title="input errs freq="+str(freqs[i]),hold=True,notext=True,norm="log")



plt.savefig(path+"/err_maps_for_dataset")
freqs=np.array(freqs)
#plot the priors and the data
#====================================================================
log_mean_temps = []
mean_temps = []
for i in range(len(freqs)):
    the_map = obs_maps[i]
    mean = np.mean(the_map[(the_map!=unobs_marker)])
    log_mean_temps.append(np.log(mean))
    mean_temps.append(mean)
log_mean_temps = np.array(log_mean_temps)
mean_temps = np.array(mean_temps)

log_freqs = np.log(freqs/f0)
fun = lambda x: np.nansum((log_mean_temps - x[0]*log_freqs -x[1])**2)
res = minimize(fun,[-2.15,np.log(16)])
print (res)

fig = plt.figure(figsize=(6,6))#figsize=(12,16))
#the fitted powerlaw
targ = np.exp(res.x[1])*((freqs/f0)**res.x[0])

ax1 = plt.subplot(1,1,1)
ax1.plot(freqs,targ,c="red",label="fitted power law")
ax1.scatter(freqs,mean_temps,label="data")
ax1.set_title("map mean temps")
ax1.legend(loc="upper right")
ax1.set_xscale("log")
ax1.set_yscale("log")
plt.savefig(path+"/input_map_mean_temps.png")


#generate a set of pre rotated EDGES beams at each of the frequencies that we want to compare the model to EDGES for
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print ("loading EDGES beams")
EDGES_beams = np.load("achromatic_EDGES_low_and_high_beam.npy")#np.empty(shape=(obs_maps.shape[1],len(freqs_for_T_v_LST_comp),len(LSTs_for_comparison)))


#Load the mock EDGES T vs LST plots (these are produced by convolving the mock sky map (before any pertubation) with the model beam at that freq)
#===============================================================================================
print ("loading the EDGES T vs LST traces")
EDGES_temps_at_calib_LSTs_and_freqs = np.load("EDGES_low_and_high_chromaticity_corrected_TvsLSTs.npy")
EDGES_errs = np.load("EDGES_low_and_high_chromaticity_corrected_TvsLST_errs.npy")


print ("EDGES data shape:",EDGES_temps_at_calib_LSTs_and_freqs.shape,EDGES_errs.shape)


#generate the EDGES noise covar mats for each freq assuming noise for each LST is independent of the other LSTs
print ("generating the EDGES inverse noise matrices")
EDGES_inv_noise_mats = []
for i in range(len(freqs_for_T_v_LST_comp)):
    inv_cov = np.diag(1/EDGES_errs[i,:]**2)
    EDGES_inv_noise_mats.append(inv_cov)
EDGES_inv_noise_mats = np.array(EDGES_inv_noise_mats)
#=====================================================================================================
#set up the likelihood function
bayes_eval = likelihood.bayes_mod(obs_maps=obs_maps,obs_freqs=freqs,inv_noise_mats=inv_noise_mats,EDGES_beams=EDGES_beams,EDGES_temps_at_calib_LSTs_and_freqs=EDGES_temps_at_calib_LSTs_and_freqs,EDGES_errs=EDGES_errs,EDGES_inv_noise_mats=EDGES_inv_noise_mats,freqs_for_T_v_LST_comp=freqs_for_T_v_LST_comp,LSTs_for_comparison=LSTs_for_comparison,no_of_comps=no_of_comps,save_root=root2,un_obs_marker=unobs_marker,map_prior_std=map_prior_std,map_prior_spec_index=map_prior_variance_spec_index,map_prior_f0=map_prior_variance_f0)



no_of_params_for_spec_mod = no_of_comps + no_to_fit_curve 
print ("we arn't fitting f0: f0=",fixed_f0," no of comps with curved spectra is",no_to_fit_curve," no of params for spectral model is",no_of_params_for_spec_mod)


#select the spec indexes 
if no_to_fit_curve==no_of_comps:
    print ("all comps are curved spec")
    spec_indexes_select = np.tile(np.array([False,True,False]),no_of_comps)
    spec_curvature_select =  np.tile(np.array([False,False,True]),no_of_comps)
    spec_f0_select =  np.tile(np.array([True,False,False]),no_of_comps)
    print ("spec_curvature_select =",spec_curvature_select)
else:
    print ("not all comps are curved spec")
    spec_indexes_select = np.tile(np.array([False,True,False]),no_of_comps)
    spec_f0_select =  np.tile(np.array([True,False,False]),no_of_comps)
    spec_curvature_select = np.array([np.array([False,False,True])*fit_curve_for_comp for fit_curve_for_comp in fit_curved]).flatten() #the indexes for spectral curvature in the final param array
    print ("spec_curvature_select =",spec_curvature_select)
    spec_not_curve_select = np.array([np.array([False,False,not_fit_curve_for_comp]) for not_fit_curve_for_comp in ~np.array(fit_curved)]).flatten() #the indexes with no curvature in the final param array
    print ("spec_not_curve select =",spec_not_curve_select)

def likelihood1(x):

    #print ("==================================================================")
    #print ("=======================LIKELIHOOD CALL============================")
    #print ("the pars from the prior is len:",len(x))
    
    everything_else = x[no_of_params_for_spec_mod:]
    spec_params = np.empty(3*no_of_comps)
    spec_params[spec_f0_select] = fixed_f0
    spec_params[spec_indexes_select] = x[:no_of_comps]
    spec_params[spec_curvature_select] = x[no_of_comps:no_of_params_for_spec_mod]
    if no_to_fit_curve!=no_of_comps:
        spec_params[spec_not_curve_select] = 0
    
    #print ("spec_pars:",spec_params)
    if no_of_fitted_noise!=0:
        noise_estimates = everything_else[:no_of_fitted_noise]
    else:
        noise_estimates = None
    #print ("noise_estimates:",noise_estimates)
    if no_of_calibrated!=0:
        zero_level_estimates = everything_else[no_of_fitted_noise:no_of_fitted_noise+no_of_calibrated]
        #print ("zero_level_estimates:",zero_level_estimates)
        scale_estimates = everything_else[no_of_fitted_noise+no_of_calibrated:]
        #print ("scale_estimates:",scale_estimates)
    else:
        zero_level_estimates = None#x[no_of_comps+nv:no_of_comps+2*nv]#[no_of_comps:no_of_comps+len(freqs)]
        scale_estimates = None#x[no_of_comps+2*nv:no_of_comps+3*nv]

    #print ("spec params:",spec_params)
    #print ("noise estimates:",noise_estimates)
    #print ("scale estimates:",scale_estimates)
    #print ("zero level estimates:",zero_level_estimates)
    
    log_l = bayes_eval.likelihood(spec_params=spec_params,noise_estimates=noise_estimates,freqs_to_fit_noise=freqs_to_fit_noise,scale_estimates=scale_estimates,zero_level_estimates=zero_level_estimates,freqs_to_calibrate=freqs_to_calibrate,joint_prior_func=None,reject_criterion=reject_criterion,print_vals=print_vals_as_calc,condition_no_threshold=cond_no_threshold)
    #print (log_l)
    return log_l[-1],[]#[term_to_plot] #return the log_posterior distribtution value for this set of parameters

#likelihood1(np.array([-2.5,-2.1,-1.8,20]))

#-------------NESTED SAMPLING PARAMS-------------

nDims =  int(no_of_params_for_spec_mod + no_of_fitted_noise + 2*no_of_calibrated)
nrepeat = 1*nDims #the nrepeat is set to 5 times the total number of pars that we fit
print ("no of dimensions for sampling region is:",nDims)
nDerived = 0 #we don't derive any parameters 
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = root
settings.nlive = nlive
settings.nrepeats = nrepeat
settings.do_clustering = True
settings.read_resume = True
settings.write_resume = True
settings.maximise = False #find the maximum of the poseterior
settings.precision_criterion = precision_criterion

prior_lower = spec_min #I have reduced the size of the prior and have centered it on the values that have previously given the best results
prior_upper = spec_max
#------------------------------------------------

#Define a box uniform prior over the specified range of values
 

print ("the indexes with the spec index are:",spec_indexes_select)
def prior(hypercube):

    #print ("prior called")
    #print (hypercube)
    
    #=====================================================#
    #define a uniform prior from spec_min to spec_max for the spectral indexes
    
    #generate the spectral params
    ret_array=np.empty(no_of_params_for_spec_mod)
    spec_par_inits=hypercube[:no_of_params_for_spec_mod]

    #select only the spectral index terms
    spec_inits = spec_par_inits[:no_of_comps]
    #select only the spectral curvature terms
    curve_inits = spec_par_inits[no_of_comps:]
        
    #define the prior for the spectral indexes
    ret_array[:no_of_comps] = UniformPrior(spec_min, spec_max)(spec_inits)
    ret_array[no_of_comps:] = GaussianPrior(curvature_mean, curvature_std)(curve_inits)
        
    
    #define a uniform prior from prior_lower to prior_upper
    uniform_prior_func = lambda vars: UniformPrior(vars[0], vars[1])(vars[2])

    #define a gaussian prior around a mean with std
    gauss_prior_func = lambda vars: GaussianPrior(vars[0],vars[1])(vars[2])
    
    if no_of_fitted_noise!=0:
        noise_inits = hypercube[len(spec_par_inits):len(spec_par_inits)+no_of_fitted_noise]
        for i in range(len(noise_inits)):
            init_vars = [noise_prior_lower[i],noise_prior_upper[i],noise_inits[i]]
            #print (init_vars)
            ret_array= np.append(ret_array, uniform_prior_func(init_vars))
    else:
        pass

    if no_of_calibrated!=0:
        #print (no_of_calibrated)
        zero_inits = hypercube[len(spec_par_inits)+no_of_fitted_noise:len(spec_par_inits)+no_of_fitted_noise+no_of_calibrated]
        scale_inits = hypercube[len(spec_par_inits)+no_of_fitted_noise+no_of_calibrated:]
        #print (zero_inits)
        #print (scale_inits)
        for i in range(len(zero_inits)):
            init_vars = [zero_lev_prior_means[i],zero_lev_prior_std[i],zero_inits[i]]
            ret_array= np.append(ret_array, gauss_prior_func(init_vars))
        #for i in range(len(scale_inits)):
        #    init_vars = [scale_prior_mean[i],scale_prior_std[i],scale_inits[i]]
        #    ret_array= np.append(ret_array, gauss_prior_func(init_vars))
        scales = UniformPrior(scale_prior_lower,scale_prior_upper)(scale_inits)
        ret_array = np.append(ret_array,scales)

    else:
        pass
    #print ("array returned")
    #print (ret_array)
    return ret_array

if print_vals_as_calc == True:
    if no_of_comps==2:
            print ("testing the prior")
            if no_to_fit_curve==no_of_comps:
                init_vals_for_prior = np.append(np.array([0.13,0.5,0.2,0.5]),0.5*np.ones(nDims-no_of_params_for_spec_mod))
                
            else:
                init_vals_for_prior = np.concatenate((np.array([0.49,0.5]),0.4*np.ones(no_to_fit_curve),0.5*np.ones(nDims-no_of_params_for_spec_mod)))
            
            prior(init_vals_for_prior)
            print ("testing likelihood")
            print (likelihood1(prior(init_vals_for_prior)))
    if no_of_comps==3:
            print ("testing the prior")
            if no_to_fit_curve==no_of_comps:
                init_vals_for_prior = np.append(np.array([0.49,0.5,0.51,0.4,0.55,0.6]),0.9*np.ones(nDims-no_of_params_for_spec_mod))
                
            else:
                init_vals_for_prior = np.concatenate((np.array([0.49,0.5,0.51]),0.4*np.ones(no_to_fit_curve),0.9*np.ones(nDims-no_of_params_for_spec_mod)))
            
            prior(init_vals_for_prior)
            print ("testing likelihood")
            print (likelihood1(prior(init_vals_for_prior)))

    
def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


print ("running polychord nlive=",settings.nlive)
#| Run PolyChord

#| PERFORM NESTED SAMPLING TO DETERMINE LOCATIONS OF CLUSTERS IN THE LOG LIKELIHOOD SURFACE
output = pypolychord.run_polychord(likelihood1, nDims, nDerived, settings, prior, dumper)

#| Create a paramnames file

paramnames = [('p%i' % i, r'n_%i' % i) for i in range(1,nDims+1)]
output.make_paramnames_files(paramnames)

#Plot the Nested Sampling results

mpl.rc("axes", titlesize=20,labelsize=16)
print ("loading samples")
try:
    print (settings.base_dir + '/' + settings.file_root)
    samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)

    print ("samples loaded")
    #plot the samples
    print (['p%i' %i for i in range(nDims)])
    print ("ploting samples")
    fig, axes = samples.plot_2d(['p%i' %i for i in range(1,nDims+1)])
    
    #fig.xticks(fontsize=15)
    #fig.yticks(fontsize=15)
    fig.savefig(path+'sampled_posterior.png')
    #plt.show()
except:
    print ("======================ERROR=======================================")
    print ("could not load files, there is an error check the chains files for this run")
