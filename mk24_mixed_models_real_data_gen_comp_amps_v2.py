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

#RUN PARAMS
#================================================================================================
chunks = [[0,10000],[10000,20000],[20000,30000],[30000,40000],[40000,50000],[50000,60000],[60000,70000],[70000,80000],[80000,90000]]#[[0,5000],[5000,10000],[10000,15000],[15000,20000],[20000,25000],[25000,30000],[30000,35000],[35000,40000],[40000,45000],[45000,50000],[50000,55000],[55000,60000],[60000,65000],[65000,70000],[70000,75000]]

chunks_to_use_for_run = [7,8]#[4,5,6]#[0,1,2]#[5,6,7,8,9,10]#[0,1,2,3,4]

#MODEL PARAMS
#================================================================================================
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




test_LSTs = np.linspace(0,24,73)#[:-1]#np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])#np.array([0,2,4,6,8,10,12,14,16,18,20,22]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
#test_freqs = [47.5,75]#,250,300,350]#[45,50.005,59.985,70.007,73.931,79.960,100,125,150,159,200,408]
#test_freqs = [100,140,200]
#test_freqs = [250,300,350]

test_freqs = [45.0,50.0,60.0,70,74,80,150,159,408]
#test_freqs = [70.0,74.0,80.0]
#test_freqs = [150.0,159.0,408.0]



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

#freqs_for_T_v_LST_comp = np.array([40.0,42.5,45.0,47.5,50.0,55.0,60.0,65.0,70.0,200.0])
freqs_for_T_v_LST_comp = np.array([40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0,100.0,105.0,110.0,115.0,120.0,125.0,130.0,135.0,140.0,145.0,150.0,155.0,160.0,165.0,170.0,175.0,180.0,185.0,190.0,195.0,200.0])


main_label = "_"+str(no_of_comps)+"_comp_cal:"+str(calibrate)+"_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres="+str(np.round(np.log10(cond_no_threshold),1))+"_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if use_equal_spaced_LSTs==True:
    #LSTs_for_comparison = np.array([2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])#np.array([0,2,4,6,8,10,12,14,16,18,20,22])#np.array([2.5,18]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
    LSTs_for_comparison = np.linspace(0,24,4)[:-1]
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


#specify what to fit for each map


#set the prior for the noise (on the Haslam map)
noise_prior_lower, noise_prior_upper = np.array([0.01]),np.array([50])

print ("noise prior is from:",noise_prior_lower,"to",noise_prior_upper,"Kelvin")

n_spec_pars = 3*no_of_comps #the number of parameters for the spectra (for each comp we have: break_freq, spec_index1, spec_index2)


nv=len(freqs) #the number of freqs that have maps
no_of_fitted_noise = np.sum(freqs_to_fit_noise) #the number of freqs at which we fit noise level
no_of_calibrated = np.sum(freqs_to_calibrate) #the number of freqs at which we fit the calibration

if np.sum(freqs_to_calibrate)!=0:
    #set the prior params for the zero levels
    zero_lev_prior_std = 200*((np.array(freqs)[freqs_to_calibrate]/100)**-2.5)
    zero_lev_prior_means = np.zeros(np.sum(freqs_to_calibrate))


    #set the prior params for the scale corrections
    scale_prior_lower=0.85
    scale_prior_upper=1.2
    

    print ("zero level prior is gauss with mean 0K, stds (Kelvin):")
    print (zero_lev_prior_std)
    print ("temp scale prior is uniform from:",scale_prior_lower,"to",scale_prior_upper)





#CREATE A DIR TO STORE RESULTS
#====================================================================#


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

#LOAD THE DATASET AND THE ERROR MAPS
#====================================================================#

obs_maps = []
inv_err_maps = []
data_err_maps = []
load_path = p+"dataset_trimmed/"
for i in range(len(freqs)):
    f=freqs[i]

    print (f)
    fname1 = "map_"+str(f)+"MHz_FWHM=5_nside=32.fits"
    fname2 = "map_errs_"+str(f)+"MHz_FWHM=5_nside=32.fits"
    
    #fname2 = "noise_"+str(f)

    if freqs_to_fit_noise[i]==False:
        try:
            #m1, err_m = np.loadtxt(load_path+fname1), np.loadtxt(load_path+fname2)
            m1 = hp.read_map(load_path+fname1)
            try:
                err_m = hp.read_map(load_path+fname2)
            except:
                print ("using aprox errors")
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

print ("max and min for the inv noise mats: ",np.max(inv_noise_mats),np.min(inv_noise_mats[(inv_noise_mats!=0)]))

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



plt.savefig(path+"sky_maps_for_dataset_for_plt")
#plt.show()
fig = plt.figure(figsize=(12,16))
for i in range(len(freqs)):
    map_i = np.copy(inv_err_maps[i])
    ax = plt.subplot(5,3,int(i+1))
    map_i[(map_i==0)]=float("NaN")
    plt.axes(ax)
    hp.mollview(1/map_i,title="input errs freq="+str(freqs[i]),hold=True,notext=True,norm="log")



plt.savefig(path+"err_maps_for_dataset_for_plt")
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
plt.savefig(path+"/input_map_for_plt_mean_temps.png")


#generate a set of pre rotated EDGES beams at each of the frequencies that we want to compare the model to EDGES for
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print ("loading EDGES beams")
EDGES_beams = np.load("achromatic_EDGES_low_and_high_beam.npy")#np.empty(shape=(obs_maps.shape[1],len(freqs_for_T_v_LST_comp),len(LSTs_for_comparison)))


#Load the mock EDGES T vs LST plots (these are produced by convolving the mock sky map (before any pertubation) with the model beam at that freq)
#===============================================================================================
print ("loading the EDGES T vs LST traces")
EDGES_temps_at_calib_LSTs_and_freqs = np.load("EDGES_low_and_high_chromaticity_corrected_TvsLSTs.npy")
EDGES_errs = np.load("EDGES_low_and_high_chromaticity_corrected_TvsLST_errs.npy")
#generate the EDGES noise covar mats for each freq assuming noise for each LST is independent of the other LSTs
EDGES_inv_noise_mats = []
for i in range(len(test_freqs)):
    inv_cov = np.diag(1/EDGES_errs[i,:]**2)
    EDGES_inv_noise_mats.append(inv_cov)
EDGES_inv_noise_mats = np.array(EDGES_inv_noise_mats)
#=====================================================================================================
#set up the likelihood function
#bayes_eval = likelihood.bayes_mod(obs_maps=obs_maps,obs_freqs=freqs,inv_noise_mats=inv_noise_mats,gaussian_prior_covar_mat=gaussian_prior_covar_matrix,gaussian_prior_mean=gaussian_prior_mean,no_of_comps=no_of_comps,f0=f0,un_obs_marker=unobs_marker)

#set up the likelihood function
bayes_eval = likelihood.bayes_mod(obs_maps=obs_maps,obs_freqs=freqs,inv_noise_mats=inv_noise_mats,EDGES_beams=EDGES_beams,EDGES_temps_at_calib_LSTs_and_freqs=EDGES_temps_at_calib_LSTs_and_freqs,EDGES_errs=EDGES_errs,EDGES_inv_noise_mats=EDGES_inv_noise_mats,freqs_for_T_v_LST_comp=None,LSTs_for_comparison=None,no_of_comps=no_of_comps,save_root=root2,un_obs_marker=unobs_marker,map_prior_std=map_prior_std,map_prior_spec_index=map_prior_variance_spec_index,map_prior_f0=map_prior_variance_f0)

#test
print ("===================================")
print ("testing")
sample_map = bayes_eval.gen_comp_map_sample([120,-2.6,0.0,120,-2.1,-0.4,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],freqs_to_fit_noise,freqs_to_calibrate)
for c in range(no_of_comps):
    m = sample_map[:,c,0]
    hp.mollview(m)
    plt.savefig("test_c="+str(c)+".png")


no_of_params_for_spec_mod = no_of_comps + no_to_fit_curve 
print ("we arn't fitting f0: f0=",fixed_f0," no of comps with curved spectra is",no_to_fit_curve," no of params for spectral model is",no_of_params_for_spec_mod)


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
#------------------------------------------------

#find the mean spectral params and their standard deviations
samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)

mean_logZ = samples.logZ()#mean
std_logZ = samples.logZ(100).std()#100 posterior samples from estimate of log Z (use these to calculate standard deviation in Z)

print ("log Z",mean_logZ,"std",std_logZ)


column_names_in_dataframe = list(samples.columns.values)
print ("column names for dataframe:", column_names_in_dataframe)
nested_samples = []
for i in range(nDims):
    print (i,column_names_in_dataframe[i])
    nested_samples.append(list(samples.loc[:,column_names_in_dataframe[i]]))

    
nested_samples = np.array(nested_samples).T#samples.loc[:,par_llamo].to_numpy()
weights = samples.weight
samps_with_weights = WeightedDataFrame(nested_samples,weight=weights)
mean_params = samps_with_weights.mean().to_numpy()
std_params = samps_with_weights.std().to_numpy()
print ("posterior mean for specs:")
print (mean_params)
print ("std")
print (std_params)


spec_pars_mean = mean_params[:no_of_params_for_spec_mod]
spec_pars_std = std_params[:no_of_params_for_spec_mod]
everything_else_mean = mean_params[no_of_params_for_spec_mod:]
everything_else_std = std_params[no_of_params_for_spec_mod:]


scale_corrections = np.ones(nv)
scale_correction_errs = np.zeros(nv)
scale_corrections[freqs_to_calibrate] = everything_else_mean[np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate):]
scale_correction_errs[freqs_to_calibrate] = everything_else_std[np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate):]
zero_corrections = np.zeros(nv)
zero_correction_errs = np.zeros(nv)
zero_corrections[freqs_to_calibrate] = everything_else_mean[np.sum(freqs_to_fit_noise):np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate)]
zero_correction_errs[freqs_to_calibrate] = everything_else_std[np.sum(freqs_to_fit_noise):np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate)]

print ("params for calibration:")
print (scale_corrections)
print (zero_corrections)
no_of_samples=nested_samples.shape[0]
print ("no of samples drawn from posterior is:",no_of_samples)

fig, axes = samples.plot_2d(['p%i' %i for i in range(1,nDims+1)])

fig.set_size_inches(16, 16)

matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 
fig.savefig(path+'sampled_posterior_update.png')

#write a text file containing the key results
with open(path+"post_run_sumary_stats.txt","w") as f:
    f.write("\n")
    f.write("a sumary file for the results of the nested sampling run.\n")
    f.write("freqs with maps in input dataset \n")
    f.write(str(freqs)+"\n")
    f.write("=========================================================\n")
    f.write("log(Z)="+str(mean_logZ)+" +/- "+str(std_logZ)+"\n")
    f.write("fixed f0 set as: "+str(fixed_f0)+"\n")
    f.write("spec indexes\n")
    f.write(str(spec_pars_mean)+"\n")
    f.write("+/-\n")
    f.write(str(spec_pars_std)+"\n")
    f.write("zero level corrections\n")
    f.write(str(zero_corrections)+"\n")
    f.write("+/-\n")
    f.write(str(zero_correction_errs)+"\n")
    f.write("temp scale correction factors\n")
    f.write(str(scale_corrections)+"\n")
    f.write("+/-\n")
    f.write(str(scale_correction_errs)+"\n")

    f.close()

#create a bool array of which params need updates
#select the spec indexes 
print ("=======================================================")
print ("Seting up spectral params selecter")
if no_to_fit_curve==no_of_comps:
    print ("all comps are curved spec")
    spec_indexes_select = np.tile(np.array([False,True,False]),no_of_comps)
    spec_curvature_select =  np.tile(np.array([False,False,True]),no_of_comps)
    spec_f0_select =  np.tile(np.array([True,False,False]),no_of_comps)
    spec_not_curve_select = np.tile(np.array([False,False,False]),no_of_comps)
    print ("spec_curvature_select =",spec_curvature_select)
else:
    print ("not all comps are curved spec")
    spec_indexes_select = np.tile(np.array([False,True,False]),no_of_comps)
    spec_f0_select =  np.tile(np.array([True,False,False]),no_of_comps)
    spec_curvature_select = np.array([np.array([False,False,True])*fit_curve_for_comp for fit_curve_for_comp in fit_curved]).flatten() #the indexes for spectral curvature in the final param array
    print ("spec_curvature_select =",spec_curvature_select)
    spec_not_curve_select = np.array([np.array([False,False,not_fit_curve_for_comp]) for not_fit_curve_for_comp in ~np.array(fit_curved)]).flatten() #the indexes with no curvature in the final param array
    print ("spec_not_curve select =",spec_not_curve_select)

#| Use the samples of the spectras to generate samples of component maps
##########################################################################
import gzip
class generate_component_map_and_sky_map_samples():
    """this loads the nested samples and generates the associated set of sample component maps and sample sky maps"""
    
    def __init__(self):
        #load the samples of the poseterior for the spectra
        samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)

    

        column_names_in_dataframe = list(samples.columns.values)
        print ("column names for dataframe:", column_names_in_dataframe)
        nested_samples = []
        
        for i in range(nDims):
            print (i,column_names_in_dataframe[i])
            nested_samples.append(list(samples.loc[:,column_names_in_dataframe[i]]))
    
        nested_samples = np.array(nested_samples).T#samples.loc[:,par_llamo].to_numpy()
        weights = samples.weight

        #put the params into the correct positions
        ns_samps_arr = np.empty((len(weights),int(3*no_of_comps + no_of_fitted_noise + 2*no_of_calibrated)))
        #put all the calibration params into the sample array
        ns_samps_arr[:,n_spec_pars:] = nested_samples[:,no_of_params_for_spec_mod:]
        spec_pars_in_correct_order = np.empty((len(weights),int(3*no_of_comps)))
        #put the spectral slope params into the correct places:
        spec_pars_in_correct_order[:,spec_indexes_select] = nested_samples[:,:no_of_comps] #all compontents have a spec slope param these are the first no_of_comp params
        #put the spectral curvature params into the correct places
        spec_pars_in_correct_order[:,spec_curvature_select] = nested_samples[:,no_of_comps:no_of_params_for_spec_mod] #the curvature params for any with fitted curvature
        #for any comps that we didn't fit the curvature, we set the curvature to 0
        spec_pars_in_correct_order[:,spec_not_curve_select] = 0
        #put the reference freqs into the correct posititions    
        spec_pars_in_correct_order[:,spec_f0_select] = fixed_f0*np.ones((len(weights),no_of_comps))
        #finaly we put the spec pars into the samples array
        ns_samps_arr[:,:n_spec_pars] = spec_pars_in_correct_order
        print ("parameters in the correct order")
        print (ns_samps_arr[0,:])
        print (ns_samps_arr[100,:])
        self.weights = weights
        print ("no of samps with non zero weight: ",np.count_nonzero(weights))
        self.nested_samples = ns_samps_arr
        print ("saving the marginal posterior samples")
        np.savetxt(path+"post_samples_marginal.csv",self.nested_samples,delimiter=",")
        np.savetxt(path+"post_samples_marginal_weights.csv",self.weights,delimiter=",")
    
        
    def calc_posterior_mean_and_std_for_comp_maps(self,chunks_to_use):
        #| Use the samples of the spectras to generate samples of component maps
        no_of_samples=self.nested_samples.shape[0]
        print ("no of samples is:",no_of_samples)
        print ("generating componet map samples")
        #sample_maps=np.empty((no_of_samples,bayes_eval.no_of_pixels*no_of_comps))

        
        #no_of_samples = 5000
        
        
        

        for i in chunks_to_use:
            chunk = chunks[i]

            samp_chunk = self.nested_samples[chunk[0]:chunk[1],:]
            print ("=======================================")
            save_name = path+"post_samples_chunk_"+str(int(i+1))+"_comp_maps.npy.gz"
            try:
                os.remove(save_name)
            except:
                print("No pre exisiting comp maps")

            print (samp_chunk)
            print ("running for chunk: ",i+1," covering indexes",chunk)
            
            
            sample_maps_ret_block = self.process_chunk(samp_chunk)

            print (sample_maps_ret_block)

            print ("saving these maps:","post_samples_chunk_"+str(int(i+1))+"_comp_maps.npy.gz")
            f = gzip.GzipFile(save_name,"w")
            np.save(file=f,arr=sample_maps_ret_block)
            f.close()
            
        
                

    def process_chunk(self,sample_chunk):
        sample_maps_block=[]
        for i in range(sample_chunk.shape[0]):
            samp = sample_chunk[i]
            #print (samp)
            sample_map = bayes_eval.gen_comp_map_sample(samp,freqs_to_fit_noise,freqs_to_calibrate)
            #with open(path+"post_samples_comp_maps.csv","a") as f_object:
            #    writer_object = writer(f_object)
            #    writer_object.writerow(list(sample_map.flatten()))
            sample_maps_block.append(sample_map.flatten())
        return np.array(sample_maps_block)

#plot the comp maps and their errors
#######################################################################################################
samples_generator = generate_component_map_and_sky_map_samples()

samples_generator.calc_posterior_mean_and_std_for_comp_maps(chunks_to_use_for_run)

