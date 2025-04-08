import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy
from itertools import combinations
from astropy.modeling.powerlaws import LogParabola1D

import traceback

class bayes_mod():
    def __init__(self,obs_maps,obs_freqs,inv_noise_mats,EDGES_beams,EDGES_temps_at_calib_LSTs_and_freqs,EDGES_errs,EDGES_inv_noise_mats,freqs_for_T_v_LST_comp,LSTs_for_comparison,save_root,un_obs_marker=-32768,no_of_comps=2,map_prior_std=1000,map_prior_spec_index=-2.6,map_prior_f0=200):
        
        """variables are:
        
        obs_maps: a np array containing the observed maps in healpix form (entry [v,:] should be the map at freq v). NOTE: maps must contain the same no of pixels
        obs_freqs: a 1d np array containing the frequencies of the observed maps
        inv_noise_mats: the inverse noise covariance matrices for each pixel shape is (npix,nv,nv)
        un_obs_marker: the value used to indicate that a pixel is unobserved
        EDGES_beams: a set of beams at all freqs and LSTs that we want to use for calibrating the model
        EDGES_temps_at_calib_LSTs_and_freqs: the temperature of the sky as seen by EDGES for all the LSTs and freqs we use for calibration
        EDGES_errs: the uncertainty on the temps observed by EDGES
        freqs_for_T_v_LST_comp: the freqs at whice we will compare our model predictions to the EDGES T v LST traces
        no_of_comps: an integer number of components to use
        map_prior_std: the amplitude range (in Kelvin) over which we integrated the component maps during marginalisation default is 1000.
        
        NOTE: for this version of the code component spectra are single-index power-laws. Future versions will need to use spectra objects, as this will allow us to have arbitary parametarisation"""
        
        self.freqs = obs_freqs#/f0

        self.data_vecs = np.empty((obs_maps.shape[1],obs_maps.shape[0],1))
        
        self.Ninvs=inv_noise_mats

        
        #misalainous params
        self.no_of_comps = no_of_comps
        self.no_of_pixels = obs_maps.shape[1]
        self.no_of_freqs = len(obs_freqs)

        self.D_T_inv = np.diag((1/((obs_freqs/map_prior_f0)**(2*map_prior_spec_index)))) # the assumed spectral behaviour of the prior inverse covariance for the true sky maps 
        
        self.map_prior_var = map_prior_std**2 #the assumed prior variance for the true sky map at the reference frequency

        print ("===============================================")
        print ("true map prior variance at ref freq of: ",map_prior_f0," is: ",self.map_prior_var," i.e. std of: ",map_prior_std)
        print ("the assumed spectral behaviour of the prior inverse covariance for the true sky maps (as a matrix) is:")
        print (np.diag(self.D_T_inv))
        print ("spec index for this prior inv covar is")
        print (np.log(np.diag(self.D_T_inv)[:-1])/(np.log(self.freqs[:-1])-np.log(408)))
        
        

        unobs_pixs = obs_maps==un_obs_marker #a boolian array that indicates which pixels have been observed 
        self.no_of_obs_map = np.count_nonzero(~unobs_pixs,axis=0)
        self.pix_to_disregard = self.no_of_obs_map<self.no_of_comps #for any pixel that has fewer observations than comps to fit, mask out predictions
        self.pix_to_regard = self.no_of_obs_map>=1 #only include pixels with at least 1 observed frequency in the likelihood calculations

        self.fully_obs_pix = self.no_of_obs_map==self.no_of_freqs
        self.partial_obs_pix = self.pix_to_regard * (self.no_of_obs_map<self.no_of_freqs)


        self.no_of_defined_pix = np.sum(self.pix_to_regard)
        self.pix_for_map_gen = self.no_of_obs_map>=self.no_of_comps
        self.no_of_pix_for_map_gen = np.sum(self.pix_for_map_gen)
        self.no_of_partial_obs_pix = np.sum(self.partial_obs_pix)
        self.no_of_fully_obs_pix = np.sum(self.fully_obs_pix)
        print ("===============================================")
        print ("no of pix with >=1 observation")
        print (self.no_of_defined_pix)
        print ("no of pix with < n_v but >=1 observations")
        print (self.no_of_partial_obs_pix)
        print ("no of fully observed pixels")
        print (self.no_of_fully_obs_pix)
    
        

        self.mean_Ninv = np.mean(inv_noise_mats[self.pix_to_regard],axis=0)
        #generate the inverse noise matrix for each pixel
        #self.Ninvs = np.empty((self.no_of_pixels,Ninv_temp.shape[0],Ninv_temp.shape[1]))
        for p in range(0,self.no_of_pixels):

            #get the data vectors into the right order for numpy array broadcasting rules
            self.data_vecs[p,:,0] = obs_maps[:,p]
            self.data_vecs_trans = np.reshape(self.data_vecs,(self.no_of_pixels,1,obs_maps.shape[0]))
        
        

        
       
        try:
            self.freqs_for_T_v_LST_comp = freqs_for_T_v_LST_comp
            self.EDGES_beams = (1/(4*np.pi))*EDGES_beams*hp.nside2pixarea(nside=hp.npix2nside(self.no_of_pixels))
            self.EDGES_noise_det_term = -np.sum(np.log(2*np.pi*EDGES_errs))
        except:
            print ("freqs cant be divided setting to None")
            self.freqs_for_T_v_LST_comp = None
            self.EDGES_beams = None
            self.EDGES_noise_det_term = None
            
        self.EDGES_temps = EDGES_temps_at_calib_LSTs_and_freqs
        self.EDGES_inv_noise_mats = EDGES_inv_noise_mats
        self.EDGES_errs = EDGES_errs
        self.LSTs_for_comparison = LSTs_for_comparison
        print ("EDGES freqs to check are:")
        print (self.freqs_for_T_v_LST_comp)
        print ("LSTs to test:")
        print (self.LSTs_for_comparison)
        #print ("EDGES Low Band powerlaw temps at LSTs")
        #print (self.EDGES_temps)
        #print ("EDGES Low Band errors")
        #print (EDGES_errs)
        print ("EDGES det term:",self.EDGES_noise_det_term)
        self.save_root = save_root
        self.select_index1 = np.tile(np.array([False,True,False]),no_of_comps)
        self.select_index2 = np.tile(np.array([False,False,True]),no_of_comps)

    def gen_A(self,spec_params,freqs_for_mix_mat,print_vals=False):

        

        A = np.empty(shape=(len(freqs_for_mix_mat),self.no_of_comps))
        for c in range(self.no_of_comps):
            pars = spec_params[3*c:3*(c+1)]
            #print (pars)
            spec = LogParabola1D(1,pars[0],-pars[1],-pars[2])(freqs_for_mix_mat)
            A[:,c] = spec

        #print (A)
        if print_vals==True:
            print ("=======================")
            print ("printing the A matrix and plotting comp spectra for pars:")
            print (spec_params)
            print ("A:")
            print (A)
            fig = plt.figure(figsize=(6,12))
            for i in range(self.no_of_comps):
                plt.subplot(self.no_of_comps,1,i+1)
                plt.plot(freqs_for_mix_mat,A[:,i],label="spec_for_comp="+str(i+1))
                plt.legend(loc="upper right")
            plt.suptitle("spectral indexes are: "+str(spec_params))
            plt.savefig(self.save_root+"spec="+str(spec_params)+"_A_specs.png")
            plt.close("all")
        return A
    def likelihood(self,spec_params,noise_estimates,freqs_to_fit_noise,scale_estimates,zero_level_estimates,freqs_to_calibrate,joint_prior_func,reject_criterion=None,print_vals=False,condition_no_threshold=1e+7,height_threshold=0.1):
        """params are:
        spec_params: the spectral indexes to use for components
        reject_criterion: if two or more spec indexes are too close in value return -inf
        """
        

        fail_ret = -1e+30#float("NaN")#-np.inf # the value to return if the spec_params fail one of our tests

        #print_vals=True

        #check if any of the spectral indexes are repeated or are within the rejection crition of another spectral index  
        if reject_criterion==None:
            pass
        else:
            #check for spec indexes (we dont check for the spectral curvature or the ref freqs these can be identicle between spectra)
            sps = spec_params[self.select_index1]

            within_reject = np.any([np.count_nonzero(sps[:i] >= (sps[i]-reject_criterion)) for i in range(1,len(sps))])
            if within_reject==True:
                #print (spec_params,"  rejected for 1")
                #return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]#[-np.inf,-np.inf,-np.inf,-np.inf]
                return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]
            
            
        
        
        #*****************************************SET UP THE VARIOUS MATRICES**************************************************

        #generate the mixing matrix
        A = self.gen_A(spec_params,self.freqs)#self.freqs[:,np.newaxis] ** spec_params[np.newaxis,:] #this models the component spectra as single-index power-laws

        
       


        #generate the data calibration matrix and vector
        if np.sum(freqs_to_calibrate)==0:
            #print ("nothing to calibrate")

            a=np.identity(self.no_of_freqs)
            a_inv = np.identity(self.no_of_freqs)
            b=np.zeros((self.no_of_freqs,1))
        else:
            a_estimates = np.ones(self.no_of_freqs)
            a_estimates[freqs_to_calibrate] = scale_estimates
            #print (a_estimates)
            b_estimates = np.zeros(self.no_of_freqs)
            b_estimates[freqs_to_calibrate] = zero_level_estimates
            a = np.diag(a_estimates)
            a_inv = np.diag(1/a_estimates)
            b = np.reshape(b_estimates,(self.no_of_freqs,1))

        #print (a)
        #print (a_inv)
        #print (b)
        #generate the template noise covariance matrix
        if np.sum(freqs_to_fit_noise)==0:
            inv_noise_mats = self.Ninvs[self.pix_to_regard]
            mean_noise_mat = self.mean_Ninv
        else:
            #fill in the diagonals corresponding to the freqs we are fitting the noise
            temp_diag = np.zeros(self.no_of_freqs)
            temp_diag[freqs_to_fit_noise] = 1/noise_estimates**2
            temp = np.diag(temp_diag)

            inv_noise_mats = self.Ninvs[self.pix_to_regard] + temp
            mean_noise_mat = self.mean_Ninv + temp
            #print (mean_noise_mat)
        
        #rescale the noise to account for the calibration
        if np.sum(freqs_to_calibrate)==0:
            calibrated_inv_noise_mats = inv_noise_mats
            test_mat = A.T @ mean_noise_mat @ A 
        else:
            calibrated_inv_noise_mats = a_inv.T @ inv_noise_mats @ a_inv
            test_mat = A.T @ a_inv @ mean_noise_mat @ a_inv @ A



        #compute the Lambda_p matrices for all pixels with at least 1 observation
        Lambda_ps = A.T @ calibrated_inv_noise_mats @ A

        #compute S^T D^-1 S
        try:
            STDS = A.T @ self.D_T_inv @ A

            STDS_inv = np.linalg.inv(STDS)
        except:
            #print ("couldnt compute STS_inv")
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]
        #compute the matrix for the determinant 
        mat_for_det = self.map_prior_var*(STDS_inv @ Lambda_ps) + np.identity(self.no_of_comps)

        #compute the matrix for the inverse 
        mat_for_inv = Lambda_ps + (1/self.map_prior_var)*STDS

        #test that the mean condition number doesn't exceed our threshold
        mean_cond_no = np.mean(np.linalg.cond(mat_for_inv))
        overall_test_val = mean_cond_no<condition_no_threshold
        if overall_test_val == False:
            #print (mean_cond_no)
            #print (overall_test_val)
            #print ("mean condition number exceeded threshold")
            #return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]#[-np.inf,-np.inf,-np.inf,-np.inf] #if the matrix is singular return -inf
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]

        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #compute the determinant term
        try:
            #calculate the term 2 (-ln(det(Lambda_p)))
            t2s = -1 * np.linalg.slogdet(mat_for_det)[1] #only calculate for the pixels that have >=self.no_of_comps observations
        except np.linalg.LinAlgError:
            #print ("failed to calculate log determinant")
            #return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]#[-np.inf,-np.inf,-np.inf,-np.inf] #if the matrix is singular return -inf
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]

        t2 = np.sum(t2s)
        #t2 = 0

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #generate the calibrated data vectors
        if np.sum(freqs_to_calibrate)==0:
            cal_dat_vecs = self.data_vecs[self.pix_to_regard]
            cal_dat_vecs_trans = np.reshape(cal_dat_vecs,(self.no_of_defined_pix,1,self.no_of_freqs))
        else:
            cal_dat_vecs = a @ self.data_vecs[self.pix_to_regard] + b
            cal_dat_vecs_trans = np.reshape(cal_dat_vecs,(self.no_of_defined_pix,1,self.no_of_freqs))
        
        #generate the bp vectors
        bps = A.T @ calibrated_inv_noise_mats @ cal_dat_vecs
        bps_trans = np.reshape(bps,(self.no_of_defined_pix,1,self.no_of_comps))

        #add on the offsets caussed by the non zero mean of the map prior
        shifted_bps = bps #+ self.C0_inv_mu
        shifted_bps_trans = bps_trans #+ self.C0_inv_mu_trans
        #print ("============================================================")
        #print ("============================================================")
        #print ("============================================================")
        #print (bps)
        #print ("============================================================")
        #print (shifted_bps)
        #print ("============================================================")
        #print (bps_trans)
        #print ("============================================================")
        #print (shifted_bps_trans)

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            #calculate term 1 (bp.T @ Cp_inv @ bp) for each pixel
            mean_comp_maps_for_obs_pix = np.linalg.solve(mat_for_inv, shifted_bps)
            t1s = shifted_bps_trans @ mean_comp_maps_for_obs_pix 

        except np.linalg.LinAlgError:
            #print ("failed to calc mean comp maps")
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]

        t1 = np.sum(t1s)

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            t3as = cal_dat_vecs_trans @ calibrated_inv_noise_mats @ cal_dat_vecs

            t3bs = 2*np.pi*inv_noise_mats[inv_noise_mats!=0]

            t3 = np.sum(np.log(t3bs)) - np.sum(t3as)
            #print ("new value for t3 is:",t3)
            #print ("old version gave:",np.sum(t3bs) - np.sum(t3as))
        except:
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #compute the mean of the conditional distribution of sky maps for this set of spectra and calibration pars
        try:
            #print ("step 1a")
            mean_comp_maps = np.zeros(shape=(self.no_of_pixels,self.no_of_comps,1))
            mean_comp_maps[self.pix_to_regard] = mean_comp_maps_for_obs_pix
            #print ("mean comp maps")
            #print (mean_comp_maps)
            #compute the mean of the posterior of the skys for this set of parameters at all observed freqs
            #print ("step 2a")
            spectral_mix_mat_for_test = self.gen_A(spec_params,self.freqs_for_T_v_LST_comp,print_vals=print_vals)#self.freqs_for_T_v_LST_comp[:,np.newaxis] ** spec_params[np.newaxis,:]
            #print ("spectral mix mat")
            #print (spectral_mix_mat_for_test)
            mean_sky_preds = spectral_mix_mat_for_test @ mean_comp_maps
            mean_sky_preds[~self.pix_to_regard] = float("NaN")
            #print ("mean sky preds")
            #print (mean_sky_preds)
            #print (mean_sky_preds.shape)
            #print (self.no_of_pixels)

            #multiply the skys by the EDGES beam at these freqs
            #print ("step 3a")
            convolved_sky_preds = mean_sky_preds * self.EDGES_beams

            #compute the integrated sky temp for each freq and LST in the freqs to compare
            integrated_skys = np.nansum(convolved_sky_preds,axis=0)

            #diff = integrated_skys - self.EDGES_temps #the difference between the model and the EDGES obs at all freqs and LSTs
            #diff_trans = np.reshape(diff,(diff.shape[0],1,diff.shape[1]))
            #diff = np.reshape(diff,(diff.shape[0],diff.shape[1],1))
            #print (diff[0,:,:])
            #print (diff_trans[0,:,:])
            #print ("step 4a")
            EDGES_likelihood_t2s = ((integrated_skys-self.EDGES_temps)/self.EDGES_errs)**2
            EDGES_likelihood_t2 = -1*np.sum(EDGES_likelihood_t2s)#-1 * np.sum(diff_trans @ self.EDGES_inv_noise_mats @ diff)
            EDGES_int_sky_temps_log_likelihood = EDGES_likelihood_t2 + self.EDGES_noise_det_term#(self.no_of_defined_pix/len(self.LSTs_for_comparison))*(EDGES_likelihood_t2 + self.EDGES_noise_det_term)
            
            #print (sky_pred_at_45)
            val=np.random.uniform(0,1)
            #if val<=0.0001:
            #    print_vals=True

            if print_vals==True:
                #print ("step 1")
                #plot the component maps for these params
                for i in range(self.no_of_comps):
                    cm = mean_comp_maps[:,i]

                    plt.subplot(1,self.no_of_comps,i+1)
                    hp.mollview(cm.flatten(),title="mean comp "+str(i+1),hold=True)
                plt.savefig(self.save_root+"spec="+str(spec_params)+"_zeros="+str(zero_level_estimates.round(3))+"_scales="+str(scale_estimates.round(3))+"_comps.png")
                
                plt.close("all")
                #plt.show()

                

            
            
                fig = plt.figure(figsize=(16,12))
                for i in range(len(self.freqs_for_T_v_LST_comp)):
                    true = self.EDGES_temps[i,:]
                    errs = self.EDGES_errs[i,:]
                    pred = integrated_skys[i,:]
                    
                    plt.subplot(15,10,i+1)
                    hp.mollview(mean_sky_preds[:,i].flatten(),title="sky pred at "+str(np.round(self.freqs_for_T_v_LST_comp[i],1)),hold=True)

                plt.suptitle("spectral indexes are: "+str(spec_params)+"\nCalibration pars, Zeros: "+str(zero_level_estimates)+" scales: "+str(scale_estimates))
                fig.subplots_adjust(hspace=0.5)
                plt.savefig(self.save_root+"spec="+str(spec_params)+"_zeros="+str(zero_level_estimates.round(3))+"_scales="+str(scale_estimates.round(3))+"_skys.png")
                #plt.show()
                plt.close("all")

                #plot the T vs LST for each of the test freqs
                fig = plt.figure(figsize=(16,12))
                for i in range(len(self.freqs_for_T_v_LST_comp)):
                    true = self.EDGES_temps[i,:]
                    errs = self.EDGES_errs[i,:]
                    pred = integrated_skys[i,:]
                    print ("==============================")
                    print ("LSTs",len(self.LSTs_for_comparison))
                    print (self.LSTs_for_comparison)
                    print ("true",len(true))
                    print (true)
                    print ("errs",len(errs))
                    print (errs)
                    print ("pred",len(pred))
                    print (pred)
                    
                    plt.subplot(15,10,i+1)
                    plt.errorbar(self.LSTs_for_comparison,true,errs,label="EDGES obs")
                    plt.plot(self.LSTs_for_comparison,pred,label="model pred")
                    plt.title("freq="+str(np.round(self.freqs_for_T_v_LST_comp[i],1))+"MHz")
                    plt.legend(loc="upper left")
                    
                plt.suptitle("spectral indexes are: "+str(spec_params)+"\nCalibration pars, Zeros: "+str(zero_level_estimates)+" scales: "+str(scale_estimates))
                fig.subplots_adjust(hspace=0.5)
                print ("saving the T vs LST plots")
                plt.savefig(self.save_root+"spec="+str(spec_params)+"_zeros="+str(zero_level_estimates.round(3))+"_scales="+str(scale_estimates.round(3))+"_T_vs_LST.png")
                print ("saved")
                #plt.show()
                plt.close("all")
                    #plt.show()
            #compute the likelihood
            #print (integrated_skys.shape)
            #print (self.EDGES_temps.shape)
            #print (self.EDGES_inv_noise_mats.shape)
            #diff = integrated_skys - self.EDGES_temps #the difference between the model and the EDGES obs at all freqs and LSTs
            #diff_trans = np.reshape(diff,(diff.shape[0],1,diff.shape[1]))
            #print (diff[0,:])
            #print (diff.bps_trans[0,:,:])


        except:
            traceback.print_exc()
            #print ("failed to comp the simulated EDGES obs")
            return [fail_ret,fail_ret,fail_ret,fail_ret,fail_ret]
        
        prior_term = 0
        tot_log_likelihood = t1 + t2 + t3 + EDGES_int_sky_temps_log_likelihood
        posterior = tot_log_likelihood + prior_term

        #print_vals=False
        if print_vals==True:
            print ("===================================================")
            print ("spec params:",spec_params)
            
            print ("term 2 (det) is",t2)
            print ("det term + prior:",prior_term + t2)
            print ("term 1 (bp.T@(Lambda_p+C0_inv)^-1 @bp is",t1)
            print ("term 3 (gp) is",t3)
            
            print ("log likelihood of observing the EDGES vals")
            print ("term from EDGES Low band temps =",EDGES_int_sky_temps_log_likelihood)
            print ("likelihood:",tot_log_likelihood)
            print ("posterior: ",posterior)

        

        return [t1, t2 + prior_term, tot_log_likelihood, posterior]
        


    def gen_comp_map_sample(self,params,freqs_to_fit_noise,freqs_to_calibrate):
        """calculate the mean and variance of the conditional distribution of component maps P(Mp|dp,S) for a specified set of spectral params.
        
        we then return a sample component map set drawn from this conditional distribution"""

        spec_params = params[:3*self.no_of_comps]
        #generate the mixing matrix
        A = self.gen_A(spec_params,self.freqs)#self.freqs[:,np.newaxis] ** spec_params[np.newaxis,:] #this models the component spectra as single-index power-laws

        #generate the data calibration matrix and vector
        if np.sum(freqs_to_calibrate)==0:
            #print ("nothing to calibrate")

            a=np.identity(self.no_of_freqs)
            a_inv = np.identity(self.no_of_freqs)
            b=np.zeros((self.no_of_freqs,1))
        else:
            zero_level_estimates = params[3*self.no_of_comps+np.sum(freqs_to_fit_noise):3*self.no_of_comps+np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate)]
            scale_estimates = params[3*self.no_of_comps+np.sum(freqs_to_fit_noise)+np.sum(freqs_to_calibrate):]
            
            a_estimates = np.ones(self.no_of_freqs)
            a_estimates[freqs_to_calibrate] = scale_estimates
            #print (a_estimates)
            b_estimates = np.zeros(self.no_of_freqs)
            b_estimates[freqs_to_calibrate] = zero_level_estimates
            a = np.diag(a_estimates)
            a_inv = np.diag(1/a_estimates)
            b = np.reshape(b_estimates,(self.no_of_freqs,1))

        #print (a)
        #print (a_inv)
        #print (b)
        #generate the template noise covariance matrix
        if np.sum(freqs_to_fit_noise)==0:
            inv_noise_mats = self.Ninvs[self.pix_to_regard]
            mean_noise_mat = self.mean_Ninv
        else:
            #fill in the diagonals corresponding to the freqs we are fitting the noise
            temp_diag = np.zeros(self.no_of_freqs)
            noise_estimates = params[3*self.no_of_comps:3*self.no_of_comps+np.sum(freqs_to_fit_noise)]
            temp_diag[freqs_to_fit_noise] = 1/noise_estimates**2
            temp = np.diag(temp_diag)

            inv_noise_mats = self.Ninvs[self.pix_to_regard] + temp
            mean_noise_mat = self.mean_Ninv + temp
            #print (mean_noise_mat)
        
        #rescale the noise to account for the calibration
        if np.sum(freqs_to_calibrate)==0:
            calibrated_inv_noise_mats = inv_noise_mats
            test_mat = A.T @ mean_noise_mat @ A 
        else:
            calibrated_inv_noise_mats = a_inv.T @ inv_noise_mats @ a_inv
            test_mat = A.T @ a_inv @ mean_noise_mat @ a_inv @ A

        #generate the set of Lambda_p matrixes
        Lambda_ps = A.T @ calibrated_inv_noise_mats @ A

        #compute S^T S
        try:
            STDS = A.T @ self.D_T_inv @ A

            STDS_inv = np.linalg.inv(STDS)
        except:
            print ("couldnt compute STS_inv")
            return np.zeros(shape=y.shape)

        #compute the matrix for the inverse 
        mat_for_inv = Lambda_ps + (1/self.map_prior_var)*STDS

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #generate the calibrated data vectors
        if np.sum(freqs_to_calibrate)==0:
            cal_dat_vecs = self.data_vecs[self.pix_to_regard]
            cal_dat_vecs_trans = np.reshape(cal_dat_vecs,(self.no_of_defined_pix,1,self.no_of_freqs))
        else:
            cal_dat_vecs = a @ self.data_vecs[self.pix_to_regard] + b
            cal_dat_vecs_trans = np.reshape(cal_dat_vecs,(self.no_of_defined_pix,1,self.no_of_freqs))
        
        #generate the bp vectors
        bps = A.T @ calibrated_inv_noise_mats @ cal_dat_vecs
        bps_trans = np.reshape(bps,(self.no_of_defined_pix,1,self.no_of_comps))


        #add on the offsets caussed by the non zero mean of the map prior
        shifted_bps = bps #+ self.C0_inv_mu

        y = np.zeros(shape=(self.no_of_pixels,self.no_of_comps,1))
    

        
        
        
        
        
        mean_maps = np.empty(shape=y.shape)
        #calculate the mean of the distribution for each pixel of this component-map-set
        try:#print (spec_params)
            
            mean_maps_for_defined_pix = np.linalg.solve(mat_for_inv, shifted_bps)#Lambda_p_invs @ bps
            

            mean_maps[self.pix_to_regard]=mean_maps_for_defined_pix
            
            
        except:
        
            print ("failed to gen mean comp maps")
            return np.zeros(shape=y.shape)
        

        #calculate cholesky decomp of Lambda_p_ivns matrices
        #NOTE: we only do this for pixels with at least k observations (i.e. where Lambda_p should be non singular).
        #For the other pixels we dont add any noise. This is fine as we mask out the predictions for these anyway.
        try:
            #invert all of the Lambda_p matrices
            C_ps =  np.linalg.inv(mat_for_inv) #the covariance matrix for the conditional distribution of comp maps for each pixel
            
            C_ps_chole = np.linalg.cholesky(C_ps) #take the cholesky decomp of the covar for each pixel
        except:
            print ("cholesky decomp failed")
            return mean_maps
        #generate noise
        #noise drawn from a unit variance gaussian (covar matraix is unit variance and is no_of_comps by no_of_comps, we generate no_of_pixels samples from this)
        standard_noise_for_defined_pix = np.random.standard_normal(size=(self.no_of_defined_pix,self.no_of_comps,1))
        
        #multiply by cholesky decomposition of the covariance matrix for each pixel
        noise_for_pix_to_regard = C_ps_chole @ standard_noise_for_defined_pix
        
        #asign the noise to the defined pixels (unobs pixels are all set to zero)
        noise = np.zeros(shape=y.shape)
        noise[self.pix_to_regard] = noise_for_pix_to_regard
        #print (noise.shape)

        samps = mean_maps + noise
        
        samps[~self.pix_to_regard]=float("NaN") #mask out regions of the sky with <k obs
        #print (samps)
        return samps

