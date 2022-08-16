from glob import glob
import numpy as np
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table
import time
from astropy import constants as const
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import copy
import pickle
from itertools import product
from pyuvdata import UVData
import os
from pygdsm import GlobalSkyModel2016
import optimal_mapping_radec_grid
from direct_optimal_mapping import data_conditioning

OUTPUT_FOLDER = '/nfs/esc/hera/zhileixu/optimal_mapping/h1c_idr22/radec_grid'

OVERWRITE = False

def radec_map_making(files, ifreq, ipol,
                     p_mat_calc=True, beam_power=1):

#     print(f'\nfile:{file}\nfreq_bin:{ifreq}, Integration:{itime_array}, Pol:{ipol}')
    #print('ifreq:', ifreq)
    t0 = time.time()
    ra_center_deg = 30
    dec_center_deg = -30.7
    ra_rng_deg = 32
    n_ra = 64
    dec_rng_deg = 16
    n_dec = 32
    sky_px = optimal_mapping_radec_grid.SkyPx()
    px_dic = sky_px.calc_px(ra_center_deg, ra_rng_deg, n_ra, 
                            dec_center_deg, dec_rng_deg, n_dec)
    
    freq = np.linspace(100e6, 200e6, 1024, endpoint=True)[ifreq]
    
    #t0 = time.time()
    uv_org = UVData()
    uv_org.read(files, freq_chans=ifreq, polarizations=ipol)

    # Data Conditioning
    dc = data_conditioning.DataConditioning(uv_org, 0, ipol)
    dc.noise_calc()
    if dc.rm_flag() is None:
        print('ifreq:', ifreq, ', all flagged. Passed.')
        return
    # Optimal Mapping & A matrix
    opt_map = optimal_mapping_radec_grid.OptMapping(dc.uv_1d, px_dic, epoch='Current')
        
    # Phase center calculation
    lsts = np.unique(opt_map.uv.lst_array)
    phase_center_ra = np.mean(lsts)
    phase_center_dec = opt_map.uv.telescope_location_lat_lon_alt[0]
    phase_center_ra_deg = np.degrees(phase_center_ra)
    phase_center_dec_deg = np.degrees(phase_center_dec)
        
    file_name = OUTPUT_FOLDER+'/data/h1c_idr22_f1_%.2fMHz_pol%d_%dbeam_radec_grid.p'%(freq/1e6, ipol, beam_power)

    if OVERWRITE == False:
        if os.path.exists(file_name):
            print(file_name, 'existed, return.')
            return

    opt_map.set_a_mat()
    opt_map.set_inv_noise_mat(dc.uvn)
    map_vis = np.matmul(np.conjugate(opt_map.a_mat.T), 
                        np.matmul(opt_map.inv_noise_mat, 
                                  opt_map.data))
    map_vis = np.real(map_vis)
    beam_weight = np.matmul(np.conjugate((opt_map.beam_mat**beam_power).T), 
                            np.diag(opt_map.inv_noise_mat),)
        
    #print(f'A matrix calculated in {time.time() - t0} seconds.')
    if p_mat_calc:
        opt_map.set_p_mat()
        #print(f'P matrix calculated in {time.time() - t0} seconds.')
    else:
        opt_map.p_mat = np.nan
    
    result_dic = {'px_dic':px_dic,
                  'map_sum':map_vis,
                  'weight_sum':beam_weight,
                  'p_sum':opt_map.p_mat,}
    with open(file_name, 'wb') as f_t:
        pickle.dump(result_dic, f_t, protocol=4) 
    print(f'ifreq:{ifreq} finished in {time.time() - t0} seconds.')
    return

if __name__ == '__main__':
    #H1C part
    #nfiles2group = 1
    data_folder = '/nfs/esc/hera/H1C_IDR22/IDR2_2_pspec/v2/one_group/data'
    files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSLP2X.uvh5')))
    #data_folder = '/nfs/esc/hera/H1C_IDR22/LSTBIN/one_group/grp1'
    #files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.*.HH.OCRSL.uvh5')))
    #data_folder = '/nfs/esc/hera/H1C_IDR32/LSTBIN/all_epochs'
    #files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.sum.uvh5')))
    #data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN'
    #files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSLPX.uvh5')))
#     data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/true_foregrounds'
#     files = np.array(sorted(glob(data_folder+'/zen.grp1.of1.LST.*.HH.OCRSL.uvh5')))
    #data_folder = '/nfs/esc/hera/Validation/test-4.0.0/pipeline/LSTBIN/true_eor'
    #files = np.array(sorted(glob(data_folder+'/zen.eor.LST.*.HH.uvh5')))
    
    nthread = 2
    ifreq_arr = np.arange(175, 336, dtype=int) #band1
    #ifreq_arr = np.arange(515, 696, dtype=int) #band2
    #ifreq_arr = np.array([600,])
    #ifreq_arr = np.array([676,])
    ipol_arr = [-6, -5]
    args = product(np.expand_dims(files[3:8], axis=0), ifreq_arr[::-1], ipol_arr)
    
    #for args_t in args:
    #    print(args_t)
    #    radec_map_making(*args_t)
    
    pool = multiprocessing.Pool(processes=nthread)
    pool.starmap(radec_map_making, args)
    pool.close()
    pool.join()
    
