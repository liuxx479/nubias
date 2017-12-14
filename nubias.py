import numpy as np
from scipy import *
import sys, os
from halotools import sim_manager
import WLanalysis
import h5py
from emcee.utils import MPIPool 

Lbox = 512.0
Lgrid = 256.0 ## the grid used for computing power spectrum
ibins=arange(Lgrid)
nn=(Lbox/Lgrid)**2

idir = '/scratch/02977/jialiu/temp/'
cosmo_arr = loadtxt('/work/02977/jialiu/neutrino-batch/cosmo_jia_arr.txt',dtype='string')
nsnaps_arr = loadtxt('/work/02977/jialiu/neutrino-batch/nsnaps_cosmo_jia.txt')
cosmo_arr = append(cosmo_arr,'mnv0.60000_om0.30000_As2.1000')
nsnaps_arr = append(nsnaps_arr,67)

columns_to_keep_dict = {'halo_id':              (0, 'i8'),
                        #'halo_descid':          (1, 'i8'),
                        'halo_mvir':            (2, 'f4'),
                        #'halo_vmax':            (3, 'f4'),
                        #'halo_vrms':            (4, 'f4'),
                        #'halo_rvir':            (5, 'f4'),
                        #'halo_rs':              (6, 'f4'),
                        #'halo_np':              (7, 'f4'),
                        'halo_x':               (8, 'f4'),
                        'halo_y':               (9, 'f4'),
                        'halo_z':               (10, 'f4'),
                        #'halo_vx':              (11, 'f4'),
                        #'halo_vy':              (12, 'f4'),
                        #'halo_vz':              (13, 'f4'),
                        #'halo_jx':              (14, 'f4'),
                        #'halo_jy':              (15, 'f4'),
                        #'halo_jz':              (16, 'f4'),
                        #'halo_spin':            (17, 'f4'),
                        #'halo_rs_klypin':       (18, 'f4'),
                        #'halo_mvir_all':        (19, 'f4'),
                        #'halo_m200b':           (20, 'f4'),
                        #'halo_m200c':           (21, 'f4'),
                        #'halo_m500c':           (22, 'f4'),
                        #'halo_m2500c':          (23, 'f4'),
                        #'halo_xoff':            (24, 'f4'),
                        #'halo_voff':            (25, 'f4'),
                        #'halo_spin_bullock':    (26, 'f4'),
                        #'halo_b_to_a':          (27, 'f4'),
                        #'halo_c_to_a':          (28, 'f4'),
                        #'halo_ax':              (29, 'f4'),
                        #'halo_ay':              (30, 'f4'),
                        #'halo_az':              (31, 'f4'),
                        #'halo_b_to_a_500c':     (32, 'f4'),
                        #'halo_c_to_a_500c':     (33, 'f4'),
                        #'halo_ax_500c':         (34, 'f4'),
                        #'halo_ay_500c':         (35, 'f4'),
                        #'halo_az_500c':         (36, 'f4'),
                        #'halo_T/|U|':           (37, 'f4'),
                        #'halo_m_pe_Behroozi':   (38, 'f4'),
                        #'halo_m_pe_diemer':     (39, 'f4'),
                        #'halo_halfmass_radius': (40, 'f4'),
                        }
    
def ps (pos):
    '''for a list of 3D positions, return the power spectrum
    '''
    #print 'gridding'
    grid = histogramdd(pos/Lbox*Lgrid,bins=[ibins,ibins,ibins])[0]
    grid = grid/mean(grid) - 1.0
    #print 'computing 3d power spectrum'
    k, ps3d = WLanalysis.PowerSpectrum3D(grid)
    return 2*pi*k/Lbox, ps3d/(Lbox/nn)**3

def process_files (cosmosnap, mcut=arange(11.0, 14.5, 0.5), dataset_name='Subsample', bins=50):    
    '''compute Pmm, Phh (several cuts) for cosmo, snap
    '''
    cosmo, snap = cosmosnap
    subsample_fn = idir+cosmo+'/snapshots_subsample/snapshot_%03d_idmod_101_0.hdf5'%(snap)
    rockstar_fn = idir+cosmo+'/rockstar/out_%i.list'%(snap)
    out_fn = '/work/02977/jialiu/nubias/Phh/Phh_%s_%03d.npy'%(cosmo, snap)
    out_arr = zeros(shape=(len(mcut)+3, bins)) ## k, Pmm, Phh of N+1 bins
    
    if not os.path.isfile(rockstar_fn) or not os.path.isfile(rockstar_fn):
        ### skips if files do not exist for some reason
        print 'Warning: file not exist, cosmo, snap'
        return
    if os.path.isfile(out_fn): ###### in case the code breaks
        return
    
    ######### read subsample files
    print 'Opening particle files:',subsample_fn
    f=h5py.File(subsample_fn,'r')
    dataset = f[dataset_name]
    particle_pos = dataset['Position']/1e3
    dataset=0 ## release memory
    out_arr[:2] = ps(particle_pos)
    particle_pos=0 ## release memory
    
    ######### read rockstar files
    print 'Opening rockstar files:', rockstar_fn
    reader = sim_manager.TabularAsciiReader(rockstar_fn, columns_to_keep_dict) 
    rock_arr = reader.read_ascii() 
    rock_pos = array([rock_arr['halo_x'],rock_arr['halo_y'],rock_arr['halo_z']]).T
    logM_arr = log10(rock_arr['halo_mvir'])
    rock_arr=0 ## release memory
    
    out_arr[2] = ps(rock_pos)[1]
    
    ######### apply mass cuts to halos
    jjj = 2
    for imcut in mcut:
        print 'Applying mass cut:', imcut, cosmo, snap
        jjj += 1
        if amax(logM_arr)<=imcut: ### no halo above this mass
            break
        out_arr[jjj] = ps(rock_pos[logM_arr>imcut])[1]
    
    save(out_fn,out_arr)

all_snaps = []
for i in range(len(cosmo_arr)):
    for isnap in arange(30, nsnaps_arr[i]):
        all_snaps.append([cosmo_arr[i], int(isnap)])

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(process_files, all_snaps)
pool.close()
