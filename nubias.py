import numpy as np
from scipy import *
import sys, os
from halotools import sim_manager
from scipy import fftpack
from scipy.fftpack import fftfreq, fftshift,ifftshift, ifftn
import WLanalysis
import h5py
from emcee.utils import MPIPool 

Lbox = 512.0 #Mpc/h
Ngrid = 256.0 ## the grid used for computing power spectrum
ibins=arange(Ngrid)
Lgrid = Lbox/Ngrid

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

def gridding (pos):
    grid = histogramdd(pos/Lbox*Ngrid,bins=[ibins,ibins,ibins])[0]
    grid = grid/mean(grid) - 1.0
    return grid

def smoothing (grid, isteep=10.0, kcut=1.0, RMpc=0.0):
    '''Smooth a 3D grid, using Logistic filter function. If RMpc is not 0, then use a Gaussian filter.
    '''
    kmax=2.0*pi/Lbox*Ngrid/2.0
    grid_fft = fftshift(fftpack.fftn(grid))
    z, y, x = np.indices(grid.shape)
    icenter=(x.max()-x.min())/2.0
    center = np.array([icenter, icenter, icenter])
    if grid.shape[0]%2 == 0:
        center+=0.5
    freq_cube = sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    freq_cube *= 2*pi/Lbox 
    if RMpc:
        WR = exp(-(freq_cube*RMpc)**2/2.0)
    else:
        WR = 1-1.0/(1.0+exp(-isteep*log(freq_cube/kcut)))
    WR[freq_cube>=kmax]=0.0 ### cut off at kmax
    grid_fft *= WR
    return ifftn(ifftshift(grid_fft*WR))

def ps (grid_smoothed):
    k, ps3d = WLanalysis.PowerSpectrum3D(grid_smoothed)
    return 2*pi*k/Lbox, ps3d*(Lgrid/Ngrid)**3

def bs (grid_smoothed):
    k, ps3d = WLanalysis.CrossPowerSpectrum3D(grid_smoothed**2, grid_smoothed)
    return 2*pi*k/Lbox, ps3d*(Lgrid/Ngrid)**3

def operation (pos):
    grid_smoothed = smoothing(gridding(pos))
    k,ips = ps(grid_smoothed)
    return k, ips, bs(grid_smoothed)[1]

mbins = [12, 12.5, 13, 16]
## mcut = arange(11.0, 14.5, 0.5)
def Phh_gen (cosmosnap, dataset_name='Subsample', bins=50):    
    '''compute Pmm, Phh (several cuts, both ends binned, not just lower limit) for cosmo, snap
    '''
    cosmo, snap = cosmosnap
    subsample_fn = idir+cosmo+'/snapshots_subsample/snapshot_%03d_idmod_101_0.hdf5'%(snap)
    rockstar_fn = idir+cosmo+'/rockstar/out_%i.list'%(snap)
    out_fn = '/work/02977/jialiu/nubias/Ph2h/Ph2h_%s_%03d.npy'%(cosmo, snap)
    out_arr = zeros(shape=(3+(len(mbins)-1)*2, bins)) 
    ## old: k, Pmm, Phh, then various cuts
    ## k, Pmm, Pm2m, Phh_bin1, Ph2h_bin1, Phh_bin2, Ph2h_bin2...
    
    if not os.path.isfile(subsample_fn) or not os.path.isfile(rockstar_fn):
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
    out_arr[:3] = operation(particle_pos)
    particle_pos=0 ## release memory
    
    ######### read rockstar files
    print 'Opening rockstar files:', rockstar_fn
    reader = sim_manager.TabularAsciiReader(rockstar_fn, columns_to_keep_dict) 
    rock_arr = reader.read_ascii() 
    rock_pos = array([rock_arr['halo_x'],rock_arr['halo_y'],rock_arr['halo_z']]).T
    logM_arr = log10(rock_arr['halo_mvir'])
    rock_arr=0 ## release memory
    Mhalo_max = amax(logM_arr) ## largest halo mass in catalogue
    
    ### now do for binned masses, not Mlim
    jjj=3
    for ii in arange(len(mbins)-1):
        iMmin, iMmax = mbins[ii],mbins[ii+1]
        if Mhalo_max<=iMmin:
            break
        print 'Applying mass bin:', iMmin, iMmax, cosmo, snap
        iout = operation(rock_pos[ (logM_arr>=iMmin) & (logM_arr<iMmax)])
        out_arr[jjj] = iout[1] ## ps
        out_arr[jjj+1] = iout[2] ## bs
        jjj+=2
    save(out_fn,out_arr)

def hmf_gen (cosmosnap, hist_bins=arange(10, 15.5, 0.1)):    
    '''compute Pmm, Phh (several cuts) for cosmo, snap
    '''
    cosmo, snap = cosmosnap
    rockstar_fn = idir+cosmo+'/rockstar/out_%i.list'%(snap)
    out_fn = '/work/02977/jialiu/nubias/hmf/hmf_%s_%03d.npy'%(cosmo, snap)

    if not os.path.isfile(rockstar_fn):
        ### skips if files do not exist for some reason
        print 'Warning: file does not exist, cosmo, snap'
        return
    if os.path.isfile(out_fn): ###### in case the code breaks
        return

    ######### read rockstar files
    print 'Opening rockstar files:', rockstar_fn
    reader = sim_manager.TabularAsciiReader(rockstar_fn, columns_to_keep_dict) 
    rock_arr = reader.read_ascii() 
    logM = log10(rock_arr['halo_mvir'])
    rock_arr = 0 ## release memory
    hmf=histogram(logM,bins=hist_bins)[0]
    save(out_fn,hmf)
    
all_snaps = []
for i in range(len(cosmo_arr)):
    for isnap in arange(30, nsnaps_arr[i]):
        all_snaps.append([cosmo_arr[i], int(isnap)])

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

pool.map(Phh_gen, all_snaps)
#pool.map(hmf_gen, all_snaps)
pool.close()
