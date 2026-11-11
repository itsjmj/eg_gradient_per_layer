#%%
import pathlib
import sys
from datetime import timedelta, datetime 
import time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from EpsFit import EpsFit
from AuxFuncs import *
import numpy as np
import matplotlib.pyplot as plt
from SimGratPerGradLayer import SimulateGratingPeriodicGradLayer
import jax
import jax.numpy as jnp
from GenDensity import GenDensity1D

workfp = workfp = pathlib.Path(__file__).parent
matfp = pathlib.Path(__file__).parent.parent.parent / 'material_data'
tio2_mat = EpsFit(matfp/'mat.txt','PHz')

incident_order = 0
target_order = 1
wavelength = 520.0
incident_angle = 50.0 #degrees
grating_width_order = 1
grating_width = np.ceil(grating_width_order*wavelength/np.sin(np.deg2rad(incident_angle)))
grating_thickness = 150.0
base_thickness = 600.0
input_polarization = 'te'

sim_params = {'wavelengths' : [wavelength],
                'ambient_permittivity' : [1.0 + 0j],
                'substrate_permittivity' : [1.5**2 + 0j],
                'grating_permittivity' : [tio2_mat.get_eps(wavelength)],
                'grating_thickness' : grating_thickness,
                'grating_width' : grating_width,
                'waveguide_thickness' : 600.0,
                'min_size_constraint' : 20.0,
                'approx_mesh_spacing' : 2.0,
                'approx_num_terms' : 300,
                'incident_zero_order_angle' : 0, # degrees
                'incident_polarization' : input_polarization,
                'incident_order' : incident_order, #integer
                'target_reflected_order' : target_order, #integer
                'target_refletced_phase' : 0, #degrees
                'output_directory' : str(pathlib.Path(__file__).parent),
                'grating_thickness_bounds' : [100.0,200.0], #float nm
                'waveguide_thickness_bounds' : [500.0,650.0], #float nm
                'layer_with_gradient_list' : ['grating','wg'], #list of layers to include gradient[grating,wg]
                }

beta_schedule = [60.0,120.0,240.0,500.0,jnp.inf]
steps_schedule = [100,100,100,100,100]

optim_hyper_params = {'beta_schedule':beta_schedule,
                    'steps_schedule':steps_schedule,
                    'lr':0.01,
                    'eta':0.5,
                    'verbose':True,
                    'num_rand_restarts':20,
                    'rand_seed':110}

sgp = SimulateGratingPeriodicGradLayer(**sim_params)
sgp.setup_sim_geom()

#%% run test optimization
sgp = SimulateGratingPeriodicGradLayer(**sim_params)
sgp.setup_sim_geom()
(opt_density, opt_grating_thickness, opt_wg_thickness, all_loss_history, all_param_history, epoch_data) = sgp.optimize_density(jnp.ones(sgp.num_optim_params)*0.5,**optim_hyper_params)
resdict['optim grating density'] = opt_density
resdict['optim wg thickness'] = opt_wg_thickness
resdict['optim grating thickness'] = opt_grating_thickness
resdict['epoch data'] = epoch_data
resdict['all loss history'] = all_loss_history
resdict['all param history'] = all_param_history


