#%%
import pathlib
import sys
# basefp = str(pathlib.Path(__file__).parent.parent)
# sys.path.insert(0, basefp) 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import jax
import optax
from optax import contrib
import jax.numpy as jnp
import fmmax
from GenDensity import GenDensity1D
import functools
import pathlib

#TODO figure out how custom fucntions can be modified to allow for wavelength batching over the last dimension

def _response_jax(static_state, optim_params, beta, eta):
    """Evaluate the farfield response for the gradient-layer periodic solver."""
    wavelengths = static_state['wavelengths']
    dx = static_state['dx']
    min_size_constraint = static_state['min_size_constraint']
    incident_zero_order_angle = static_state['incident_zero_order_angle']
    x_nm = static_state['x_nm']
    x_nm_fmmax = static_state['x_nm_fmmax']
    grating_permittivity = static_state['grating_permittivity']
    ambient_permittivity = static_state['ambient_permittivity']
    substrate_permittivity = static_state['substrate_permittivity']
    uniform_base_thickness = static_state['uniform_base_thickness']
    total_size_x = static_state['total_size_x']
    total_size_y = static_state['total_size_y']
    approx_num_terms = static_state['approx_num_terms']
    grating_thickness = static_state['grating_thickness']
    incident_order_index = static_state['incident_order_index']
    target_order_index = static_state['target_order_index']
    grating_thickness_bounds = static_state['grating_thickness_bounds']
    waveguide_thickness_bounds = static_state['waveguide_thickness_bounds']
    layer_with_gradient_list = static_state['layer_with_gradient_list']
    num_grating_mesh_points = static_state['num_grating_mesh_points']

    thickness_params = optim_params[num_grating_mesh_points:]
    grating_density = optim_params[:num_grating_mesh_points]
    
    def linear_scale(norm_param,bounds):
        return norm_param*(bounds[1] - bounds[0]) + bounds[0]

    # Construct the density array in the grating layer and apply the filter.
    density_filter = GenDensity1D(dx=dx, is_periodic=True, filter_width=min_size_constraint, filter_type='gauss')
    filtered_grating_density = density_filter.generate_filt_density(grating_density, beta, eta)
    grating_layer_density_2d = jnp.tile(filtered_grating_density[:, jnp.newaxis], (1, x_nm.shape[1]))

    grating_layer_permittivity = fmmax.interpolate_permittivity(
        permittivity_solid=grating_permittivity[0],
        permittivity_void=ambient_permittivity[0],
        density=grating_layer_density_2d)

    primitive_lattice_vectors = fmmax.LatticeVectors(
        u=total_size_x * fmmax.X, v=total_size_y * fmmax.Y
    )
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approx_num_terms,
        truncation=fmmax.Truncation.CIRCULAR,
    )

    in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength=wavelengths,
        polar_angle=jnp.asarray(incident_zero_order_angle),
        azimuthal_angle=jnp.asarray(0),
        permittivity=jnp.asarray(ambient_permittivity))

    eigensolve = functools.partial(
        fmmax.eigensolve_isotropic_media,
        wavelength=wavelengths[0],
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
    )

    solve_result_ambient = eigensolve(permittivity=
                                      ambient_permittivity[:,jnp.newaxis,jnp.newaxis])
    
    solve_result_grating = eigensolve(permittivity=
                                      grating_layer_permittivity)
    
    solve_result_base = eigensolve(permittivity=
                                   grating_permittivity[:,jnp.newaxis,jnp.newaxis])
    
    solve_result_substrate = eigensolve(permittivity=
                                        substrate_permittivity[:,jnp.newaxis,jnp.newaxis])   

    if 'grating' in layer_with_gradient_list and 'wg' in layer_with_gradient_list:
        thicknesses =  [100.0, #nm distance above grating to source
                       linear_scale(thickness_params[0],grating_thickness_bounds),
                       linear_scale(thickness_params[1],waveguide_thickness_bounds),
                       500.0, #nm substrate thickness
                       ]
        
        stack_layer_solve_results = [solve_result_ambient,
                                solve_result_grating,
                                solve_result_base,
                                solve_result_substrate]
        
    elif 'grating' in layer_with_gradient_list and 'wg' not in layer_with_gradient_list and uniform_base_thickness>0.0:
        thicknesses =  [100.0, #nm distance above grating to source
                      linear_scale(thickness_params[0],grating_thickness_bounds),
                      uniform_base_thickness,
                      500.0, #nm substrate thickness
                      ]
        
        stack_layer_solve_results = [solve_result_ambient,
                                solve_result_grating,
                                solve_result_base,
                                solve_result_substrate]
        
    elif 'grating' in layer_with_gradient_list and 'wg' not in layer_with_gradient_list and uniform_base_thickness==0.0:
        thicknesses =  [100.0, #nm distance above grating to source
                      linear_scale(thickness_params[0],grating_thickness_bounds),
                      500.0, #nm substrate thickness
                      ]
        
        stack_layer_solve_results = [solve_result_ambient,
                                solve_result_grating,
                                solve_result_substrate]

    elif 'grating' not in layer_with_gradient_list and 'wg' in layer_with_gradient_list:
        thicknesses =  [100.0, #nm distance above grating to source
                     grating_thickness,
                     linear_scale(thickness_params[0],waveguide_thickness_bounds),
                     500.0, #nm substrate thickness
                     ]
        
        stack_layer_solve_results = [solve_result_ambient,
                    solve_result_grating,
                    solve_result_base,
                    solve_result_substrate]
        

    source_amplitude_vector = jnp.zeros((2 * expansion.num_terms, 1),
                                          dtype=complex).at[incident_order_index, 0].set(1)

    incident_poynting_flux, _ = fmmax.amplitude_poynting_flux(
        forward_amplitude=source_amplitude_vector,
        backward_amplitude=jnp.zeros_like(source_amplitude_vector),
        layer_solve_result=solve_result_ambient,
    )

    source_amplitude_vector /= jnp.sqrt(jnp.sum(incident_poynting_flux, axis=-2, keepdims=True))

    # Create the scattering matrix of the whole device stack.
    scattering_matrix = fmmax.stack_s_matrix(
        layer_solve_results=stack_layer_solve_results,
        layer_thicknesses=thicknesses)

    backward_amplitude_at_stack_end = scattering_matrix.s21 @ source_amplitude_vector

    _, reflected_flux = fmmax.amplitude_poynting_flux(
        forward_amplitude=source_amplitude_vector,
        backward_amplitude=backward_amplitude_at_stack_end,
        layer_solve_result=solve_result_ambient,
    )

    result = jnp.sqrt(jnp.abs(reflected_flux[0, target_order_index, :])) * jnp.exp(
        1j * jnp.angle(backward_amplitude_at_stack_end[0, target_order_index, :])
    )

    return result

class SimulateGratingPeriodicGradLayer:
    """Periodic solver variant that also optimizes selected layer thicknesses."""

    def __init__(self,
                wavelengths = [518.6],
                ambient_permittivity = [1.0 + 0j],
                substrate_permittivity = [1.0**2 + 0j],
                grating_permittivity = [3.5**2 + 0j],
                grating_thickness = 500.0, #float nm
                grating_width = 2000.0, #float nm
                waveguide_thickness = 0.0, #float nm
                min_size_constraint = 100.0, #float nm
                approx_mesh_spacing = 2.0, #float nm
                approx_num_terms = 500, #int
                incident_zero_order_angle = 0, # degrees
                incident_polarization = 'tm', #str [te, tm]
                incident_order = 1, #integer
                target_reflected_order = 0, #integer
                target_reflected_phase = 0, #degrees
                output_directory = str(pathlib.Path().cwd()), #str
                grating_thickness_bounds = [50.0,250.0], #float nm
                waveguide_thickness_bounds = [305.0,505.0], #float nm
                layer_with_gradient_list = ['grating','wg'], #list of layers to include gradient[grating,wg]
                ):
        
        self.wavelengths = jnp.asarray(wavelengths)
        self.ambient_permittivity = jnp.asarray(ambient_permittivity) 
        self.substrate_permittivity = jnp.asarray(substrate_permittivity)
        self.grating_permittivity = jnp.asarray(grating_permittivity)
        self.grating_thickness = grating_thickness
        self.uniform_base_thickness = waveguide_thickness
        self.grating_width = grating_width
        self.approx_mesh_spacing = approx_mesh_spacing
        self.min_size_constraint = min_size_constraint
        self.approx_num_terms = approx_num_terms
        self.incident_zero_order_angle = jnp.deg2rad(incident_zero_order_angle)
        self.incident_pol = incident_polarization
        self.incident_order = incident_order
        self.target_order = target_reflected_order
        self.target_phase = jnp.deg2rad(target_reflected_phase)
        self.output_dir = pathlib.Path(output_directory)
        self.grating_thickness_bounds = jnp.asarray(grating_thickness_bounds)
        self.waveguide_thickness_bounds = jnp.asarray(waveguide_thickness_bounds)
        self.layer_with_gradient_list = layer_with_gradient_list

        pass


    def setup_sim_geom(self):
        """Set up the numerical grid and expansion basis for the gradient-layer layout."""
        self.total_size_x = self.grating_width
        nx = int(self.total_size_x/self.approx_mesh_spacing) + int(not int(self.total_size_x/self.approx_mesh_spacing)%2) #force odd number of grid points
        x = jnp.linspace(-self.total_size_x/2,self.total_size_x/2,nx)
        self.dx = jnp.abs(x[1] - x[0])
        ny = 3
        self.total_size_y = ny*self.dx
        y = jnp.linspace(-self.total_size_y/2,self.total_size_y/2,ny)
        self.num_grating_mesh_points = nx
        self.x_nm, self.y_nm = jnp.meshgrid(x,y,indexing='ij')
        self.x_nm_fmmax = jnp.linspace(0,self.total_size_x,nx)
        self.y_nm_fmmax = jnp.linspace(0,self.total_size_y,ny)

        # get correct amplitude vector indices of desired fourier orders
        primitive_lattice_vectors = fmmax.LatticeVectors(
            u=self.total_size_x * fmmax.X, v=self.total_size_y * fmmax.Y
        )

        expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=self.approx_num_terms,
            truncation=fmmax.Truncation.CIRCULAR,
        )

        self.incident_order_index = jnp.argmin(jnp.abs(expansion.basis_coefficients[:,0]-
                                                       self.incident_order))
        
        self.target_order_index = jnp.argmin(jnp.abs(expansion.basis_coefficients[:,0]-
                                                self.target_order))
        
        # select correct polarization component in amplitude vector
        if self.incident_pol == 'tm':
            self.incident_order_index += expansion.num_terms
            self.target_order_index += expansion.num_terms

        self.num_optim_params = int(self.num_grating_mesh_points + len(self.layer_with_gradient_list))
        pass
    
    def generate_expansion(self, optim_params, beta, eta, filter_density=True):
        """Build layer solve results for the thickness- and density-parameterized stack."""
        # Keep the expensive expansion setup outside the response function so
        # JAX can propagate gradients correctly through the optimization loop.


        def linear_scale(norm_param,bounds):
            if norm_param <= 1.0:
                return norm_param*(bounds[1] - bounds[0]) + bounds[0]
            else:
                return norm_param

        thickness_params = optim_params[self.num_grating_mesh_points:]
        grating_density = optim_params[:self.num_grating_mesh_points]

        # construct density array in grating layer
        density_filter = GenDensity1D(dx=self.dx, is_periodic=True, filter_width=self.min_size_constraint, filter_type='gauss')
        filtered_grating_density = density_filter.generate_filt_density(grating_density, beta, eta)
        if filter_density == False:
            filtered_grating_density = grating_density
        grating_layer_density_2d = jnp.tile(filtered_grating_density[:, jnp.newaxis], (1, self.x_nm.shape[1]))

        grating_layer_permittivity = fmmax.interpolate_permittivity(
            permittivity_solid=self.grating_permittivity[0],
            permittivity_void=self.ambient_permittivity[0],
            density=grating_layer_density_2d)

        primitive_lattice_vectors = fmmax.LatticeVectors(
            u=self.total_size_x * fmmax.X, v=self.total_size_y * fmmax.Y
        )
        expansion = fmmax.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=self.approx_num_terms,
            truncation=fmmax.Truncation.CIRCULAR,
        )

        in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
            wavelength=self.wavelengths,
            polar_angle=jnp.asarray(self.incident_zero_order_angle),
            azimuthal_angle=jnp.asarray(0),
            permittivity=jnp.asarray(self.ambient_permittivity))

        eigensolve = functools.partial(
            fmmax.eigensolve_isotropic_media,
            wavelength=self.wavelengths[0],
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
        )

        self.solve_result_ambient = eigensolve(permittivity=
                                               self.ambient_permittivity[:,jnp.newaxis,jnp.newaxis])
        
        self.solve_result_grating = eigensolve(permittivity=
                                               grating_layer_permittivity)
        
        self.solve_result_base = eigensolve(permittivity=
                                            self.grating_permittivity[:,jnp.newaxis,jnp.newaxis])
        
        self.solve_result_substrate = eigensolve(permittivity=
                                            self.substrate_permittivity[:,jnp.newaxis,jnp.newaxis])   

        if 'grating' in self.layer_with_gradient_list and 'wg' in self.layer_with_gradient_list:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                linear_scale(thickness_params[0],self.grating_thickness_bounds),
                                linear_scale(thickness_params[1],self.waveguide_thickness_bounds),
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                                self.solve_result_grating,
                                                self.solve_result_base,
                                                self.solve_result_substrate]
                    
        elif 'grating' in self.layer_with_gradient_list and 'wg' not in self.layer_with_gradient_list and self.uniform_base_thickness>0.0:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                linear_scale(thickness_params[0],self.grating_thickness_bounds),
                                self.uniform_base_thickness,
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                                self.solve_result_grating,
                                                self.solve_result_base,
                                                self.solve_result_substrate]
            
        elif 'grating' in self.layer_with_gradient_list and 'wg' not in self.layer_with_gradient_list and self.uniform_base_thickness==0.0:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                linear_scale(thickness_params[0],self.grating_thickness_bounds),
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                                self.solve_result_grating,
                                                self.solve_result_substrate]

        elif 'grating' not in self.layer_with_gradient_list and 'wg' in self.layer_with_gradient_list:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                self.grating_thickness,
                                linear_scale(thickness_params[0],self.waveguide_thickness_bounds),
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                    self.solve_result_grating,
                                    self.solve_result_base,
                                    self.solve_result_substrate]
            
        elif 'grating' not in self.layer_with_gradient_list and 'wg' not in self.layer_with_gradient_list and self.uniform_base_thickness==0.0:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                self.grating_thickness,
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                                self.solve_result_grating,
                                                self.solve_result_substrate]
            
        elif 'grating' not in self.layer_with_gradient_list and 'wg' not in self.layer_with_gradient_list and self.uniform_base_thickness>0.0:
            self.thicknesses =  [100.0, #nm distance above grating to source
                                self.grating_thickness,
                                self.uniform_base_thickness,
                                500.0, #nm substrate thickness
                                ]
            
            self.stack_layer_solve_results = [self.solve_result_ambient,
                                                self.solve_result_grating,
                                                self.solve_result_base,
                                                self.solve_result_substrate]

        self.expansion = expansion

        pass

    # def response(self,optim_params,beta,eta):
    #     static_state = self.__dict__
    #     return _response_jax(static_state,optim_params,beta,eta)
    
    def response(self,optim_params,beta,eta):
        static_state = self.__dict__
        if isinstance(optim_params,dict):
            density = optim_params['density']
            layer_thickness = optim_params['thickness']
            combined_params = jnp.hstack((density,layer_thickness))
        else:
            combined_params = optim_params
        return _response_jax(static_state,combined_params,beta,eta)
    
    def optimize_density(self,
                        init_params,
                        beta_schedule=[16.0, 32.0, jnp.inf],
                        steps_schedule=[20,20,20],
                        lr=1.0,
                        eta=0.5,
                        verbose=True,
                        num_rand_restarts = 0,
                        rand_seed=100):

        assert len(init_params) == self.num_optim_params, 'Initial parameter length not equal to assigned value'

        #apply partition
        def apply_param_partition(params):
            return {'density':params[:self.num_grating_mesh_points],'thickness':params[self.num_grating_mesh_points:]}
        
        def remove_partition(partition_params):
            return jnp.hstack((partition_params['density'],partition_params['thickness']))
        
        
        def map_nested_fn(fn):
            '''Recursively apply `fn` to key-value pairs of a nested dict.'''
            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                        for k, v in nested_dict.items()}
            return map_fn
        
        label_fn = map_nested_fn(lambda k, _: k)

        max_grad_l2_norm = jnp.sqrt(init_params.size)*0.1
        density_filter = GenDensity1D(dx=self.dx, is_periodic=True, filter_width=self.min_size_constraint, filter_type='gauss')
        
        def softplus_beta(x, beta=5.0):
            return jax.nn.softplus(beta * x) / beta
        
        def linear_scale(norm_param,bounds):
            return norm_param*(bounds[1] - bounds[0]) + bounds[0]

        def loss_fn(p, beta_val):
            # dens = density_from_params(p)
            # dens = soft_clamp(p)
            dens = p
            farfield = self.response(dens, beta_val, eta)
            field = farfield[0]
            intensity = jnp.abs(field)**2
            phase = jnp.angle(field)
            
            intensity_loss = jnp.abs((1.0 - intensity))
            phase_diff = jnp.abs(jnp.angle(jnp.exp(1j * (self.target_phase - phase))))/(2*jnp.pi)
            phase_loss = phase_diff

            soft_intensity_loss = softplus_beta(intensity_loss-0.1,beta=5.0)
            soft_phase_loss = softplus_beta(phase_loss-0.02,beta=5.0)

            # self.loss_farfield = jnp.copy(farfield[0]) Note: jax does not seem to allow intermediary results
            # return 100.0*(0.6*soft_intensity_loss + 0.4*soft_phase_loss)
            return 100.0*(intensity_loss)


        # @jax.jit
        # def step(p, s, beta_val):
        #     val, g = jax.value_and_grad(loss_fn)(p, beta_val)
        #     updates, s = optimizer.update(g, s)
        #     p = optax.apply_updates(p, updates)
        #     return p, s, val
        
        def step(params, state, beta_val):
            value, grad = jax.value_and_grad(loss_fn)(params, beta_val)
            updates, state = optimizer.update(grad, state, value=value)
            params = optax.apply_updates(params, updates)
            tree_lower = {'density':0.0,'thickness':0.0}
            tree_upper = {'density':1.0,'thickness':1.0}
            params = optax.projections.projection_box(params,lower=tree_lower, upper=tree_upper)
            return params, state, value, grad
        
        # def step(params, state, beta_val):
        #     value, grad = jax.value_and_grad(loss_fn)(params, beta_val)
        #     updates, state = optimizer.update(
        #         grad, 
        #         state, 
        #         params=params,
        #         value=value,
        #         grad=grad,
        #         value_fn=lambda p: loss_fn(p, beta_val)  # ← closure for linesearch
        #     )
        #     params = optax.apply_updates(params, updates)
        #     params = optax.projections.projection_box(params, lower=0.0, upper=1.0)
        #     return params, state, value, grad

        if num_rand_restarts > 0:
            print(f"\n=== Performing {int(num_rand_restarts)} random restarts ===")
            key = jax.random.key(rand_seed)
            restart_loss_vals = np.zeros(int(num_rand_restarts))
            restart_param_vals = np.zeros((int(num_rand_restarts),init_params.size))

            for restart_idx in range(num_rand_restarts):
                key, subkey = jax.random.split(key)
                rand_params = jax.random.uniform(subkey, shape=init_params.shape, minval=0.1, maxval=0.9)

                rand_params = apply_param_partition(rand_params)

                # optimizer = optax.chain(
                # optax.clip_by_global_norm(max_grad_l2_norm),
                # optax.zero_nans(),
                # optax.adam(lr))
                # opt_state = optimizer.init(rand_params)

                # optimizer = optax.chain(
                #     optax.clip_by_global_norm(max_grad_l2_norm),
                #     optax.zero_nans(),
                #     optax.adabelief(lr,eps=1e-4))

                optimizer = optax.partition({'density':optax.chain(optax.clip_by_global_norm(max_grad_l2_norm),
                                                            optax.zero_nans(),
                                                            optax.adabelief(lr,eps=1e-4)),
                                            'thickness':optax.chain(optax.clip_by_global_norm(max_grad_l2_norm),
                                                                optax.zero_nans(),
                                                                optax.adabelief(lr,eps=1e-4))},
                                            label_fn)
                
                opt_state = optimizer.init(rand_params)
                p = rand_params

                if verbose:
                    # plot_params = soft_clamp(p)
                    pp = jnp.copy(remove_partition(p))
                    plot_params = pp[:self.num_grating_mesh_points]
                    if 'grating' in self.layer_with_gradient_list:
                        plot_thickness = linear_scale(pp[self.num_grating_mesh_points:][0],self.grating_thickness_bounds)
                    else:
                        plot_thickness = self.grating_thickness
                    plot_density = density_filter.generate_filt_density(plot_params, beta_schedule[0], eta)
                    density_filter.plot_density_1d(plot_density, plot_thickness,
                                       f'Restart {restart_idx+1:03d}, Beta = {beta_schedule[0]} initial',
                                       save_fp=self.output_dir/'current_rand_restart_density_initial.png')

                for step_idx in range(5):
                    p, opt_state, value, grad = step(p, opt_state, beta_schedule[0])
                    
                    if verbose and (step_idx % 1 == 0):
                        print(f"  restart {restart_idx+1:03d}, step {step_idx+1:03d} loss {value:.6e}")

                if verbose:
                    # plot_params = soft_clamp(p)
                    pp = jnp.copy(remove_partition(p))
                    plot_params = pp[:self.num_grating_mesh_points]
                    if 'grating' in self.layer_with_gradient_list:
                        plot_thickness = linear_scale(pp[self.num_grating_mesh_points:][0],self.grating_thickness_bounds)
                    else:
                        plot_thickness = self.grating_thickness
                    plot_density = density_filter.generate_filt_density(plot_params, beta_schedule[0], eta)
                    density_filter.plot_density_1d(plot_density, plot_thickness,
                                       f'Restart {restart_idx+1:03d}, Beta = {beta_schedule[0]} final',
                                       save_fp=self.output_dir/'current_rand_restart_density_final.png')

                pp = jnp.copy(remove_partition(p))
                restart_loss_vals[restart_idx] = value
                restart_param_vals[restart_idx,:] = pp

            best_restart_idx = np.argmin(restart_loss_vals)
            init_params = restart_param_vals[best_restart_idx]
            print(f"  restart {best_restart_idx+1} selected with loss {restart_loss_vals[best_restart_idx]:.6e}")

            if verbose:
                plot_params = init_params[:self.num_grating_mesh_points]
                if 'grating' in self.layer_with_gradient_list:
                    plot_thickness = linear_scale(init_params[self.num_grating_mesh_points:][0],self.grating_thickness_bounds)
                else:
                    plot_thickness = self.grating_thickness
                plot_density = density_filter.generate_filt_density(plot_params, beta_schedule[0], eta)
                density_filter.plot_density_1d(plot_density, plot_thickness,
                                    f'Restart {restart_idx+1:03d}, Beta = {beta_schedule[0]} initial',
                                    save_fp=self.output_dir/'selected_rand_restart_density.png')
                print(' randam restart thicknesses:',init_params[self.num_grating_mesh_points:])


        # Initialize optimizer (once, reused across epochs)
        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(max_grad_l2_norm),
        #     optax.zero_nans(),
        #     optax.adam(lr),
        #     contrib.reduce_on_plateau(
        #         patience=100,
        #         cooldown=0,
        #         factor=0.5,
        #         rtol=0.02,
        #         accumulation_size=1))
        # opt_state = optimizer.init(init_density)

        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(max_grad_l2_norm),
        #     optax.zero_nans(),
        #     optax.adam(lr))
        # opt_state = optimizer.init(init_params)

        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(max_grad_l2_norm),
        #     optax.zero_nans(),
        #     optax.adabelief(lr,eps=1e-4))
        # opt_state = optimizer.init(init_params)
        if not isinstance(init_params,dict):
            init_params = apply_param_partition(init_params)

        lr_fn = optax.schedules.cosine_decay_schedule(init_value=lr,
                                                      decay_steps=jnp.sum(jnp.array(steps_schedule)),
                                                      alpha=1e-4)

        optimizer = optax.partition({'density':optax.chain(optax.clip_by_global_norm(max_grad_l2_norm),
                                            optax.zero_nans(),
                                            optax.adabelief(lr_fn,eps=1e-4)),
                                    'thickness':optax.chain(optax.clip_by_global_norm(max_grad_l2_norm),
                                            optax.zero_nans(),
                                            optax.adabelief(lr_fn,eps=1e-4))},
                                    label_fn)
        opt_state = optimizer.init(init_params)
        
        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(max_grad_l2_norm),
        #     optax.zero_nans(),
        #     optax.lbfgs(learning_rate=lr))
        # opt_state = optimizer.init(init_params)

        # Storage across all epochs
        all_farfield_history = []
        all_loss_history = []
        all_param_history = []
        epoch_data = {'beta_values': beta_schedule, 'epoch_losses': [], 'epoch_final_params': []}

        p = init_params

        for epoch, beta in enumerate(beta_schedule):
            if verbose:
                print(f"\n=== Epoch {epoch + 1}/{len(beta_schedule)} (beta={beta}) ===")
            
            epoch_loss_history = []
            
            for step_idx in range(steps_schedule[epoch]):
                p, opt_state, value, grad = step(p, opt_state, beta)
                
                all_loss_history.append(float(jnp.copy(value)))
                epoch_loss_history.append(float(jnp.copy(value)))
                # all_param_history.append(soft_clamp(jnp.copy(p)))
                all_param_history.append(jnp.copy(remove_partition(p)))
                # lr_scale = optax.tree.get(opt_state, "scale")

                
                if verbose and (step_idx % 1 == 0):
                    pp = jnp.copy(remove_partition(p))
                    ggrad = jnp.copy(remove_partition(grad))
                    # print(f"  step {step_idx:04d} loss {value:.6e} lr scale {lr_scale:.6e}")
                    print(f"  step {step_idx:04d} loss {value:.6e}")
                    print(f'  current optim thicknesses {pp[self.num_grating_mesh_points:]}')

                if verbose and (step_idx % 5 == 0) or (step_idx == 0):
                    # plot_params = soft_clamp(p)
                    plot_params = pp[:self.num_grating_mesh_points]
                    plot_grad = ggrad[:self.num_grating_mesh_points]
                    if 'grating' in self.layer_with_gradient_list:
                        plot_thickness = linear_scale(pp[self.num_grating_mesh_points:][0],self.grating_thickness_bounds)
                    else:
                        plot_thickness = self.grating_thickness
                    plot_density = density_filter.generate_filt_density(plot_params, beta, eta)
                    density_filter.plot_density_1d(plot_density, plot_thickness,
                                       f'Epoch {epoch+1}, Step = {step_idx}, Beta = {beta}',
                                       save_fp=self.output_dir/'current_optim_density.png')
                    
                    density_filter.plot_gradient_1d(plot_grad, plot_thickness,
                                       f'Epoch {epoch+1}, Step = {step_idx}, Beta = {beta}',
                                       save_fp=self.output_dir/'current_optim_gradient.png')
                    
            
            epoch_data['epoch_losses'].append(epoch_loss_history)
            epoch_data['epoch_final_params'].append(jnp.copy(remove_partition(p)))
            
            if verbose:
                print(f"  epoch final loss: {epoch_loss_history[-1]:.6e}")

        # opt_density = density_from_params(p)
        # opt_density = soft_clamp(p)
        pp = jnp.copy(remove_partition(p))
        density_params = pp[:self.num_grating_mesh_points]
        thickness_params = pp[self.num_grating_mesh_points:]
        
        if 'grating' in self.layer_with_gradient_list and 'wg' in self.layer_with_gradient_list:
            opt_grating_thickness = linear_scale(thickness_params[0],self.grating_thickness_bounds)
            opt_wg_thickness = linear_scale(thickness_params[1],self.waveguide_thickness_bounds)
        elif 'grating' in self.layer_with_gradient_list and 'wg' not in self.layer_with_gradient_list:
            opt_grating_thickness = linear_scale(thickness_params[0],self.grating_thickness_bounds)
            opt_wg_thickness = self.uniform_base_thickness
        elif 'grating' not in self.layer_with_gradient_list and 'wg' in self.layer_with_gradient_list:
            opt_grating_thickness = self.grating_thickness
            opt_wg_thickness = linear_scale(thickness_params[0],self.waveguide_thickness_bounds)

        opt_density = density_filter.generate_filt_density(density_params, jnp.inf, eta)
        # clean edges due to differentiable beta = inf implementation
        opt_density = np.array(opt_density)
        opt_density[opt_density>=0.5] = 1.0
        opt_density[opt_density<0.5] = 0.0
        
        # Attach to self for access
        self.opt_loss_history = np.array(all_loss_history)
        self.opt_farfield_history = all_farfield_history
        self.opt_param_history = all_param_history
        self.opt_epoch_data = epoch_data
        self.opt_density = opt_density
        self.opt_grating_thickness = opt_grating_thickness
        self.opt_wg_thickness = opt_wg_thickness
        
        return opt_density, opt_grating_thickness, opt_wg_thickness, all_loss_history, all_param_history, epoch_data
    
    def export_grating_txt(self,binary_density,grating_thickness,wg_thickness,file_name,output_plot=False,delimiter=' '):

        export_x_nm = self.x_nm[:,0]

        assert binary_density.size == export_x_nm.size, 'export density size must match defined x-coord size'

        binary_density = np.array(binary_density)
        binary_density[binary_density>=0.5] = 1.0
        binary_density[binary_density<0.5] = 0.0

        rollf = np.roll(binary_density,1)
        rollf[0] = 0.0
        rollb = np.roll(binary_density,-1)
        rollb[-1] = 0.0

        left_edge = np.abs(binary_density - rollf)
        left_edge = np.argwhere(left_edge>=0.9)[:,0]
        left_edge = left_edge[0::2]

        right_edge = np.abs(binary_density - rollb)
        right_edge = np.argwhere(right_edge>=0.9)[:,0]
        right_edge = np.flip(right_edge[-1::-2])

        left_edge_x = export_x_nm[left_edge]
        right_edge_x = export_x_nm[right_edge]

        centre_x = left_edge_x/2 + right_edge_x/2
        span_x = np.abs(right_edge_x - left_edge_x)

        if output_plot:
            density_filter = GenDensity1D(dx=self.dx, is_periodic=True,
                              filter_width=self.min_size_constraint, filter_type='gauss')
            density_filter.plot_density_1d(binary_density, self.grating_thickness, file_name,
                            save_fp=self.output_dir/f'{file_name}.png')
            
        export = np.hstack((centre_x[:,np.newaxis],span_x[:,np.newaxis]))
        header = f'P = {self.total_size_x:.2f}, T = {grating_thickness:.2f}, WG = {wg_thickness:.2f}\ncentre [nm] span [nm]'
        np.savetxt(self.output_dir/f'{file_name}.txt',export,
                   header=header,fmt='%.2f',delimiter=delimiter)
        
        self.centre_x_nm = centre_x
        self.span_x_nm = span_x



    def solve_to_visualize_fields_interior_source(self,grating_density,beta,eta,filter_density=True):

        self.generate_expansion(grating_density,beta,eta,filter_density=filter_density)

        vis_thickness = [1000.0] + self.thicknesses[0:-1] + [1000.0]
        
        s_matrices_interior_before_source = fmmax.stack_s_matrices_interior(
            layer_solve_results=[self.solve_result_ambient],
            layer_thicknesses=[vis_thickness[0]])
        
        s_matrices_interior_after_source = fmmax.stack_s_matrices_interior(
            layer_solve_results=self.stack_layer_solve_results,
            layer_thicknesses=vis_thickness[1:])        

        source_amplitude_vector = jnp.zeros((2 * self.expansion.num_terms, 1),
                                             dtype=complex).at[self.incident_order_index, 0].set(1)

        incident_poynting_flux, _ = fmmax.amplitude_poynting_flux(
            forward_amplitude=source_amplitude_vector,
            backward_amplitude=jnp.zeros_like(source_amplitude_vector),
            layer_solve_result=self.solve_result_ambient,
        )

        source_amplitude_vector /= jnp.sqrt(jnp.sum(incident_poynting_flux, axis=-2, keepdims=True))

        # Create the scattering matrix of the whole device stack.
        scattering_matrix = fmmax.stack_s_matrix(
            layer_solve_results=self.stack_layer_solve_results,
            layer_thicknesses=self.thicknesses)

        backward_amplitude_at_stack_end = scattering_matrix.s21 @ source_amplitude_vector

        # Propagate the reflected field away from the grating.
        _, backward_amplitude_at_stack_start = fmmax.colocate_amplitudes(
            jnp.zeros_like(source_amplitude_vector),
            backward_amplitude_at_stack_end,
            z_offset=jnp.zeros(()),  # At the start of the ambient.
            layer_solve_result=self.solve_result_ambient,
            layer_thickness=self.thicknesses[0],
        )

        amplitudes_interior = fmmax.stack_amplitudes_interior_with_source(s_matrices_interior_before_source=s_matrices_interior_before_source,
                                                                          s_matrices_interior_after_source=s_matrices_interior_after_source,
                                                                          backward_amplitude_before_end=backward_amplitude_at_stack_start,
                                                                          forward_amplitude_after_start=source_amplitude_vector)

        vis_layer_znum = jnp.ceil(jnp.array(vis_thickness)/10).astype(int).tolist()

        (ex, ey, ez), (hx, hy, hz), (x_plot, y_plot, z_plot) = fmmax.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[self.solve_result_ambient] + self.stack_layer_solve_results,
            layer_thicknesses=vis_thickness,
            layer_znum=vis_layer_znum,
            x=jnp.linspace(0,self.total_size_x,self.x_nm.shape[0]),
            y=jnp.full(self.x_nm.shape[0],0.0))
        
        self.plot_efield = (ex, ey, ez)
        self.plot_hfield = (hx, hy, hz)
        self.plot_coords = (x_plot, y_plot, z_plot)
        self.visualize_thickness = vis_thickness

        _, reflected_flux = fmmax.amplitude_poynting_flux(
            forward_amplitude=source_amplitude_vector,
            backward_amplitude=backward_amplitude_at_stack_end,
            layer_solve_result=self.solve_result_ambient,
        )

        self.plot_fourier_order = self.expansion.basis_coefficients[:,0]
        num_fourier_terms = self.expansion.num_terms

        self.test_reflected_flux = reflected_flux
        if self.incident_pol == 'tm':
            self.plot_reflection_coeff = jnp.sqrt(jnp.abs(reflected_flux[0, num_fourier_terms:, 0])) * jnp.exp(
                1j * jnp.angle(backward_amplitude_at_stack_end[0, num_fourier_terms:, 0])
            )
        elif self.incident_pol == 'te':
            self.plot_reflection_coeff = jnp.sqrt(jnp.abs(reflected_flux[0, 0:num_fourier_terms, 0])) * jnp.exp(
                1j * jnp.angle(backward_amplitude_at_stack_end[0, 0:num_fourier_terms, 0])
            )

        self.response = jnp.sqrt(jnp.abs(reflected_flux[0, self.target_order_index, :])) * jnp.exp(
            1j * jnp.angle(backward_amplitude_at_stack_end[0, self.target_order_index, :])
        )


        pass

def plot_solved_grating_fieldmap(sgp):
    
    pol_ind = {'tm':0,'te':1}[sgp.incident_pol]
    pol_name = {'tm':'x','te':'y'}[sgp.incident_pol]


    x_plot, z_plot = jnp.meshgrid(sgp.plot_coords[0],sgp.plot_coords[2],indexing='ij')
    x_plot = x_plot - jnp.mean(sgp.plot_coords[0])

    z_plot = z_plot - (sgp.visualize_thickness[0]+sgp.visualize_thickness[1])
    grat_z_cen = sgp.grating_thickness/2
    # grat_z_cen = 1250

    real_plot = jnp.real(sgp.plot_efield[pol_ind][0,:,:,0])
    abs_plot = jnp.sqrt((jnp.abs(sgp.plot_efield[0][0,:,:,0])**2 + jnp.abs(sgp.plot_efield[1][0,:,:,0])**2 + 
                            jnp.abs(sgp.plot_efield[2][0,:,:,0])**2))
    
    real_plot_filt = real_plot[(jnp.abs(x_plot) <= sgp.grating_width/2 + 50)]
    real_plot_filt = real_plot[(jnp.abs(z_plot-grat_z_cen) <= sgp.grating_thickness/2 + 100)]

    abs_plot_filt = abs_plot[(jnp.abs(x_plot) <= sgp.grating_width/2 + 50)]
    abs_plot_filt = abs_plot[(jnp.abs(z_plot-grat_z_cen) <= sgp.grating_thickness/2 + 100)]
    

    
    with plt.style.context(['default', {'font.size': 7}]):
        figW = 5
        figH = figW/1.3
        fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
        axs.set_aspect('equal')
        # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
        # plt_kw = {'cmap':'RdBu','shading':'nearest','vmin':jnp.min(real_plot_filt),
        #           'vmax':jnp.max(real_plot_filt)}
        plt_kw = {'cmap':'RdBu','shading':'nearest','vmin':-1,
                  'vmax':1}

        plt_kw = {'cmap':'RdBu','shading':'nearest'}

        cm1 = axs.pcolormesh(x_plot,z_plot,real_plot,**plt_kw)

        axs.set_ylabel('z [nm]')
        axs.set_xlabel('x [nm]')
        # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

        cbar1 = fig.colorbar(cm1, pad=0.03, ax=axs,fraction=0.036, aspect=8.5)
        cbar1.ax.set_ylabel(f'Re[E{pol_name}]', rotation=270, labelpad=9)

        fig.tight_layout()
        plt.show()

    with plt.style.context(['default', {'font.size': 7}]):
        figW = 5
        figH = figW/1.3
        fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
        axs.set_aspect('equal')
        # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
        plt_kw = {'cmap':'turbo','shading':'nearest','vmin':0.0,
                  'vmax':jnp.max(abs_plot_filt)}

        cm1 = axs.pcolormesh(x_plot,z_plot,abs_plot,**plt_kw)

        axs.set_ylabel('z [nm]')
        axs.set_xlabel('x [nm]')
        # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

        cbar1 = fig.colorbar(cm1, pad=0.03, ax=axs,fraction=0.036, aspect=8.5)
        cbar1.ax.set_ylabel('|E|', rotation=270, labelpad=9)

        fig.tight_layout()
        plt.show()

    pass


def plot_solved_grating_farfield(sgp):
    
    plot_orders = sgp.plot_fourier_order
    reflected_phase = jnp.angle(sgp.plot_reflection_coeff)
    reflected_eff = jnp.abs(sgp.plot_reflection_coeff)**2

    with plt.style.context(['default', {'font.size': 7}]):
        figW = 5
        figH = figW/1.3
        fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
        # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
        plt_kw = {'lw':0.8}

        axs.stem(plot_orders,reflected_eff)

        axs.set_xlim([-10,10])

        axs.set_xlabel('Diffraction order along x')
        axs.set_ylabel('Relative intensity [a.u.]')
        # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

        fig.tight_layout()
        plt.show()

    with plt.style.context(['default', {'font.size': 7}]):
        figW = 5
        figH = figW/1.3
        fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
        # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
        plt_kw = {'lw':0.8}

        axs.stem(plot_orders,reflected_phase)

        axs.set_xlim([-10,10])

        axs.set_xlabel('Diffraction order along x')
        axs.set_ylabel('Phase [rad]')
        # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

        fig.tight_layout()
        plt.show()

    pass


#%%
if __name__ == "__main__":
    sgpgl = SimulateGratingPeriodicGradLayer()
    sgpgl.setup_sim_geom()

    # def test(density):
    #     farfield = sgpgl.response(density,8,0.5)
    #     return jnp.abs(jnp.abs(farfield[0])**2-1)

    # test_grad = jax.grad(test)
    # grad_test = test_grad(jnp.ones(sgpgl.num_optim_params))

    # opt_density, all_loss_history, all_param_history, epoch_data = sgp.optimize_density(np.random.rand(sgp.num_grating_mesh_points)*0.5)
    # opt_density, all_loss_history, all_param_history, epoch_data = sgp.optimize_density(jnp.ones(sgp.num_grating_mesh_points)*0.5)
    # sgpgl.solve_to_visualize_fields_interior_source(jnp.ones(sgp.num_grating_mesh_points),jnp.inf,0.5)

    # test_density = np.random.rand(sgp.num_grating_mesh_points)
    # sgp.solve_to_visualize_fields_interior_source(test_density,jnp.inf,0.5)
    # sgpgl.solve_to_visualize_fields_interior_source(np.full(sgpgl.num_optim_params,1.0),jnp.inf,0.5)

    # plot_solved_grating_fieldmap(sgpgl)
    # plot_solved_grating_farfield(sgpgl)


# #%%
# sgp.solve_to_visualize_fields_interior_source(opt_density,jnp.inf,0.5)
# #%%
# sgp = SimulateGratingPML()
# sgp.setup_sim_geom()
# # sgp.response(np.random.rand(sgp.num_grating_mesh_points),100,0.5)
# sgp.response(jnp.ones(sgp.num_grating_mesh_points),100,0.5)
# sgp.solve_to_visualize_fields()

# #%%
# sgp = SimulateGratingPML()
# sgp.setup_sim_geom()
# sgp.solve_to_visualize_fields_interior_source(jnp.ones(sgp.num_grating_mesh_points),100,0.5)
# #%%
# def test(density):
#     farfield = sgp.response(density,8,0.5)
#     return jnp.abs(jnp.abs(farfield[0])**2-1)

# test_grad = jax.grad(test)

# print(jnp.max(jnp.abs(test_grad(np.random.rand(sgp.num_grating_mesh_points)))))
# # print(jnp.max(test_grad(np.random.rand(sgp.num_grating_mesh_points))))
# # %%
# def plot_solved_grating_fieldmap(sgp):
    
#     pol_ind = {'tm':0,'te':1}[sgp.incident_pol]
#     pol_name = {'tm':'x','te':'y'}[sgp.incident_pol]


#     x_plot, z_plot = jnp.meshgrid(sgp.plot_coords[0],sgp.plot_coords[2],indexing='ij')
#     x_plot = x_plot - jnp.mean(sgp.plot_coords[0])

#     #z_plot = z_plot - (sgp.visulaize_thickness[0]+sgp.visulaize_thickness[1])
#     #grat_z_cen = sgp.grating_thickness/2
#     grat_z_cen = 1250

#     real_plot = jnp.real(sgp.plot_efield[pol_ind][0,:,:,0])
#     abs_plot = jnp.sqrt((jnp.abs(sgp.plot_efield[0][0,:,:,0])**2 + jnp.abs(sgp.plot_efield[1][0,:,:,0])**2 + 
#                             jnp.abs(sgp.plot_efield[2][0,:,:,0])**2))
    
#     real_plot_filt = real_plot[(jnp.abs(x_plot) <= sgp.grating_width/2 + 50)]
#     real_plot_filt = real_plot[(jnp.abs(z_plot-grat_z_cen) <= sgp.grating_thickness/2 + 100)]

#     abs_plot_filt = abs_plot[(jnp.abs(x_plot) <= sgp.grating_width/2 + 50)]
#     abs_plot_filt = abs_plot[(jnp.abs(z_plot-grat_z_cen) <= sgp.grating_thickness/2 + 100)]
    

    
#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         axs.set_aspect('equal')
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         # plt_kw = {'cmap':'RdBu','shading':'nearest','vmin':jnp.min(real_plot_filt),
#         #           'vmax':jnp.max(real_plot_filt)}
#         plt_kw = {'cmap':'RdBu','shading':'nearest','vmin':-1,
#                   'vmax':1}

#         cm1 = axs.pcolormesh(x_plot,z_plot,real_plot,**plt_kw)

#         axs.set_ylabel('z [nm]')
#         axs.set_xlabel('x [nm]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         cbar1 = fig.colorbar(cm1, pad=0.03, ax=axs,fraction=0.036, aspect=8.5)
#         cbar1.ax.set_ylabel(f'Re[E{pol_name}]', rotation=270, labelpad=9)

#         fig.tight_layout()
#         plt.show()

#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         axs.set_aspect('equal')
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         plt_kw = {'cmap':'turbo','shading':'nearest','vmin':0.0,
#                   'vmax':jnp.max(abs_plot_filt)}

#         cm1 = axs.pcolormesh(x_plot,z_plot,abs_plot,**plt_kw)

#         axs.set_ylabel('z [nm]')
#         axs.set_xlabel('x [nm]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         cbar1 = fig.colorbar(cm1, pad=0.03, ax=axs,fraction=0.036, aspect=8.5)
#         cbar1.ax.set_ylabel('|E|', rotation=270, labelpad=9)

#         fig.tight_layout()
#         plt.show()

#     pass
# plot_solved_grating_fieldmap(sgp)

# def plot_solved_grating_farfield(sgp):
#     theta = jnp.linspace(-90,90,181)
#     farfield = [0]*theta.size

#     x, z = sgp.plot_coords[0], sgp.plot_coords[2]
#     z_cut = 700
#     z_ind = jnp.argmin(jnp.abs(z - z_cut))
#     ex = sgp.plot_efield[0][0,:,z_ind,0]
#     ey = sgp.plot_efield[1][0,:,z_ind,0]

#     hx = sgp.plot_hfield[0][0,:,z_ind,0]
#     hy = sgp.plot_hfield[1][0,:,z_ind,0]

#     for i in range(theta.size):
#         farfield[i] = sgp.stratton_chu_farfield_2D(-1,jnp.deg2rad(theta[i]),x,ex,ey,hx,hy)[0]

#     farfield = jnp.asarray(farfield)

#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         plt_kw = {'lw':0.8}

#         axs.plot(theta,jnp.abs(farfield)**2/sgp.max_farfield_power,**plt_kw)

#         axs.set_xlabel('Farfield angle [°]')
#         axs.set_ylabel('Relative intensity [a.u.]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         fig.tight_layout()
#         plt.show()

#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         plt_kw = {'lw':0.8}

#         axs.plot(theta,jnp.angle(farfield),**plt_kw)

#         axs.set_xlabel('Farfield angle [°]')
#         axs.set_ylabel('Phase [rad]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         fig.tight_layout()
#         plt.show()

#     pass

# # %%
# def test_source_farfield(sgp):
#     theta = jnp.linspace(-90,90,181)
#     farfield = [0]*theta.size
#     # theta = 0

#     x = sgp.x_nm[:,0]
#     (ex,ey),(hx,hy) = sgp.in_plane_source_fields(sgp.incident_angle)

#     for i in range(theta.size):
#         farfield[i] = sgp.stratton_chu_farfield_2D(1,jnp.deg2rad(theta[i]),x,ex[:,:,0,0],ey[:,:,0,0],hx[:,:,0,0],hy[:,:,0,0])[0]

#     k = 2*np.pi/sgp.wavelengths
#     r = 1
#     farfield = jnp.asarray(farfield)
#     print('sum: ', jnp.sum(jnp.abs(farfield)**2))

#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         plt_kw = {'lw':0.8}

#         axs.plot(theta,jnp.abs(farfield)**2/sgp.max_farfield_power,**plt_kw)

#         axs.set_xlabel('Farfield angle [°]')
#         axs.set_ylabel('Intensity [a.u.]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         fig.tight_layout()
#         plt.show()

#     with plt.style.context(['default', {'font.size': 7}]):
#         figW = 5
#         figH = figW/1.3
#         fig, axs = plt.subplots(1, 1, dpi=600, figsize=(figW, figH))
#         # plt_kw = {'cmap':'viridis', 'norm': mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))}
#         plt_kw = {'lw':0.8}

#         axs.plot(theta,jnp.angle(farfield),**plt_kw)

#         axs.set_xlabel('Farfield angle [°]')
#         axs.set_ylabel('Phase [rad]')
#         # axs.set_title(f'{title}-pol, α = {alpha:.2f}')

#         fig.tight_layout()
#         plt.show()

# test_source_farfield(sgp)

# %%
