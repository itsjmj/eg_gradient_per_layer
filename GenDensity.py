"""Density filtering and projection helpers for the inverse-design solvers.

The functions in this module map raw design variables into a smoothed density
field that preserves the existing numerical workflow used by the periodic and
PML-based optimizers.
"""
#%%
import pathlib
import sys
# basefp = str(pathlib.Path(__file__).parent.parent)
# sys.path.insert(0, basefp) 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import jax
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering, utils

def _pad_periodic(arr):
    # make sure on edge nodes are not repeaed twice
    return jnp.hstack([arr[0:-1],arr,arr[1:]])

def _pad_zero(arr):
    # length in keeping with periodic definition
    left_arr = jnp.full(arr.shape[0]-1,0.0)
    right_arr = jnp.full(arr.shape[0]-1,0.0)
    return jnp.hstack([left_arr,arr,right_arr])

def _inv_pad(pad_arr):
    orig_len = (pad_arr.shape[0] + 2)//3
    return pad_arr[orig_len-1:2*orig_len-1]

def _convolve_density_with_kernel(density,kernel):
    return jnp.real(jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.fft(density)*jnp.fft.fft(kernel))))

def _apply_filter_kernel(density, filter_width, dx, is_periodic, filter_type):
    """Apply the requested spatial kernel to a 1D density profile."""
    if is_periodic:
        pad_density = _pad_periodic(density)
    else:
        pad_density = _pad_zero(density)

    x_coords = jnp.linspace(
        -dx * (pad_density.shape[0] - 1) / 2,
        dx * (pad_density.shape[0] - 1) / 2,
        pad_density.shape[0],
    )

    radius = filter_width  # TODO: verify whether this should be divided by 2.
    if filter_type == 'conic':
        kernel = jnp.where(x_coords**2 <= radius**2, (1 - jnp.abs(x_coords / radius)), 0.0)
    else:
        kernel = jnp.exp(-x_coords**2 / radius**2)
    kernel = kernel / jnp.sum(kernel)

    filt_density = _convolve_density_with_kernel(pad_density, kernel)
    return _inv_pad(filt_density)


def _apply_conic_filter(density, filter_width, dx, is_periodic):
    """Apply the conic smoothing filter to a density profile."""
    return _apply_filter_kernel(density, filter_width, dx, is_periodic, filter_type='conic')


def _apply_gaussian_filter(density, filter_width, dx, is_periodic):
    """Apply the Gaussian smoothing filter to a density profile."""
    return _apply_filter_kernel(density, filter_width, dx, is_periodic, filter_type='gauss')

def _tanh_projection(density, beta, eta):
    if beta == jnp.inf:
        # Note that backpropagating through here can produce NaNs. So we
        # manually specify the step function to keep the gradient clean.
        return jnp.where(density > eta, 1.0, 0.0)
    else:
        return (jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))) / (
                jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
    
# def _tanh_projection(density, beta, eta):
#     # Use jnp.where to avoid boolean conversion of traced values
#     is_infinite = jnp.isinf(jnp.full_like(density,beta))
#     step_result = jnp.where(density > eta, 1.0, 0.0)
#     tanh_result = (jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))) / (
#             jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
#     return jnp.where(is_infinite, step_result, tanh_result)

def _smoothed_projection(rho_filt, beta, eta, dx):
    R_smoothing = 0.55*dx
    rho_proj = _tanh_projection(rho_filt,beta,eta)
    rho_filt_grad = jnp.gradient(rho_filt)
    rho_filt_grad_norm2 = (rho_filt_grad/dx)**2 # for 2d + (rho_filt_grad[1]/dy)**2
    nonzero_norm = jnp.abs(rho_filt_grad_norm2) > 0
    rho_filt_grad_norm = jnp.sqrt(jnp.where(nonzero_norm,rho_filt_grad_norm2,1))
    rho_filt_grad_norm_eff = jnp.where(nonzero_norm,rho_filt_grad_norm,1)
    d = (eta-rho_filt) / rho_filt_grad_norm_eff
    needs_smoothing = nonzero_norm & (jnp.abs(d) < R_smoothing)
    d_R = d / R_smoothing
    F = jnp.where(needs_smoothing,
                    0.5 - 15/16*d_R + 5/8*d_R**3 - 3/16*d_R**5,
                    1.0)
    F_minus = jnp.where(needs_smoothing,
                    0.5 + 15/16*d_R - 5/8*d_R**3 + 3/16*d_R**5,
                    1.0)
    rho_filt_minus = rho_filt - R_smoothing*rho_filt_grad_norm_eff*F
    rho_filt_plus = rho_filt + R_smoothing*rho_filt_grad_norm_eff*F_minus
    rho_minus_eff_proj = _tanh_projection(rho_filt_minus,beta,eta)
    rho_plus_eff_proj = _tanh_projection(rho_filt_plus,beta,eta)
    rhp_proj_smoothed = (1-F) * rho_minus_eff_proj +F*rho_plus_eff_proj
    return jnp.where(needs_smoothing,rhp_proj_smoothed,rho_proj)

def _generate_filt_density(static_state,raw_density,beta,eta):
    dx = static_state['dx']
    is_periodic = static_state['is_periodic']
    filter_width = static_state['filter_width']
    filter_type = static_state['filter_type']

    if filter_type == 'conic':
        filt_density = _apply_conic_filter(raw_density,filter_width,dx,is_periodic)
    else:
        filt_density = _apply_gaussian_filter(raw_density,filter_width,dx,is_periodic)
    ssp_density = _smoothed_projection(filt_density,beta,eta,dx)
    return ssp_density

class GenDensity1D:

    def __init__(self,dx=2,is_periodic=False,filter_width=50,filter_type='gauss'):
        self.dx = dx
        self.is_periodic = is_periodic
        self.filter_width = filter_width
        self.filter_type = filter_type
        pass

    def pad_periodic(self,arr):
        # make sure on edge nodes are not repeaed twice
        return jnp.hstack([arr[0:-1],arr,arr[1:]])
    
    def pad_zero(self,arr):
        # length in keeping with periodic definition
        left_arr = jnp.full(arr.shape[0]-1,0.0)
        right_arr = jnp.full(arr.shape[0]-1,0.0)
        return jnp.hstack([left_arr,arr,right_arr])
    
    def inv_pad(self,pad_arr):
        orig_len = (pad_arr.shape[0] + 2)//3
        return pad_arr[orig_len-1:2*orig_len-1]
    
    # steps are pad array, apply filter, apply tanh binarisation filter, apply ssp filter
    # to do the spatial filter we need to know the design region resolution
    # this code should focus solely on manipulating the material desnity, other metarial operations will happen outside
    
    def convolve_density_with_kernel(self,density,kernel):
        return jnp.real(jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.fft(density)*jnp.fft.fft(kernel))))

    def apply_conic_filter(self, density):
        """Apply the conic smoothing filter using the class settings."""
        return _apply_filter_kernel(
            density=density,
            filter_width=self.filter_width,
            dx=self.dx,
            is_periodic=self.is_periodic,
            filter_type='conic',
        )

    def apply_gaussian_filter(self, density):
        """Apply the Gaussian smoothing filter using the class settings."""
        return _apply_filter_kernel(
            density=density,
            filter_width=self.filter_width,
            dx=self.dx,
            is_periodic=self.is_periodic,
            filter_type='gauss',
        )
    
    def tanh_projection(self,density, beta, eta):
        if beta == jnp.inf:
            # Note that backpropagating through here can produce NaNs. So we
            # manually specify the step function to keep the gradient clean.
            return jnp.where(density > eta, 1.0, 0.0)
        else:
            return (jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))) / (
                    jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
        
    # def tanh_projection(self, density, beta, eta):
    #     # Use jnp.where to avoid boolean conversion of traced values
    #     is_infinite = jnp.isinf(jnp.full_like(density,beta))
    #     step_result = jnp.where(density > eta, 1.0, 0.0)
    #     tanh_result = (jnp.tanh(beta * eta) + jnp.tanh(beta * (density - eta))) / (
    #             jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
    #     return jnp.where(is_infinite, step_result, tanh_result)
    
    def smoothed_projection(self,rho_filt, beta, eta):
        R_smoothing = 0.55*self.dx
        rho_proj = self.tanh_projection(rho_filt,beta,eta)
        rho_filt_grad = jnp.gradient(rho_filt)
        rho_filt_grad_norm2 = (rho_filt_grad/self.dx)**2 # for 2d + (rho_filt_grad[1]/dy)**2
        nonzero_norm = jnp.abs(rho_filt_grad_norm2) > 0
        rho_filt_grad_norm = jnp.sqrt(jnp.where(nonzero_norm,rho_filt_grad_norm2,1))
        rho_filt_grad_norm_eff = jnp.where(nonzero_norm,rho_filt_grad_norm,1)
        d = (eta-rho_filt) / rho_filt_grad_norm_eff
        needs_smoothing = nonzero_norm & (jnp.abs(d) < R_smoothing)
        d_R = d / R_smoothing
        F = jnp.where(needs_smoothing,
                      0.5 - 15/16*d_R + 5/8*d_R**3 - 3/16*d_R**5,
                      1.0)
        F_minus = jnp.where(needs_smoothing,
                      0.5 + 15/16*d_R - 5/8*d_R**3 + 3/16*d_R**5,
                      1.0)
        rho_filt_minus = rho_filt - R_smoothing*rho_filt_grad_norm_eff*F
        rho_filt_plus = rho_filt + R_smoothing*rho_filt_grad_norm_eff*F_minus
        rho_minus_eff_proj = self.tanh_projection(rho_filt_minus,beta,eta)
        rho_plus_eff_proj = self.tanh_projection(rho_filt_plus,beta,eta)
        rhp_proj_smoothed = (1-F) * rho_minus_eff_proj +F*rho_plus_eff_proj
        return jnp.where(needs_smoothing,rhp_proj_smoothed,rho_proj)
    

    # def generate_filt_density(self,raw_density,beta,eta):
    #     if self.filter_type == 'conic':
    #         filt_density = self.apply_conic_filter(raw_density)
    #     else:
    #         filt_density = self.apply_gaussian_filter(raw_density)
    #     ssp_density = self.smoothed_projection(filt_density,beta,eta)
    #     return ssp_density

    def generate_filt_density(self, raw_density, beta, eta):
        """Return the filtered and projected density for the supplied design field."""
        static_state = self.__dict__
        return _generate_filt_density(static_state, raw_density, beta, eta)

    def plot_density_1d(self,density,layer_h,title,save_fp=None):
        x = jnp.linspace(-self.dx*(density.shape[0]-1)/2,self.dx*(density.shape[0]-1)/2,density.shape[0])
        y = jnp.linspace(0,layer_h,int(layer_h/self.dx)+1)
        X,Y = jnp.meshgrid(x,y,indexing='ij')
        density_2d = jnp.tile(density[:,jnp.newaxis],(1,int(layer_h/self.dx)+1))

        with plt.style.context(['default', {'font.size': 7}]):
            figW = 5
            figH = figW
            fig, axs = plt.subplots(1, 1, dpi=300, figsize=(figW, figH))
            axs.set_aspect('equal')
            plt_kw = {'cmap':'binary', 'vmin':0, 'vmax':1, 'shading':'nearest'}
            # plt_kw = {'cmap':'binary', 'shading':'nearest'}

            cm1 = axs.pcolormesh(X,Y,density_2d,**plt_kw)

            axs.set_xlabel('x [nm]')
            axs.set_ylabel('y [nm]')
            axs.set_title(title)

            # control the width of colorbar in the figure

            cbar1 = fig.colorbar(cm1, pad=0.03, fraction=0.02, aspect=10)
            cbar1.ax.set_ylabel('Material density', rotation=270, labelpad=9)
            fig.tight_layout()

            if save_fp is None:
                plt.show()
                plt.clf()
                plt.close()
            else:
                fig.savefig(save_fp,dpi=300,bbox_inches='tight')
                plt.clf()
                plt.close()
            
        return fig
    
    def plot_gradient_1d(self,grad,layer_h,title,save_fp=None):
        x = jnp.linspace(-self.dx*(grad.shape[0]-1)/2,self.dx*(grad.shape[0]-1)/2,grad.shape[0])
        y = jnp.linspace(0,layer_h,int(layer_h/self.dx)+1)
        X,Y = jnp.meshgrid(x,y,indexing='ij')
        density_2d = jnp.tile(grad[:,jnp.newaxis],(1,int(layer_h/self.dx)+1))

        with plt.style.context(['default', {'font.size': 7}]):
            figW = 5
            figH = figW
            fig, axs = plt.subplots(1, 1, dpi=300, figsize=(figW, figH))
            axs.set_aspect('equal')
            plt_kw = {'cmap':'turbo', 'shading':'nearest'}
            # plt_kw = {'cmap':'binary', 'shading':'nearest'}

            cm1 = axs.pcolormesh(X,Y,density_2d,**plt_kw)

            axs.set_xlabel('x [nm]')
            axs.set_ylabel('y [nm]')
            axs.set_title(title)

            # control the width of colorbar in the figure

            cbar1 = fig.colorbar(cm1, pad=0.03, fraction=0.02, aspect=10)
            cbar1.ax.set_ylabel('Gradient value', rotation=270, labelpad=9)
            fig.tight_layout()

            if save_fp is None:
                plt.show()
                plt.clf()
                plt.close()
            else:
                fig.savefig(save_fp,dpi=300,bbox_inches='tight')
                plt.clf()
                plt.close()
            
        return fig

def check_gen_density_gradients(gd, raw_density, beta, eta,
                                eps=1e-6, ncheck=None):
    """
    finite‑difference check for `gd.generate_filt_density`.

    Args
    ----
    gd : GenDensity1D instance
    raw_density : 1‑D jnp.array of design variables
    beta, eta : filter parameters (scalars or jnp arrays)
    eps : FD step size
    ncheck : if int, number of random components to test;
             if None, test them all (slower).

    Returns
    -------
    grad_jax : jnp.ndarray
    grad_fd : jnp.ndarray
    rel_err : jnp.ndarray   relative error per component
    """
    import numpy as _np

    def loss(rho):
        # any scalar function of the filtered density will do
        filt = gd.generate_filt_density(rho, beta, eta)
        return jnp.sum(filt**2)

    # JAX gradient
    grad_jax = jax.grad(loss)(raw_density)

    # finite‑difference gradient
    N = raw_density.shape[0]
    grad_fd = jnp.zeros_like(raw_density)

    idxs = list(range(N)) if ncheck is None else _np.random.choice(N, ncheck, replace=False)
    for i in idxs:
        ei = jnp.zeros_like(raw_density).at[i].set(1.0)
        lp = loss(raw_density + eps*ei)
        lm = loss(raw_density - eps*ei)
        grad_fd = grad_fd.at[i].set((lp - lm) / (2*eps))

    # relative error
    denom = jnp.abs(grad_fd) + jnp.abs(grad_jax) + 1e-14
    rel_err = jnp.abs(grad_fd - grad_jax) / denom

    print(f"checked {len(idxs)} components; "
          f"max rel.err {float(jnp.max(rel_err)):.3e}, "
          f"mean rel.err {float(jnp.mean(rel_err)):.3e}")

    return grad_jax, grad_fd, rel_err

# if __name__ == "__main__":
#     # gd = GenDensity1D(dx=2, is_periodic=True, filter_width=50, filter_type='gauss')
#     # rho0 = np.random.rand(100)
#     # grad_j, grad_f, rel = check_gen_density_gradients(gd, rho0,
#     #                                                     beta=8.0, eta=0.5,
#     #                                                     eps=1e-6, ncheck=20)
#     dx = 2
#     sx = 1500
#     filt_width = 100
#     beta = jnp.inf
#     eta = 0.5
#     x = jnp.linspace(-sx/2,sx/2,int(sx/dx)+1)
#     # raw_density = jnp.where(jnp.abs(x-200)<=40,1.0,0.0)
#     raw_density = np.random.rand(int(sx/dx)+1)
#     gd = GenDensity1D(dx=dx,is_periodic=True,filter_width=filt_width,filter_type='gauss')
#     filt_density = gd.generate_filt_density(raw_density,beta,eta)
#     gd.plot_density_1d(filt_density,300,f'Beta = {beta:.0f}, Eta = {eta:.1f}')

#     @jax.jit
#     def test_sine_diff(raw_density):
#         x = jnp.linspace(-sx/2,sx/2,int(sx/dx)+1)
#         gd = GenDensity1D(dx=dx,is_periodic=False,filter_width=filt_width,filter_type='gauss')
#         filt_density = gd.generate_filt_density(raw_density,beta,eta)
#         sine_diff = jnp.sum(jnp.abs(filt_density-jnp.sin(x)**2))
#         return sine_diff
    
#     test_sine_diff_grad = jax.grad(test_sine_diff)
#     print(test_sine_diff_grad(raw_density))

    
# %%
