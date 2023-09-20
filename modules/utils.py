#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal

import torch
import torch.nn as nn
import math

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import numpy as np

def to_cartesian(points):
    theta, phi = points[..., 0], points[..., 1]

    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    return torch.stack([x, y, z], dim=-1)



class PosEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, num_frequencies=10):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        self.out_features = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[1], self.out_features)


class EucPosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)


def components_from_spherical_harmonics(levels, directions):
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical hamonic coefficients
    """
    num_components = levels**2
    components = torch.zeros(
        (*directions.shape[:-1], num_components), device=directions.device
    )

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert (
        directions.shape[-1] == 3
    ), f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.4425326924449826 * (
            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
        )

    return components


def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1))
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val


def psnr(x, xhat, weight):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2)*weight)

    snrval = 10*np.log10(np.max(x)/denom)

    return snrval

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = normalize(images[cnt, ...], True)
            cnt += 1
    
    return montage_im
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_coords(H, W, T=None):
    '''
        Get 2D/3D coordinates
    '''
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
    
    return torch.tensor(coords.astype(np.float32))

def resize(cube, scale):
    '''
        Resize a multi-channel image
        
        Inputs:
            cube: (H, W, nchan) image stack
            scale: Scaling 
    '''
    H, W, nchan = cube.shape
    
    im0_lr = cv2.resize(cube[..., 0], None, fx=scale, fy=scale)
    Hl, Wl = im0_lr.shape
    
    cube_lr = np.zeros((Hl, Wl, nchan), dtype=cube.dtype)
    
    for idx in range(nchan):
        cube_lr[..., idx] = cv2.resize(cube[..., idx], None,
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
    return cube_lr

def get_inpainting_mask(imsize, mask_type='random2d', mask_frac=0.5):
    '''
        Get a 2D mask for image inpainting
        
        Inputs:
            imsize: Image size
            mask_type: one of 'random2d', 'random1d'
            mask_frac: Fraction of non-zeros in the mask
            
        Outputs:
            mask: A 2D mask image
    '''
    H, W = imsize

    if mask_type == 'random2d':
        mask = np.random.rand(H, W) < mask_frac
    elif mask_type == 'random1d':
        mask_row = np.random.rand(1, W) < mask_frac
        mask = np.ones((H, 1)).dot(mask_row)
    elif mask_type == 'bayer':
        mask = np.zeros((H, W))
        mask[::2, ::2] = 1
        
    return mask.astype(np.float32)

@torch.no_grad()
def get_layer_outputs(model, coords, imsize,
                      nfilters_vis=16,
                      get_imag=False):
    '''
        get activation images after each layer
        
        Inputs:
            model: INR model
            coords: 2D coordinates
            imsize: Size of the image
            nfilters_vis: Number of filters to visualize
            get_imag: If True, get imaginary component of the outputs
            
        Outputs:
            atoms_montages: A list of 2d grid of outputs
    '''
    H, W = imsize

    if model.pos_encode:
        coords = model.positional_encoding(coords)
        
    atom_montages = []
    
    for idx in range(len(model.net)-1):
        layer_output = model.net[idx](coords)
        layer_images = layer_output.reshape(1, H, W, -1)[0]
        
        if nfilters_vis is not 'all':
            layer_images = layer_images[..., :nfilters_vis]
        
        if get_imag:
            atoms = layer_images.detach().cpu().numpy().imag
        else:
            atoms = layer_images.detach().cpu().numpy().real
            
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        signs = (abs(atoms_min) > abs(atoms_max))
        atoms = (1 - 2*signs)*atoms
        
        # Arrange them by variance
        atoms_std = atoms.std((0,1))
        std_indices = np.argsort(atoms_std)
        
        atoms = atoms[..., std_indices]
        
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        atoms = (atoms - atoms_min)/np.maximum(1e-14, atoms_max - atoms_min)
        
        atoms[:, [0, -1], :] = 1
        atoms[[0, -1], :, :] = 1
        
        atoms_montage = build_montage(np.transpose(atoms, [2, 0, 1]))
        
        atom_montages.append(atoms_montage)
        coords = layer_output
        
    return atom_montages
