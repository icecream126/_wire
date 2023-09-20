#!/usr/bin/env python
from argparse import ArgumentParser
import os
import sys
from tqdm import tqdm
import importlib
import time

import numpy as np
from scipy import io

import matplotlib.pyplot as plt
plt.gray()

import cv2
# from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
# from pytorch_msssim import ssim
from PIL import Image

from modules import models
from modules import utils
from modules.utils import to_cartesian
import os
import wandb
from modules import euc_relu

os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0
os.environ['CUDA_VISIBLE_DEVICES']='5'
model_dict = {'euc_relu': euc_relu}

def save_checkpoint(model, optimizer, epoch, best_mse, filename="checkpoint.pth"):
    """
    Save model checkpoint.
    """
    state = {
        'epoch': epoch,
        'best_mse': best_mse,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='./data/sun360')
    parser.add_argument("--nonlin", type=str, default="euc_relu")

    # Dataset argument
    parser.add_argument("--panorama_idx", type=int, default=1)
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--tau", type=float, default=70)
    parser.add_argument("--snr", type=float, default=1)

    # Model argument
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--skip", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--posenc_freq", type=int, default=10)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=256*256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=1000)
    parser.add_argument("--niters",type=int, default=3000)
    parser.add_argument("--patience",type=int, default=50)

    parser.add_argument("--project_name", type=str, default="fair_denoising")
    
    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()
    args.in_features = 3
    args.out_features = 3
    wandb.init(project='_denoising', config=args, name=str(args.nonlin))
    
    
    nonlin = 'posenc'            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 3000               # Number of SGD iterations
    learning_rate = 1e-3        # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 4.0           # Frequency of sinusoid
    sigma0 = 4.0           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = 2       # Number of hidden layers in the MLP
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = 256*256     # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(plt.imread(args.dataset_dir+'/'+str(args.panorama_idx)+'.jpg').astype(np.float32), True)
    # im = cv2.resize(im, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    
    # Create a noisy image
    im_noisy = utils.measure(im, noise_snr, tau)
    
    posencode=True
    if tau < 100:
        sidelength = int(max(H, W)/3)
    else:
        sidelength = int(max(H, W))

        
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=3, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=sidelength)
        
    # Send model to CUDA
    model.cuda()
    
    print('Number of parameters: ', utils.count_parameters(model))
    print('Input PSNR: %.2f dB'%utils.psnr(im, im_noisy))
    
    
    parameters = model.get_optimizer_parameters(weight_decay=0.01)
    optim = torch.optim.Adam(parameters, lr=0.001)  # This will apply weight decay only to the last two layers
    # Create an optimizer
    # optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(H*W)),
    #                          params=model.parameters())
    
    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...] # [1, 65536, 2]
    
    gt = torch.tensor(im).cuda().reshape(H*W, 3)[None, ...]
    gt_noisy = torch.tensor(im_noisy).cuda().reshape(H*W, 3)[None, ...]
    
    mse_array = torch.zeros(niters, device='cuda')
    mse_loss_array = torch.zeros(niters, device='cuda')
    time_array = torch.zeros_like(mse_array)
    
    best_mse = torch.tensor(float('inf'))
    best_img = None
    
    rec = torch.zeros_like(gt)
    
    tbar = tqdm(range(niters))
    init_time = time.time()
    for epoch in tbar:
        indices = torch.randperm(H*W)
        batch_psnr_array = []
        batch_mse_array = []
        
        for b_idx in range(0, H*W, maxpoints):
            b_indices = indices[b_idx:min(H*W, b_idx+maxpoints)] # [65536]
            b_coords = coords[:, b_indices, ...].cuda() # [1, 65536, 2]
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords) # [1, 65536, 3]
            
            with torch.no_grad():
                rec[:, b_indices, :] = pixelvalues
    
            loss = ((pixelvalues - gt_noisy[:, b_indices, :])**2).mean() 
            batch_psnr =  -10*torch.log10(loss)
            batch_psnr_array.append(float(batch_psnr.detach().cpu().numpy()))
            batch_mse_array.append(float(loss.detach().cpu().numpy()))
            
            wandb.log({'noisy_batch_mse':loss})
            wandb.log({'noisy_batch_psnr':batch_psnr})
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        time_array[epoch] = time.time() - init_time
        
        with torch.no_grad():
            noisy_mse = ((gt_noisy - rec)**2 ).mean().item()
            gt_mse =  ((gt - rec)**2).mean().item()
            
            mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
            mse_array[epoch] = ((gt - rec)**2).mean().item()
                        
            wandb.log({'noisy_all_mse':noisy_mse})
            wandb.log({'gt_all_mse':gt_mse})
            
            im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            
            psnrval = -10*torch.log10(mse_array[epoch])
            noisy_psnrval = -10*torch.log10(mse_loss_array[epoch])
            wandb.log({'noisy_all_psnr':noisy_psnrval})
            wandb.log({'gt_all_psnr':psnrval})
            
            tbar.set_description('%.1f'%psnrval)
            tbar.refresh()
        
        avg_batch_psnr = np.mean(batch_psnr_array)
        avg_batch_mse = np.mean(batch_mse_array)
        wandb.log({'avg_noisy_batch_psnr':avg_batch_psnr})
        wandb.log({'avg_noisy_batch_mse':avg_batch_mse})
        scheduler.step()
        
        imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()
            
        cv2.imshow('Reconstruction', imrec[..., ::-1])            
        cv2.waitKey(1)
    
        if (mse_array[epoch] < best_mse) or (epoch == 0):
            best_mse = mse_array[epoch]
            best_img = imrec
            os.makedirs('./checkpoints/',exist_ok=True)
            save_checkpoint(model, optim, epoch, best_mse, filename=f"./checkpoints/checkpoint_{nonlin}_{wandb.run.id}.pth")
    
    if posencode:
        nonlin = 'posenc'
    best_img = Image.fromarray(((best_img-best_img.min())/(best_img.max()-best_img.min())*255).astype(np.uint8), 'RGB')
    wandb.log({"Prediction" :wandb.Image(best_img, caption=f"prediction")})
    wandb.log({"Ground Truth" :wandb.Image(im, caption=f"ground truth")})
    wandb.log({"Noisy Truth" :wandb.Image(im_noisy, caption=f"noisy truth")})
    wandb.save(f"checkpoint_{nonlin}_{wandb.run.id}.pth")
        
    mdict = {'rec': best_img,
             'gt': im,
             'im_noisy': im_noisy,
             'mse_noisy_array': mse_loss_array.detach().cpu().numpy(), 
             'mse_array': mse_array.detach().cpu().numpy(),
             'time_array': time_array.detach().cpu().numpy()}
    
    os.makedirs('results/denoising', exist_ok=True)
    # io.savemat('results/denoising/%s.mat'%nonlin, mdict)

    best_psnr = utils.psnr(im, best_img)
    print('Best PSNR: %.2f dB'%best_psnr)