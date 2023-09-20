#!/usr/bin/env python
from argparse import ArgumentParser
import os
import sys
from tqdm import tqdm
import importlib
import time

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim_func

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
os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0

from modules import gauss, mfn, relu, siren, wire, wire2d, swinr, shinr
model_dict = {'gauss': gauss,
              'mfn': mfn,
              'relu': relu,
              'siren': siren,
              'wire': wire,
              'wire2d': wire2d,
              'swinr':swinr,
              'shinr':shinr}

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
    parser.add_argument("--dataset_dir", type=str, default='./dataset/sun360')
    parser.add_argument("--model", type=str, default="relu")

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
    wandb.init(project='final_denoising', config=args, name=str(args.model))
    
    
    nonlin = args.model            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = args.niters               # Number of SGD iterations
    learning_rate = args.lr        # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    
    tau = args.tau                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = args.snr              # Readout noise (dB)
    
    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = args.omega           # Frequency of sinusoid
    sigma0 = args.sigma           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = args.hidden_layers       # Number of hidden layers in the MLP
    hidden_features = args.hidden_features   # Number of hidden units per layer
    maxpoints = args.batch_size    # Batch size
    
    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(plt.imread(args.dataset_dir+'/'+str(args.panorama_idx)+'.jpg').astype(np.float32), True)
    # im = cv2.resize(im, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    
    # Create a noisy image
    im_noisy = utils.measure(im, noise_snr, tau)
    # saveimg = Image.fromarray(((im_noisy-im_noisy.min())/(im_noisy.max()-im_noisy.min())*255).astype(np.uint8), 'RGB')
    # saveimg.save('noisy_'+str(tau)+'_'+str(noise_snr)+'_'+str(utils.psnr(im, im_noisy))+'.png')

    
    if nonlin == 'relu':
        posencode = True
        
        if tau < 100:
            sidelength = int(max(H, W)/3)
        else:
            sidelength = int(max(H, W))
            
    else:
        posencode = False
        sidelength = H

        
    model = model_dict[args.model].INR(**vars(args))
        
    # Send model to CUDA
    model.cuda()
    
    
    parameters = model.get_optimizer_parameters(weight_decay=0.01)
    optim = torch.optim.Adam(params = parameters, 
                             lr=learning_rate * min(1, maxpoints / (H*W)))  # This will apply weight decay only to the last two layers
   
    # Create an optimizer
    # optim = torch.optim.Adam(lr=learning_rate*min(1, maxpoints/(H*W)),
    #                          params=model.parameters())
    
    # Schedule to reduce lr to 0.1 times the initial rate in final epoch
    scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat, lon  = np.deg2rad(lat), np.deg2rad(lon)
    lat, lon = torch.meshgrid(lat, lon)
    lat = lat.flatten()
    lon = lon.flatten()
    coords = torch.stack([lat, lon], dim=-1)[None,...]
    
    mean_lat_weight = torch.cos(lat).mean().cuda()
    weight = torch.cos(lat).cuda()
    weight = weight / mean_lat_weight
       
    print('Number of parameters: ', utils.count_parameters(model))
    print('Input PSNR: %.2f dB'%utils.psnr(im, im_noisy, weight.reshape(H,W).unsqueeze(-1).detach().cpu().numpy())) 
    
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
            b_lat = b_coords[...,0].cuda()
            b_weight = torch.cos(b_lat) / mean_lat_weight
            b_indices = b_indices.cuda()
            
            b_coords = to_cartesian(b_coords)
            pixelvalues = model(b_coords) # [1, 65536, 3]
            
            with torch.no_grad():
                rec[:, b_indices, :] = pixelvalues
    
            loss = ((pixelvalues - gt_noisy[:, b_indices, :])**2 * b_weight.unsqueeze(-1)).mean() 
            gt_loss = ((pixelvalues - gt[:, b_indices, :])**2 * b_weight.unsqueeze(-1)).mean() 
            batch_psnr =  -10*torch.log10(loss)
            gt_batch_psnr =  -10*torch.log10(gt_loss)
            batch_psnr_array.append(float(batch_psnr.detach().cpu().numpy()))
            batch_mse_array.append(float(loss.detach().cpu().numpy()))
            
            wandb.log({'noisy_batch_mse':loss})
            wandb.log({'noisy_batch_psnr':batch_psnr})
            wandb.log({'gt_batch_mse':gt_loss})
            wandb.log({'gt_batch_psnr':gt_batch_psnr})
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        time_array[epoch] = time.time() - init_time
        
        with torch.no_grad():
            noisy_mse = ((gt_noisy - rec)**2 * weight.unsqueeze(-1)).mean().item()
            gt_mse =  ((gt - rec)**2 * weight.unsqueeze(-1)).mean().item()
            
            mse_loss_array[epoch] = noisy_mse
            mse_array[epoch] =gt_mse
            
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
            
    
        if (mse_array[epoch] < best_mse) or (epoch == 0):
            best_mse = mse_array[epoch]
            best_img = imrec
            # saveimg = Image.fromarray(((best_img-best_img.min())/(best_img.max()-best_img.min())*255).astype(np.uint8), 'RGB')
            # saveimg.save('./best_images/best_img'+str(epoch)+'.png')

            os.makedirs('./checkpoints/',exist_ok=True)
            save_checkpoint(model, optim, epoch, best_mse, filename=f"./checkpoints/checkpoint_{nonlin}_{wandb.run.id}.pth")
    
    best_psnr = utils.psnr(im, best_img, weight.reshape(H,W).unsqueeze(-1).detach().cpu().numpy())
    print('Best PSNR: %.2f dB'%best_psnr)
    wandb.log({'best_psnr':best_psnr})
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
