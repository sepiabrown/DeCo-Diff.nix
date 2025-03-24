
import os
import torch
from skimage.transform import resize
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import argparse
from PIL import Image
import numpy as np
from IPython.display import display
torch.set_grad_enabled(False)
import torch.nn.functional as F
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from glob import glob

from torch.utils.data import DataLoader
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from scipy.ndimage import gaussian_filter

from anomalib import metrics
from sklearn.metrics import roc_auc_score, average_precision_score





from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc




def calculate_metrics(ground_truth, prediction):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    

    auprc = metrics.AUPR()
    auprc_score = auprc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    # aupro_score = 0
    aupro = metrics.AUPRO(fpr_limit=0.3)
    aupro_score = compute_pro(ground_truth, prediction)
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    gt_list_sp = []
    pr_list_sp = []
    for idx in range(len(ground_truth)):
        gt_list_sp.append(np.max(ground_truth[idx]))
        sp_score = np.max(prediction[idx])
        pr_list_sp.append(sp_score)

    gt_list_sp = np.array(gt_list_sp).astype(np.int32)
    pr_list_sp = np.array(pr_list_sp)

    apsp = average_precision_score(gt_list_sp, pr_list_sp)
    aurocsp = auroc(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    f1sp = f1max(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    
    return auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp




def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask
    

def evaluate(x0_s, segmentation_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256):
        pr = []
        pr2 = []
        gt = []
        image_differences = []
        latent_differences = []
        input_images = []
        output_images = []
        for x, segmentation, encoded,  image_samples, latent_samples in zip(x0_s, segmentation_s, encoded_s,  image_samples_s, latent_samples_s):
                
                input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
                output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
                input_images.append(input_image)
                output_images.append(output_image)

                # image_difference = ((((1-out_image_mask)*((torch.abs(image_samples-x))).to(torch.float32)).sum(axis=0)/((1-out_image_mask).sum(axis=0))).detach().cpu().numpy().transpose(1,2,0).sum(axis=2))
                image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
                image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
                image_difference = smooth_mask(image_difference, sigma=1)
                image_differences.append(image_difference)
                
                latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
                latent_difference = (np.clip(latent_difference, 0.0 , 0.4)) * 2.5
                latent_difference = smooth_mask(latent_difference, sigma=1)
                latent_difference = resize(latent_difference, (center_size, center_size))
                latent_differences.append(latent_difference)
                
                final_anomaly = image_difference * latent_difference
                final_anomaly = np.sqrt(final_anomaly)
                final_anomaly = smooth_mask(final_anomaly, sigma=1)
                final_anomaly2 = 1/2*image_difference + 1/2*latent_difference
                final_anomaly2 = smooth_mask(final_anomaly2, sigma=1)
                pr.append(final_anomaly)
                pr2.append(final_anomaly2)

                gt.append(segmentation[0,:,:].cpu().numpy())
        pr = np.stack(pr, axis=0)
        pr2 = np.stack(pr2, axis=0)
        latent_differences = np.stack(latent_differences, axis=0)
        image_differences = np.stack(image_differences, axis=0)
        gt = np.stack(gt, axis=0)
        gt = (gt>0).astype(np.int32)

        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(gt, pr)
        print(f'4-{t}-final_anomaly_geometric: ', {'auroc':auroc_score, 'aupro':np.round(aupro_score, 4),'f1_max':f1_max_score, 'ap':np.round(ap,4), 'aurocsp':aurocsp, 'apsp':apsp, 'f1sp':f1sp})
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(gt, pr2)
        print(f'4-{t}-final_anomaly_arithmeti: ', {'auroc':auroc_score, 'aupro':np.round(aupro_score, 4),'f1_max':f1_max_score, 'ap':np.round(ap,4), 'aurocsp':aurocsp, 'apsp':apsp, 'f1sp':f1sp})
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(gt, image_differences)
        print(f'4-{t}-image_differences: ', {'auroc':auroc_score, 'aupro':np.round(aupro_score, 4),'f1_max':f1_max_score, 'ap':np.round(ap,4), 'aurocsp':aurocsp, 'apsp':apsp, 'f1sp':f1sp})
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(gt, latent_differences)
        print(f'4-{t}-latent_differences: ', {'auroc':auroc_score, 'aupro':np.round(aupro_score, 4),'f1_max':f1_max_score, 'ap':np.round(ap,4), 'aurocsp':aurocsp, 'apsp':apsp, 'f1sp':f1sp})




def evaluation(args):

    vae_model = f"stabilityai/sd-vae-ft-{args.vae}" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()

    try:
        
        if args.pretrained != ''
            ckpt = args.pretrained
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    
            try:
                ckpt = sorted(glob(f'{path}/last.pt'))[-1]
            except:
                ckpt = sorted(glob(f'{path}/*/last.pt'))[-1]
    except:
        raise Exception("Please provide the model's pretrained path using --pretrained")
    

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size)
    
    state_dict = torch.load(ckpt)['model']
    print(model.load_state_dict(state_dict))
    model.eval() # important!
    model.cuda()
    print('model loaded')


    for category in args.categories:


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object:
        diffusion = create_diffusion(f'ddim5', predict_deviation=True, sigma_small=False, predict_xstart=False, diffusion_steps=10)
            
        
        x_s = []
        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        segmentation_s = []
        
        
        
        test_dataset = MVTECDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=input_size, center_size=center_size, center_crop=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
        
        for ii, (x, y, object_cls) in enumerate(test_loader):
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                encoded = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)
                model_kwargs = {
                'context':object_cls.to(device).unsqueeze(1),
                'mask': None
                }
                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model, encoded.shape, noise = encoded, clip_denoised=False, 
                    start_t = args.reverse_steps,
                    model_kwargs=model_kwargs, progress=False, device=device,
                    eta = 0
                )



                image_samples = vae.decode(latent_samples / 0.18215).sample 
                x0 = vae.decode(encoded / 0.18215).sample 


            x_s += [_x.unsqueeze(0) for _x in x]
            segmentation_s += [_y.unsqueeze(0) for _y in y]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]

        print(args.center_size, category)
        evaluate(x0_s, segmentation_s, encoded_s,  image_samples_s, latent_samples_s, center_size=args.center_size)
            

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default= 288 )
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=bool, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--pretrained", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", , type=int, default=5)
    
    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    args.results_dir = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size

    if args.object_category=='all' and args.dataset=='mvtec':
        args.categories=[
            "bottle",
            "cable",
            "capsule",
            "hazelnut",
            "metal_nut",
            "pill",
            "screw",
            "toothbrush",
            "transistor",
            "zipper",

            "carpet",
            "grid",
            "leather",
            "tile",
            "wood",
            ]
    elif args.object_category=='all' and args.dataset=='visa':
        args.categories=[
            "candle",
            "cashew",
            "fryum",
            "macaroni2",
            "pcb2",
            "pcb4",
            "capsules",
            "chewinggum",
            "macaroni1",
            "pcb1",
            "pcb3",
            "pipe_fryum"
            ]
    else:
        args.categories = [args.object_category]
        
    evaluation(args)
