import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from glob import glob

from torch.utils.data import DataLoader
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from PCBDataLoader import PCBDataset
from scipy.ndimage import gaussian_filter

from anomalib import metrics
from sklearn.metrics import average_precision_score
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

import os
import sys
import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity, structural_similarity_index_measure
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
from collections import defaultdict

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
    
    return auroc_score.numpy(), aupro_score ,f1_max_score.numpy(), ap, aurocsp.numpy(), apsp, f1sp.numpy()


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


    

def calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256):
    pred_geometric = []
    pred_aritmetic = []
    image_differences = []
    latent_differences = []
    input_images = []
    output_images = []
    for x, encoded,  image_samples, latent_samples in zip(x0_s, encoded_s,  image_samples_s, latent_samples_s):
            
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        input_images.append(input_image)
        output_images.append(output_image)

        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
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
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
            
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    return {'anomaly_geometric':pred_aritmetic, 'anomaly_geometric':pred_aritmetic, 'latent_discrepancy':latent_differences, 'image_discrepancy':image_differences}



def evaluate_anomaly_maps(anomaly_maps, segmentation):
    for key in anomaly_maps.keys():
        auroc_score, aupro_score ,f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(segmentation, anomaly_maps[key])
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp, = np.round(auroc_score, 4), np.round(aupro_score, 4), np.round(f1_max_score, 4), np.round(ap, 4), np.round(aurocsp, 4), np.round(apsp, 4), np.round(f1sp, 4)
        print('{}: auroc:{:.4f}, aupro:{:.4f}, f1_max:{:.4f}, ap:{:.4f}, aurocsp:{:.4f}, apsp:{:.4f}, f1sp:{:.4f}'.format(key, auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp))


def evaluation(args):
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    try:
        if args.pretrained != '':
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
    model = UNET_models[args.model_size](latent_size=latent_size)
    
    state_dict = torch.load(ckpt)['model']
    print(model.load_state_dict(state_dict))
    model.eval() # important!
    model.cuda()
    print('model loaded')


    print('=='*30)
    print('Starting Evaluation...')
    print('=='*30)

    for category in args.categories:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object:
        diffusion = create_diffusion(f'ddim{args.reverse_steps}', predict_deviation=True, sigma_small=False, predict_xstart=False, diffusion_steps=10)
            

        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        x_s = []
        segmentation_s = []
        

        if args.dataset == 'mvtec':
            test_dataset = MVTECDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=True)
        elif args.dataset == 'visa':
            test_dataset = VISADataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=True)
        elif args.dataset == 'pcb':
            test_dataset = PCBDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)
        
        for ii, (x, seg, object_cls) in enumerate(test_loader):
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

            segmentation_s += [_seg.squeeze() for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]
            x_s += [_x.unsqueeze(0) for _x in x]

        records = [
            ImagePairRecord(
                split="test",
                original_image=torch.clamp(img1, -1.0, 1.0).cpu().numpy() if hasattr(img1, 'cpu') else np.clip(img1, -1.0, 1.0),
                reconstructed_image=torch.clamp(img2, -1.0, 1.0).cpu().numpy() if hasattr(img2, 'cpu') else np.clip(img2, -1.0, 1.0)
            )
            for img1, img2 in zip(x_s, image_samples_s)
        ]

        plot_distribution(records, device=device)

        #anomaly_maps = calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=args.center_size)
        
        #evaluate_anomaly_maps(anomaly_maps, np.stack(segmentation_s, axis=0))
        print('=='*30)  

def cal_similarity(img1, img2, device=None, similarity_type='lpips'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def to_tensor(img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dtype != torch.float32:
            img = img.float()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        if img.ndim == 4 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        return img
    img1 = to_tensor(img1)
    img2 = to_tensor(img2)
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    if similarity_type == 'lpips':
        sim = learned_perceptual_image_patch_similarity(img1, img2, net_type='alex')
    elif similarity_type == 'ssim':
        sim = structural_similarity_index_measure(img1, img2)
    else:
        raise ValueError(f"Invalid similarity type: {similarity_type}")
    return sim.cpu().item()

@dataclass(frozen=True)
class ImagePairRecord:
    split: str  # 'train', 'val', or 'test'
    original_image: object  # np.ndarray or torch.Tensor
    reconstructed_image: object  # np.ndarray or torch.Tensor

def plot_distribution(records: List[ImagePairRecord], device=None):
    if not isinstance(records, list):
        raise TypeError("records must be a list of ImagePairRecord")
    if not all(isinstance(rec, ImagePairRecord) for rec in records):
        raise TypeError("All elements in records must be of type ImagePairRecord")

    splits = defaultdict(list)
    for rec in records:
        splits[rec.split].append((rec.original_image, rec.reconstructed_image))
    plt.figure(figsize=(10, 6))
    for split, pairs in splits.items():
        scores = [cal_similarity(orig, recon, device=device) for orig, recon in pairs]
        plt.hist(scores, bins=30, alpha=0.5, label=split, density=True)
    plt.xlabel('LPIPS Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of LPIPS Similarity Scores by Split')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    REPO_ROOT = os.environ.get('REPO_ROOT', None)
    if REPO_ROOT is not None:
        os.chdir(os.path.dirname(REPO_ROOT))
        print("Current path:", os.getcwd())
        if "ipykernel_launcher" in sys.argv[0]:
            sys.argv = [
                "" ,
                "--dataset", "pcb",
                "--data-dir", os.path.expanduser("~/dataset/PCB/Huang/PCB_DATASET/PCB_gray_128"),
                "--model-size", "UNet_L",
                "--object-category", "all",
                "--anomaly-class", "all",
                "--image-size", "128",
                "--center-size", "128",
                "--center-crop", "False",
                "--pretrained", "DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L/checkpoints/best.pt",
            ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa','pcb'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default= 288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--pretrained", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", type=int, default=5)

    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    elif args.dataset == 'pcb':
        args.num_classes = 1
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
    elif args.object_category=='all' and args.dataset=='pcb':
        args.categories=[
            "pcb",
            ]
    else:
        args.categories = [args.object_category]
        
    evaluation(args)

# Below are cell makrkers used in VSCode
# %%
#
# %%
