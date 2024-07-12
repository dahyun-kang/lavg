# imported from TokenCut (CVPR 2022)

import torch
import numpy as np
import PIL.Image as Image
from scipy import ndimage

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h

def IoU(mask1, mask2):
    mask1, mask2 = (mask1 > 0.5).to(torch.bool), (mask2 > 0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

def detect_box(bipartition, seed, dims):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    b = bipartition.cpu().numpy()
    # b = ndimage.binary_opening(b)
    objects, _ = ndimage.label(b)
    cc = objects[np.unravel_index(seed.item(), dims)]
    objects = torch.from_numpy(objects)

    binary_mask = objects.to(bipartition.device).to(torch.float)
    binary_mask[objects == cc] = 1.
    binary_mask[objects != cc] = 0.
    return binary_mask