#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# modified from https://github.com/facebookresearch/CutLER:maskcut/maskcut.py

import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# modfied by Xudong Wang based on third_party/TokenCut
from utils import IoU, resize_pil, detect_box

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats)
    # convert the affinity matrix to a binary one.
    A = (A > tau).float()
    A[A == 0] = eps  # positive semi-definite
    d_i = torch.sum(A, dim=1)
    D = torch.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    v, eigenvectors = torch.lobpcg(A=D-A, B=D, k=2, largest=False)
    second_smallest_vec = eigenvectors[:, 1]
    return second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    bipartition = second_smallest_vec > torch.mean(second_smallest_vec)
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def panoptic_cut_forward(feats, dims, scales, init_image_size, tau=0, N=3, unoccupied=None):
    """
    Implementation of panoptic_cut.`
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """
    bipartitions = []
    unoccupied = torch.ones(dims, dtype=torch.bool).view(-1)
    occupied = 0

    for i in range(N):
        # process Ncut with unoccupied pixels only
        feats_i = feats[:, unoccupied]

        A, D = get_affinity_matrix(feats_i, tau)
        # sp: sparse
        eigenvec_sp = second_smallest_eigenvector(A, D)
        bipartition_sp = get_salient_areas(eigenvec_sp)

        bipartition = torch.zeros(dims, dtype=torch.bool).view(-1).to(bipartition_sp)
        eigenvec = torch.zeros(dims, dtype=torch.float).view(-1).to(eigenvec_sp)

        bipartition[unoccupied] = bipartition_sp
        eigenvec[unoccupied] = eigenvec_sp
        eigenvec[unoccupied == False] = torch.abs(eigenvec_sp).min()

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = torch.argmax(torch.abs(eigenvec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition_sp = torch.logical_not(bipartition_sp)  # should negate at the sparse matrix!
            bipartition = torch.zeros(dims, dtype=torch.bool).view(-1).to(bipartition_sp)
            bipartition[unoccupied] = bipartition_sp
        seed = torch.argmax(eigenvec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).float()
        pseudo_mask  = detect_box(bipartition, seed, dims)

        # mask out foreground areas in previous stages
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartitions.append(bipartition)

        occupied += pseudo_mask.view(-1)
        occupied = (occupied > 0).float()
        unoccupied = torch.logical_not(occupied)

        if unoccupied.sum() <= 5:
            print('early stop at', len(bipartitions))
            return [b.cpu().numpy() for b in bipartitions] 

    return [b.cpu().numpy() for b in bipartitions] 

@torch.no_grad()
def panoptic_cut(img_path, backbone,patch_size, tau, N=1, fixed_size=480, cpu=False):
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((fixed_size, fixed_size), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = resize_pil(I_new, patch_size)

    unoccupied = None

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]

    bipartition = panoptic_cut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, unoccupied=unoccupied)
    bipartitions += bipartition
    return bipartitions, I_new