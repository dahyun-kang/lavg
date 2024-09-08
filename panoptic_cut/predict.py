"""
download pretrained weights to ./weights
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth
"""

import os
import argparse
import matplotlib.pyplot as plt

from utils import IoU
import numpy as np
import PIL.Image as Image
import torch
from scipy import ndimage
from colormap import random_color

import dino
from crf import densecrf, densecrf_postprocess
from panoptic_cut import panoptic_cut

from cog import BasePredictor, Input, Path
from torchvision import transforms

from utils import IoU, resize_pil, detect_box
import torch.nn.functional as F
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

class Predictor(BasePredictor):
    def setup(self, args):
        self.outputroot = args.outputroot
        self.N = args.N
        self.logs = args.logs
        self.imgsize = args.imgsize
        self.pos_sim_hist = torch.zeros(2000).cuda()
        self.neg_sim_hist = torch.zeros(2000).cuda()

        outputroot = os.path.join(self.outputroot, self.logs)
        if os.path.exists(outputroot):
            pass  # print(f'{outputroot} exists') ; exit()
        else:
            os.makedirs(outputroot)


        """Load the model into memory to make running multiple predictions efficient"""

        # DINO pre-trained model
        vit_features = "k"
        self.patch_size = 8

        # adapted dino.ViTFeat to load from local pretrained_path
        dino_ckpt_path = 'dino_vitbase8_pretrain.pth'
        if not os.path.exists(dino_ckpt_path):
            import wget
            wget.download('https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth')

        self.backbone_base = dino.ViTFeat(
            dino_ckpt_path,
            768,
            "base",
            vit_features,
            self.patch_size,
        )

        self.backbone_base.eval()
        self.backbone_base.cuda()

    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        model: str = Input(
            description="Choose the model architecture",
            default="base",
            choices=["small", "base"]
        ),
        n_pseudo_masks: int = Input(
            description="The maximum number of pseudo-masks per image",
            default=3,
        ),
        tau: float = Input(
            description="Threshold used for producing binary graph",
            default=0.15,
        ),
        vis: bool = Input(
            description="Boolean for mask visualization",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        backbone = self.backbone_base

        fixed_size = self.imgsize

        bipartitions, I_new = panoptic_cut(
            str(image),
            backbone,
            self.patch_size,
            tau,
            N=n_pseudo_masks,
            fixed_size=fixed_size,
            cpu=False,
        )

        I = Image.open(str(image)).convert("RGB")
        pseudo_mask_list = []
        for idx, bipartition in enumerate(bipartitions):
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(pseudo_mask).cuda()

            if IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            pseudo_mask = densecrf_postprocess(pseudo_mask, size=I.size)
            pseudo_mask_list.append(pseudo_mask)
        bg_mask = np.stack(pseudo_mask_list, axis=0).sum(axis=0) == 0
        pseudo_mask_list.append(bg_mask)

        pseudo_mask_tensor = torch.stack([torch.tensor(e) for e in pseudo_mask_list], dim=0).cuda()
        n_pseudo_masks = len(pseudo_mask_list) - 1

        fname = image.split('/')[-1].split('.')[0]
        output_path = os.path.join(self.outputroot, self.logs, f'{fname}.pth')
        torch.save(pseudo_mask_tensor, output_path)

        if vis:
            self.vis(image, pseudo_mask_list, output_path)

        return pseudo_mask_list

    def vis(self, image, pseudo_mask_list, output_path):
        I = Image.open(str(image)).convert("RGB")
        out = np.array(I)
        plt.rcParams['figure.figsize'] = [16, 8]
        for i, pseudo_mask in enumerate(pseudo_mask_list):
            out = vis_mask(out, pseudo_mask, random_color(rgb=True))
            # out = vis_mask(out, pseudo_mask, colors[i % len(colors)])
            plt.subplot(4, 5, i + 3).set_title(f'mask {i}'); plt.axis('off')
            plt.imshow(pseudo_mask)

        plt.subplot(4, 5, 1).set_title(f'input'); plt.axis('off')
        plt.imshow(I)
        plt.subplot(4, 5, 2).set_title(f'all'); plt.axis('off')
        plt.imshow(out)
        plt.savefig(str(output_path) + '.jpg', bbox_inches='tight')
        plt.clf() ; plt.cla() ; plt.close()

        return Path(output_path)

def vis_mask(input, mask, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CutCut')
    parser.add_argument('--datasetroot', type=str, default='yourdatasetroot/datasets', help='Dataset root')
    parser.add_argument('--logs', type=str, default='', help='Output log dir name')
    parser.add_argument('--dataset', type=str, default='coco_stuff100', help='Output path')
    parser.add_argument('--N', type=int, default=8, help='N pseudo masks')
    parser.add_argument('--imgsize', type=int, default=640, help='Image size')
    parser.add_argument('--tau', nargs='+', type=float, default=[0.3], help='threshold')
    parser.add_argument('--demo', action='store_true', help='flag for demo')
    parser.add_argument('--vis', action='store_true', help='flag for visualization of masks')

    args = parser.parse_args()

    if args.demo:
        args.imgroot = 'assets'
        args.outputroot = os.path.join('pred', 'demo')
        args.vis = True  # force visualization for demo
        imglist = os.listdir(args.imgroot)
    else:
        if args.dataset in ['coco_object', 'coco_stuff']:
            args.imgroot = os.path.join(args.datasetroot, 'coco_stuff164k/images/val2017')
            args.outputroot = os.path.join('pred', 'coco_stuff164k')
            imglistfile = 'imglist/coco_val2017.txt'
        elif args.dataset == 'ade21k':
            args.imgroot = os.path.join(args.datasetroot, 'ADEChallengeData2016/images/validation')
            args.outputroot = os.path.join('pred', 'ADEChallengeData2016')
            imglistfile = 'imglist/ade_imgs.txt'
        elif args.dataset in ['voc21', 'voc20']:
            args.imgroot = os.path.join(args.datasetroot, 'VOCdevkit/VOC2012/JPEGImages')
            args.outputroot = os.path.join('pred', 'VOC2012')
            imglistfile = 'imglist/voc_imgs.txt'
        elif args.dataset in ['context60', 'context59']:
            args.imgroot = os.path.join(args.datasetroot, 'VOCdevkit/VOC2012/JPEGImages')
            args.outputroot = os.path.join('pred', 'VOC2012')  # force to share the same output dir with voc20/21
            imglistfile = 'imglist/context_imgs.txt'
        elif args.dataset == 'cityscapes':
            args.imgroot = os.path.join(args.datasetroot, 'Cityscapes/leftImg8bit/val')
            args.outputroot = os.path.join('pred', 'Cityscapes')
            imglistfile = 'imglist/cityscapes_imgs.txt'
        else:
            raise NotImplementedError

        imglist = [line.rstrip() for line in open(imglistfile)]
    imglist.sort()
    total_len = len(imglist)

    predictor = Predictor()
    predictor.setup(args)

    for i, img in enumerate(imglist):
        fname = img.split('/')[-1].split('.')[0]
        output_path = os.path.join(args.outputroot, args.logs, f'{fname}.pth')
        if os.path.exists(output_path):
            print(i, 'continuing')
            continue

        for tau in args.tau:
            fname = os.path.join(args.imgroot, img)
            predictor.predict(image=fname, model='base', n_pseudo_masks=args.N, tau=tau, vis=args.vis)
        print(f'{i}/{total_len}: {fname}')

