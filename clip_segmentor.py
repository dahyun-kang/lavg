import torch
import torch.nn as nn
import sys
sys.path.append("..")

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmengine.structures import PixelData

from prompts.imagenet_template import openai_imagenet_template
import clip  # from the current working directory
from pamr import PAMR


@MODELS.register_module()
class CLIPForSegmentation(BaseSegmentor):
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                    pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40,
                    slide_stride=112, slide_crop=224, area_thd=None, maskpred_root=''):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _ = clip.load(clip_path, device=device, jit=False)

        query_words, self.query_idx = get_cls_idx(name_path)
        self.query_words = query_words
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.bgidx = self.query_idx.index(1)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.maskpredroot = f'panoptic_cut/pred/{maskpred_root}/'
        self.remove_falsefg = 'voc21' in name_path

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None, obj_masks=None):  # current best
        ''' step 2) object grounding '''
        if type(img) == list:
            img = img[0]

        image_features = self.net.encode_image(img, return_all=True, csa=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features[:, 1:]

        patch_size = self.net.visual.patch_size
        b, wh, c = image_features.shape
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size

        image_features = image_features.permute(0, 2, 1).reshape(b, c, w, h)

        if logit_size == None:
            image_features = nn.functional.interpolate(image_features, size=img.shape[-2:], mode='bilinear', align_corners=False)
        else:
            image_features = nn.functional.interpolate(image_features, size=logit_size, mode='bilinear', align_corners=False)
        logits = torch.einsum('b d h w, c d -> b h w c', image_features, self.query_features)
        # logits = torch.zeros(1, *image_features.shape[-2], self.query_features.shape[0]).to(image_features.device).type(image_features.dtype)

        pred_fg = torch.zeros(logits.shape[1:3]).type(torch.bool).to(logits.device)
        for i in reversed(range(obj_masks.shape[0])):
            if obj_masks[i].sum() == 0: continue
            mask_feat = torch.sum(image_features * obj_masks[i], dim=[-1, -2]) / obj_masks[i].sum()
            assert b == 1, 'use repeat instead of unsqueeze(0)'
            logits[obj_masks[i].unsqueeze(0)] = mask_feat @ self.query_features.T
            pred_fg[obj_masks[i]] = True

        if self.remove_falsefg:
            bglogit = torch.zeros(logits.shape).type(torch.bool).to(logits.device)
            bglogit[:, :, :, self.bgidx:][obj_masks[-1].unsqueeze(0)] = True
            logits[bglogit == True] = logits.min().item()

        return logits.permute(0, 3, 1, 2)

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                obj_masks = img_metas[0]['maskpred']['pred_masks']
                if obj_masks.shape[0]:
                    obj_masks = nn.functional.interpolate(obj_masks.unsqueeze(0).type(torch.float32), size=(h_img, w_img), mode='nearest').squeeze(0).type(torch.bool)  # avoid mask shirnkage
                    crop_obj_masks = obj_masks[:, y1:y2, x1:x2]
                else:
                    crop_obj_masks = torch.zeros(0, *crop_img.shape[2:]).type(torch.bool)
                if crop_img.shape[-2:] != crop_obj_masks.shape[-2:]:
                    print(crop_img.shape, crop_obj_masks.shape); import pdb ; pdb.set_trace()
                    h, w = crop_img.shape[-2:]
                    m, ch, cw = crop_obj_masks.shape
                    _crop_obj_masks = torch.zeros(m, h, w).type(crop_obj_masks.dtype).to(crop_obj_masks.device)
                    _crop_obj_masks[:, :ch, :cw] = crop_obj_masks
                    crop_obj_masks = _crop_obj_masks

                crop_seg_logit = self.forward_feature(crop_img, obj_masks=crop_obj_masks)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        fname = batch_img_metas[0]['img_path'].split('/')[-1].split('.')[0]

        maskpred = torch.load(self.maskpredroot + f'{fname}.pth')
        batch_img_metas[0]['maskpred'] = dict()
        batch_img_metas[0]['maskpred']['pred_masks'] = maskpred

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], obj_masks=maskpred)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)
                seg_logits[1:] *= area_pred.transpose(0, -1)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
