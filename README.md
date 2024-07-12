<div align="center">
  <h1> In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation </h1>
</div>


<div align="center">
  <h3><a href=http://dahyun-kang.github.io>Dahyun Kang</a> &nbsp;&nbsp;&nbsp;&nbsp; <a href=http://cvlab.postech.ac.kr/~mcho/>Minsu Cho</a></h3>
</div>
<br />

This repo is the official implementation of the ECCV 2024 paper [In Defense of Lazy Visual Grounding for Open-Vocabulary Semantic Segmentation]()


## Conda installation command
```bash
conda env create -f environment.yml --prefix $YOURPREFIX
```
`$YOUPREFIX` is typically `/home/anaconda3`


## Dependencies

This repo is built on [CLIP](https://github.com/openai/CLIP), [SCLIP](https://github.com/wangf3014/SCLIP), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). 

```
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex yapf==0.40.1
```


## Dataset preparation
Please make it compatible with Pascal VOC 2012, Pascal Context, COCO stuff 164K, COCO object, and ADEChallengeData2016 following the [MMSeg data preparation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md).
The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

Place them under `yourdatasetroot/` directory such that:
```
    yourdatasetroot/
    ├── ADEChallengeData2016/
    │   ├── annotations/
    │   ├── images/
    │   ├── ...
    ├── VOC2012/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   ├── ...
    ├── coco_stuff164k/
    │   ├── annotations/
    │   ├── images/
    │   ├── ...
    ├── ...
```



## 1) Panoptic Cut for unsupervised object mask discovery
```bash
cd panoptic_cut
python predict.py --logs panoptic_cut --dataset {VOC2012/coco_stuff164k/ADEChallengeData2016} --N 16 --imgsize 640 --datasetroot yourdatasetroot
```


## 2) Visual grounding & Segmentation evaluation
```bash
cd lazygrounding
python eval.py --config ./configs/{cfg_context59/cfg_context60/cfg_voc20/cfg_voc21}.py --maskpred_root VOC2012/panoptic_cut
python eval.py --config ./configs/cfg_ade20k.py --maskpred_root ADEChallengeData2016/panoptic_cut
python eval.py --config ./configs/{cfg_coco_object/fg_coco_stuff164k}.py --maskpred_root coco_stuff164k/panoptic_cut
```
The run is a single-GPU compatible.


## Related repos
Our project refers to and heavily borrows some the codes from the following repos:

* [[SCLIP]](https://github.com/wangf3014/SCLIP)
* [[CutLER]](https://github.com/facebookresearch/CutLER)


## Acknowledgements
This work was supported by Samsung Electronics (IO201208-07822-01), the NRF grant (NRF-2021R1A2C3012728 (45%)), and the IITP grants (RS-2022-II220959: Few-Shot Learning of Causal Inference in Vision and Language for Decision Making (50%$), RS-2019-II191906: AI Graduate School Program at POSTECH (5%$)) funded by Ministry of Science and ICT, Korea.
We also thank [Sua Choi](https://github.com/sua-choi) for her helpful discussion.
