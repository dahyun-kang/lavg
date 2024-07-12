import os
import argparse
import clip_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='OVSeg evaluation with MMSeg')
    parser.add_argument('--config', default='')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument('--wait_time', default='1')
    parser.add_argument('--pamr_steps', type=int, default=0)
    parser.add_argument('--maskpred_root', type=str, default='panoptic_cut')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = int(args.wait_time)
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pamr_steps = int(args.pamr_steps)
    cfg.model.maskpred_root = args.maskpred_root
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    cfg = trigger_visualization_hook(cfg, args)

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()
