# ------------------------------------------------------------------------------
# Training code.
# Example command:
# python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import comm
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.data import build_train_loader_from_cfg, build_test_loader_from_cfg
from segmentation.solver import get_lr_group_id
from segmentation.utils import save_debug_images,save_pastis_images
from segmentation.utils import AverageMeter
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module
from segmentation.evaluation import  PanopticMeter
from segmentation.pastis_augment import applyTransforms
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger('segmentation')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=config.OUTPUT_DIR, distributed_rank=args.local_rank)

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    
    # build model
    model = build_segmentation_model_from_cfg(config)
    logger.info("Model:\n{}".format(model))

    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
    
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.to(device)
    
    data_loader = build_test_loader_from_cfg(config)
   
    pano_meter = PanopticMeter(
            num_classes=19, void_label=20
        )
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    # initialize model
    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')
    
    test_out_dir = os.path.join(config.OUTPUT_DIR,'test_results')
    if not os.path.exists(test_out_dir):
        os.mkdir(test_out_dir)
    
    try:
        ## Test loop
        PQ = np.zeros(shape=len(data_loader))
        SQ = np.zeros(shape=len(data_loader))            
        RQ = np.zeros(shape=len(data_loader))
        model.eval()
        with torch.no_grad():
            print("Start Testing Loop")
            for i,val_data in enumerate(tqdm(data_loader)):
                #val_data = data_loader.dataset.__getitem__(i)
                val_data = to_cuda(val_data, device)
                image = val_data['image']
                val_out_dict = model(image.type(torch.cuda.FloatTensor), val_data)
                save_pastis_images(image,val_data,val_out_dict, test_out_dir,i)
                pano_meter.add(val_out_dict, val_data)
                sq, rq, pq = pano_meter.value()
                PQ[i]= pq.cpu().numpy()[()]*100
                RQ[i] = rq.cpu().numpy()[()]*100
                SQ[i] = sq.cpu().numpy()[()]*100
        print(
        "Step [{}/{}], SQ {:.1f},  RQ {:.1f}  , PQ {:.1f}".format(
            i,
            i + 1,
            len(data_loader),
            SQ.mean() * 100,
            RQ.mean() * 100,
            PQ.mean() * 100,
        ))
        all_results = pd.DataFrame(columns=['SQ','RQ','PQ'])
        all_results['PQ']= PQ
        all_results['RQ']= RQ
        all_results['SQ']= SQ
        all_results.to_csv(f'panoptic_scores_{config.OUTPUT_DIR.split("/")[-1]}.csv')
    except Exception:
        logger.exception("Exception during training:")
        raise
if __name__ == '__main__':
    main()
