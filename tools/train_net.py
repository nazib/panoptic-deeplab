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
from tqdm import tqdm
import _init_paths
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
from segmentation.evaluation import COCOPanopticEvaluator
from segmentation.pastis_augment import applyTransforms

import numpy as np
from torch.utils.tensorboard import SummaryWriter


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
    
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    
    data_loader = build_train_loader_from_cfg(config)
    val_loader = build_test_loader_from_cfg(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    ### Evaluator ###
    pano_meter = PanopticMeter(
            num_classes=19, void_label=20
        )
    panoptic_metric = COCOPanopticEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=val_loader.dataset.train_id_to_eval_id(),
                label_divisor=val_loader.dataset.label_divisor,
                void_label=val_loader.dataset.label_divisor * val_loader.dataset.ignore_label,
                gt_dir=config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=val_loader.dataset.num_classes
            )

    #data_loader_iter = iter(data_loader)
    #val_loader_iter = iter(val_loader)

    start_iter = 0
    max_iter = len(data_loader)#config.TRAIN.MAX_ITER
    best_param_group_id = get_lr_group_id(optimizer)

    # initialize model
    if os.path.isfile(config.MODEL.WEIGHTS):
        model_weights = torch.load(config.MODEL.WEIGHTS)
        get_module(model, distributed).load_state_dict(model_weights, strict=False)
        logger.info('Pre-trained model from {}'.format(config.MODEL.WEIGHTS))
    elif not config.MODEL.BACKBONE.PRETRAINED:
        if os.path.isfile(config.MODEL.BACKBONE.WEIGHTS):
            pretrained_weights = torch.load(config.MODEL.BACKBONE.WEIGHTS)
            get_module(model, distributed).backbone.load_state_dict(pretrained_weights, strict=False)
            logger.info('Pre-trained backbone from {}'.format(config.MODEL.BACKBONE.WEIGHTS))
        else:
            logger.info('No pre-trained weights for backbone, training from scratch.')

    # load model
    total_iter =0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            total_iter = checkpoint['start_iter']
            get_module(model, distributed).load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    writer = SummaryWriter(config.OUTPUT_DIR+"/curve/")

    # Debug output.
    if config.DEBUG.DEBUG:
        debug_out_dir = os.path.join(config.OUTPUT_DIR, 'debug_train')
        PathManager.mkdirs(debug_out_dir)

    # Train loop.
    try:
        for epoch in range(config.SOLVER.Epochs):
            model.train()
            
            for i,data in enumerate(data_loader):
                # data
                start_time = time.time()
                #data = next(data_loader_iter)
                if not distributed:
                    data = to_cuda(data, device)
                data_time.update(time.time() - start_time)
                
                image = data.pop('image')
                image = applyTransforms(image)
                out_dict = model(image.type(torch.cuda.FloatTensor), data)
                loss = out_dict['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Get lr.
                lr = optimizer.param_groups[best_param_group_id]["lr"]
                lr_scheduler.step()

                batch_time.update(time.time() - start_time)
                loss_meter.update(loss.detach().cpu().item(), image.size(0))
                
                if total_iter == 0 or (total_iter + 1) % config.PRINT_FREQ == 0:
                    msg = f'Epoch:{epoch}/{epoch,config.SOLVER.Epochs} \
                    [{i+1}/{max_iter}] LR: {lr}\t' \
                        'Time: {batch_time}s \t' \
                        'Data: {data_time}s\t'
                    loss_dict = get_module(model, distributed).loss_meter_dict
                    msg += get_loss_info_str(loss_dict)
                    writer.add_scalar("Loss",loss_dict['Loss'].val,total_iter)
                    writer.add_scalar("SegLoss",loss_dict['Semantic loss'].val,total_iter)
                    writer.add_scalar("CenterLoss",loss_dict['Center loss'].val,total_iter)
                    writer.add_scalar("OffsetLoss",loss_dict['Offset loss'].val,total_iter)
                    logger.info(msg)
                if total_iter == 0 or (total_iter + 1) % config.CKPT_FREQ == 0:
                    save_pastis_images(image,data,out_dict, debug_out_dir,total_iter)
                    torch.save({
                        'start_iter': total_iter,
                        'epoch':epoch,
                        'state_dict': get_module(model, distributed).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(config.OUTPUT_DIR, 'checkpoint.pth.tar'))
                total_iter = total_iter+1
            '''
            ## Validation loop
            PQ = np.zeros(shape=len(val_loader))
            SQ = np.zeros(shape=len(val_loader))            
            RQ = np.zeros(shape=len(val_loader))
            model.eval()
            with torch.no_grad():
                val_loader = build_test_loader_from_cfg(config)
                print("Validating")
                for i,val_data in enumerate(val_loader):
                    val_data = to_cuda(val_data, device)
                    image = val_data['image']
                    val_out_dict = model(image.type(torch.cuda.FloatTensor), val_data)
                    if i % 100==0: #config.DEBUG.DEBUG_FREQ == 0:
                        #if comm.is_main_process() and config.DEBUG.DEBUG:
                        save_pastis_images(image,val_data,val_out_dict, debug_out_dir,i)
                    print(f'Val Image{i}')

                    pano_meter.add(val_out_dict, val_data)
                    sq, rq, pq = pano_meter.value()
                    PQ[i]= pq.cpu().numpy()[()]
                    RQ[i] = rq.cpu().numpy()[()]
                    SQ[i] = sq.cpu().numpy()[()]
                    if i==1000:
                        break
            print(
            "Epoch:{}/{} Step [{}/{}], SQ {:.1f},  RQ {:.1f}  , PQ {:.1f}".format(
                epoch,
                config.SOLVER.Epochs,
                i + 1,
                len(data_loader),
                SQ.mean() * 100,
                RQ.mean() * 100,
                PQ.mean() * 100,
            ))
            writer.add_scalar("SQ",SQ.mean()*100,epoch)
            writer.add_scalar("RQ",RQ.mean()*100,epoch)
            writer.add_scalar("PQ",PQ.mean()*100,epoch)
            '''

    except Exception:
        logger.exception("Exception during training:")
        raise
    finally:
        torch.save(get_module(model, distributed).state_dict(),
            os.path.join(config.OUTPUT_DIR, 'final_state.pth')) 
        logger.info("Training finished.")
        writer.close()

if __name__ == '__main__':
    main()
