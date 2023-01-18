# ------------------------------------------------------------------------------
# Testing code.
# Example command:
# python tools/test_net_single_core.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import cv2
import os
import pprint
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.data import build_train_loader_from_cfg, build_test_loader_from_cfg
from segmentation.utils import save_debug_images,save_pastis_images
from segmentation.utils import AverageMeter
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils.save_annotation import save_annotation, save_instance_annotation, save_panoptic_annotation
from segmentation.evaluation import (
    SemanticEvaluator, CityscapesInstanceEvaluator, CityscapesPanopticEvaluator,
    COCOInstanceEvaluator, COCOPanopticEvaluator)
from segmentation.model.post_processing import get_cityscapes_instance_format
from segmentation.utils.test_utils import multi_scale_inference
from segmentation.utils.debug import plot_pano_predictions
from segmentation.evaluation import  PanopticMeter

def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger('segmentation_test')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=config.OUTPUT_DIR, name='segmentation_test')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    # build model
    model = build_segmentation_model_from_cfg(config)

    # Change ASPP image pooling
    output_stride = 2 ** (5 - sum(config.MODEL.BACKBONE.DILATION))
    train_crop_h, train_crop_w = config.TEST.CROP_SIZE
    scale = 1. / output_stride
    pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
    pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)

    model.set_image_pooling((1, 1))

    logger.info("Model:\n{}".format(model))
    model = model.to(device)

    # build data_loader
    data_loader = build_test_loader_from_cfg(config)

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

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

    data_time = AverageMeter()
    net_time = AverageMeter()
    post_time = AverageMeter()
    timing_warmup_iter = 10

    semantic_metric = SemanticEvaluator(
        num_classes=data_loader.dataset.num_classes,
        ignore_label=data_loader.dataset.ignore_label,
        output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.SEMANTIC_FOLDER),
        train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id()
    )

    instance_metric = None
    panoptic_metric = None

    if config.TEST.EVAL_INSTANCE:
        if 'cityscapes' in config.DATASET.DATASET:
            instance_metric = CityscapesInstanceEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.INSTANCE_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                gt_dir=os.path.join(config.DATASET.ROOT, 'gtFine', config.DATASET.TEST_SPLIT)
            )
        elif 'coco' in config.DATASET.DATASET:
            instance_metric = COCOInstanceEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.INSTANCE_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                gt_dir=os.path.join(config.DATASET.ROOT, 'annotations',
                                    'instances_{}.json'.format(config.DATASET.TEST_SPLIT))
            )
        elif 'pastis' in config.DATASET.DATASET:
            instance_metric = COCOInstanceEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.INSTANCE_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                gt_dir=os.path.join(config.DATASET.ROOT,f'{config.DATASET.TEST_SPLIT}.json')
            )
        else:
            raise ValueError('Undefined evaluator for dataset {}'.format(config.DATASET.DATASET))

    if config.TEST.EVAL_PANOPTIC:
        if 'cityscapes' in config.DATASET.DATASET:
            panoptic_metric = CityscapesPanopticEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                gt_dir=config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=data_loader.dataset.num_classes
            )
        elif 'coco' in config.DATASET.DATASET:
            panoptic_metric = COCOPanopticEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                gt_dir=config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=data_loader.dataset.num_classes
            )
        elif 'pastis' in config.DATASET.DATASET:
            panoptic_metric = COCOPanopticEvaluator(
                output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.PANOPTIC_FOLDER),
                train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                label_divisor=data_loader.dataset.label_divisor,
                void_label=data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                gt_dir= config.DATASET.ROOT,
                split=config.DATASET.TEST_SPLIT,
                num_classes=data_loader.dataset.num_classes
            )
        else:
            raise ValueError('Undefined evaluator for dataset {}'.format(config.DATASET.DATASET))

    foreground_metric = None
    if config.TEST.EVAL_FOREGROUND:
        foreground_metric = SemanticEvaluator(
            num_classes=2,
            ignore_label=data_loader.dataset.ignore_label,
            output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.FOREGROUND_FOLDER)
        )
    
    if 'pastis' in config.DATASET.DATASET:
        import glob
        file_list = glob.glob(os.path.join(config.DATASET.ROOT,'PASTIS_test',"*.jpg"))
        image_filename_list = [
            os.path.splitext(os.path.basename(ann))[0] for ann in file_list]
    else:
        image_filename_list = [
            os.path.splitext(os.path.basename(ann))[0] for ann in data_loader.dataset.ann_list]

    # Debug output.
    if config.TEST.DEBUG:
        debug_out_dir = os.path.join(config.OUTPUT_DIR, 'debug_test')
        PathManager.mkdirs(debug_out_dir)
    
    if not config.TEST.TEST_TIME_AUGMENTATION:
        if config.TEST.FLIP_TEST or len(config.TEST.SCALE_LIST) > 1:
            config.TEST.TEST_TIME_AUGMENTATION = True
            logger.warning(
                "Override TEST.TEST_TIME_AUGMENTATION to True because test time augmentation detected."
                "Please check your config file if you think it is a mistake.")
    
    pano_meter = PanopticMeter(
            num_classes=19, void_label=20
        )
    PQ = np.zeros(shape=len(data_loader))
    SQ = np.zeros(shape=len(data_loader))            
    RQ = np.zeros(shape=len(data_loader))
    # Test loop.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i == 10:
                break
            if i == timing_warmup_iter:
                data_time.reset()
                net_time.reset()
                post_time.reset()

            # data
            start_time = time.time()
            for key in data.keys():
                try:
                    data[key] = data[key].to(device)
                except:
                    pass

            image = data.pop('image')
            image = image.type(torch.cuda.FloatTensor)
            image = image/image.max()
            torch.cuda.synchronize(device)
            data_time.update(time.time() - start_time)
            start_time = time.time()
            out_dict = model(image.type(torch.cuda.FloatTensor))
            torch.cuda.synchronize(device)
            net_time.update(time.time() - start_time)
            start_time = time.time()
            save_pastis_images(image,data,out_dict, debug_out_dir,i)
            pano_meter.add(out_dict, data)
            
        sq, rq, pq = pano_meter.value()
        PQ  = pq.cpu().numpy()[()]
        RQ  = rq.cpu().numpy()[()]
        SQ = sq.cpu().numpy()[()]
        print(f"Panoptic scores PQ:{PQ} RQ:{RQ} SQ:{SQ}")
                


if __name__ == '__main__':
    main()
