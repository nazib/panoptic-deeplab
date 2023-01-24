# ------------------------------------------------------------------------------
# Saves raw outputs and targets.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

import torch

from .save_annotation import label_to_color_image
from .flow_vis import flow_compute_color
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib


import numpy as np

import cv2

import warnings
warnings.filterwarnings('ignore')
#Colormap (same as in the paper)
from matplotlib import patches
cm = matplotlib.cm.get_cmap('tab20')
def_colors = cm.colors
cus_colors = [def_colors[i] for i in range(1,19)]
cmap = ListedColormap(colors = cus_colors, name='agri',N=20)
label_names = [
"Background",
"Meadow",
"Soft winter wheat",
"Corn",
"Winter barley",
"Winter rapeseed",
"Spring barley",
"Sunflower",
"Grapevine",
"Beet",
 "Winter triticale",
 "Winter durum wheat",
 "Fruits,  vegetables, flowers",
 "Potatoes",
 "Leguminous fodder",
 "Soybeans",
 "Orchard",
 "Mixed cereal",
 "Sorghum",
 "Void label"]

def get_rgb(x,b):
    """Gets an observation from a time series and normalises it for visualisation."""
    im = x[b].cpu().numpy()
    im = im.swapaxes(0,2).swapaxes(0,1)
    mx = im.max()
    mi = im.min()   
    im = (im - mi)/(mx - mi)
    im = np.clip(im, a_max=1, a_min=0)
    return im


def plot_pano_predictions(pano_predictions, pano_gt):
    pano_instances = pano_predictions['center'].detach().squeeze().cpu().numpy()
    pano_semantic_preds = pano_predictions['semantic'].detach().squeeze().cpu().numpy()
    pano_semantic_preds = np.argmax(pano_semantic_preds,axis=0)
    ground_truth_semantic = pano_gt['semantic'].squeeze().cpu().numpy()
    ground_truth_semantic[ground_truth_semantic==255]=0
    colored_mask = np.zeros(shape=(128,128,3),dtype=np.float64)
    semantic_pred = Image.fromarray(pano_semantic_preds,'L').convert('RGB')
    print(f"Unique classes:{np.unique(pano_semantic_preds)}")
    print(f"Unique Instances:{np.unique(pano_instances)}")
    for inst_id in np.unique(pano_instances):
        if inst_id==0:
            continue # ignore background
        mask = (pano_instances==inst_id)
        try:
            # Get polygon contour of the instance mask
            #c,h= cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the ground truth semantic label of the segment
            u,cnt  = np.unique(ground_truth_semantic[mask==1], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]
            if cl==19 or cl==0: # Not showing predictions for "Void" segments
                continue
            else:
                cl = pano_semantic_preds[mask==1].mean()
                color = cmap.colors[int(cl)]
                mask = mask.astype(np.uint8)
                idx = np.where(mask==1)
                colored_mask[idx[0],idx[1],0] = color[0]*255
                colored_mask[idx[0],idx[1],1] = color[1]*255
                colored_mask[idx[0],idx[1],2] = color[2]*255
            # Get the predicted semantic label of the segment
            '''
            for co in c[0::2]:
                poly = patches.Polygon(co[:,0,:], fill=True, alpha=alpha, linewidth=0, color=color)
                ax.add_patch(poly)
                poly = patches.Polygon(co[:,0,:], fill=False, alpha=.8, linewidth=4, color=color)
                ax.add_patch(poly)
            '''
        except ValueError as e:
            print( cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE))
    return Image.fromarray(colored_mask.astype(np.uint8),'RGB')

def save_label(image_numpy, image_path,colormap):
        label = image_numpy[0]
        MAX = image_numpy.max()
        #image_numpy = image_numpy[None,:,:]
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + MAX) / 2.0*MAX * 255.0
        
        for i,cl in enumerate(colormap):
            mask = (label==i).astype(int)
            idx = np.where(mask==1)
            if i==0 or i==19:
                image_numpy[idx[0],idx[1],:] = [0,0,0]
            else:
                image_numpy[idx[0],idx[1],:] = colormap[i]

        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        image_pil.save(image_path)
            
def plot_pano_gt(pano_gt):
    ground_truth_instances = pano_gt['instance'].cpu().numpy()
    ground_truth_semantic = pano_gt['semantic'].cpu().numpy()
    ground_truth_semantic[ground_truth_semantic==19]=0
    colored_mask = np.zeros(shape=(128,128,3),dtype=np.float64)
    for inst_id in np.unique(ground_truth_semantic):
        if inst_id==0 or inst_id==19:
            continue  
        mask = (ground_truth_semantic==inst_id).astype(int)
        try:
            #c,h =  cv2.findContours(mask[0].astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #c,h= cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            u,cnt  = np.unique(ground_truth_semantic[mask==1], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]
            
            if cl==19 or cl==0: # Not showing predictions for Void objects
                continue
            else:
                mask = mask[0].astype(np.uint8)
                idx = np.where(mask==1)
                color = cmap.colors[int(cl)]
                colored_mask[idx[0],idx[1],0] = color[0]*255
                colored_mask[idx[0],idx[1],1] = color[1]*255
                colored_mask[idx[0],idx[1],2] = color[2]*255
        except ValueError as e:
            print(e)
    return Image.fromarray(colored_mask.astype(np.uint8),'RGB')
    

def save_pastis_images(batch_image,input_data,output_data, out_dir,iteration,colormap):
    #batch_size = len(batch_image)
    #im = get_rgb(batch_image, b=b)
    #colored_gt = plot_pano_gt(pano_gt=input_data)
    file_path = '%s/%s_%d.png' % (out_dir, 'debug_batch_gt', iteration)
    #colored_gt.save(file_path,'PNG')
    input = input_data['semantic'].cpu().numpy()
    print(f"Input classes: {np.unique(input)}")
    save_label(input,file_path,colormap)
    
    ## Plot predicted instances
    #colored_pred = plot_pano_predictions(pano_predictions=output_data,
    #                    pano_gt=input_data)
    file_path = '%s/%s_%d.png' % (out_dir, 'debug_batch_output', iteration)
    #colored_pred.save(file_path, 'PNG')
    output = np.squeeze(output_data['semantic'].cpu().numpy(),axis=0)
    #output = output_data['instance'].cpu().numpy()
    #print(f"Output classes: {np.unique(output)}")
    save_label(output,file_path,colormap)
        

def save_debug_images(dataset, batch_images, batch_targets, batch_outputs, out_dir=None, iteration=0,
                      target_keys=('semantic', 'center', 'offset', 'center_weights', 'offset_weights'),
                      output_keys=('semantic', 'center', 'offset'),
                      iteration_to_remove=-1, is_train=True):
    """Saves a mini-batch of images for debugging purpose.
        - image: the augmented input image
        - label: the augmented labels including
            - semantic: semantic segmentation label
            - center: center heatmap
            - offset: offset field
            - instance_ignore_mask: ignore mask
        - prediction: the raw output of the model (without post-processing)
            - semantic: semantic segmentation label
            - center: center heatmap
            - offset: offset field
    Args:
        dataset: The Dataset.
        batch_images: Tensor of shape [N, 3, H, W], a batch of input images.
        batch_targets: Dict, a dict containing batch of targets.
            - semantic: a Tensor of shape [N, H, W]
            - center: a Tensor of shape [N, 1, H, W]
            - offset: a Tensor of shape [N, 2, H, W]
            - semantic_weights: a Tensor of shape [N, H, W]
            - center_weights: a Tensor of shape [N, H, W]
            - offset_weights: a Tensor of shape [N, H, W]
        batch_outputs: Dict, a dict containing batch of outputs.
            - semantic: a Tensor of shape [N, H, W]
            - center: a Tensor of shape [N, 1, H, W]
            - offset: a Tensor of shape [N, 2, H, W]
        out_dir: String, the directory to which the results will be saved.
        iteration: Integer, iteration number.
        target_keys: List, target keys to save.
        output_keys: List, output keys to save.
        iteration_to_remove: Integer, iteration number to remove.
        is_train: Boolean, save train or test debugging image.
    """
    
    batch_size = batch_images.size(0)
    map_height = batch_images.size(2)
    map_width = batch_images.size(3)

    grid_image = np.zeros(
        (map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    num_targets = len(target_keys)
    grid_target = np.zeros(
        (num_targets * map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    num_outputs = len(output_keys)
    grid_output = np.zeros(
        (num_outputs * map_height, batch_size * map_width, 3), dtype=np.uint8
    )

    semantic_pred = torch.argmax(batch_outputs['semantic'].detach(), dim=1)
    if 'foreground' in batch_outputs:
        foreground_pred = torch.argmax(batch_outputs['foreground'].detach(), dim=1)
    else:
        foreground_pred = None

   
    for i in range(batch_size):
        width_begin = map_width * i
        width_end = map_width * (i + 1)

        # save images
        image = dataset.reverse_transform(batch_images[i])
        grid_image[:, width_begin:width_end, :] = image
        #import pdb
        #pdb.set_trace()
        if 'semantic' in target_keys:
            # save gt semantic
            gt_sem = batch_targets['semantic'][i].cpu().numpy()
            gt_sem = label_to_color_image(gt_sem, dataset.create_label_colormap())
            grid_target[:map_height, width_begin:width_end, :] = gt_sem

        if 'center' in target_keys:
            # save gt center
            gt_ctr = batch_targets['center'][i].squeeze().cpu().numpy()
            gt_ctr = gt_ctr[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
            gt_ctr = gt_ctr.clip(0, 255)
            # gt_ctr = 0.7 * gt_ctr + (1 - 0.3) * image
            grid_target[map_height:2 * map_height, width_begin:width_end, :] = gt_ctr

        if 'offset' in target_keys:
            # save gt offset
            gt_off = batch_targets['offset'][i].permute(1, 2, 0).cpu().numpy()
            gt_off = flow_compute_color(gt_off[:, :, 1], gt_off[:, :, 0])
            grid_target[2 * map_height:3 * map_height, width_begin:width_end, :] = gt_off

        if 'semantic_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['semantic_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] / np.max(gt_ign) * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[3 * map_height:4 * map_height, width_begin:width_end, :] = gt_ign

        if 'center_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['center_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[4 * map_height:5 * map_height, width_begin:width_end, :] = gt_ign

        if 'offset_weights' in target_keys:
            # save ignore mask
            gt_ign = batch_targets['offset_weights'][i].cpu().numpy()
            gt_ign = gt_ign[:, :, None] * 255
            gt_ign = np.tile(gt_ign, (1, 1, 3))
            grid_target[5 * map_height:6 * map_height, width_begin:width_end, :] = gt_ign

        if 'foreground' in target_keys:
            # save gt foreground
            gt_fg = batch_targets['foreground'][i].cpu().numpy()
            gt_fg = gt_fg[:, :, None] * 255
            grid_target[6 * map_height:7 * map_height, width_begin:width_end, :] = gt_fg

        if 'semantic' in output_keys:
            # save pred semantic
            pred_sem = semantic_pred[i].cpu().numpy()
            pred_sem = label_to_color_image(pred_sem, dataset.create_label_colormap())
            grid_output[:map_height, width_begin:width_end, :] = pred_sem

        if 'center' in output_keys:
            # save pred center
            pred_ctr = batch_outputs['center'][i].detach().squeeze().cpu().numpy()
            pred_ctr = pred_ctr[:, :, None] * np.array([255, 0, 0]).reshape((1, 1, 3))
            pred_ctr = pred_ctr.clip(0, 255)
            # pred_ctr = 0.7 * pred_ctr + (1 - 0.3) * image
            grid_output[map_height:2 * map_height, width_begin:width_end, :] = pred_ctr

        if 'offset' in output_keys:
            # save pred offset
            pred_ctr = batch_outputs['offset'][i].detach().permute(1, 2, 0).cpu().numpy()
            pred_ctr = flow_compute_color(pred_ctr[:, :, 1], pred_ctr[:, :, 0])
            grid_output[2 * map_height:3 * map_height, width_begin:width_end, :] = pred_ctr

        if 'foreground' in output_keys:
            # save pred foreground
            if foreground_pred is not None:
                pred_fg = foreground_pred[i].cpu().numpy()
                pred_fg = pred_fg[:, :, None] * 255
                grid_output[3 * map_height:4 * map_height, width_begin:width_end, :] = pred_fg

    if out_dir is not None:
        if is_train:
            pil_image = img.fromarray(grid_image.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_target.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_output.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
        else:
            pil_image = img.fromarray(grid_image.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_test_images', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')
            if grid_target.size:
                pil_image = img.fromarray(grid_target.astype(dtype=np.uint8))
                with open('%s/%s_%d.png' % (out_dir, 'debug_test_targets', iteration), mode='wb') as f:
                    pil_image.save(f, 'PNG')
            pil_image = img.fromarray(grid_output.astype(dtype=np.uint8))
            with open('%s/%s_%d.png' % (out_dir, 'debug_test_outputs', iteration), mode='wb') as f:
                pil_image.save(f, 'PNG')

    if is_train:
        if iteration_to_remove >= 0:
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_images', iteration_to_remove))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', iteration_to_remove))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration_to_remove)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', iteration_to_remove))
            # 0 is a special iter
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_images', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_images', 0))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_targets', 0))
            if os.path.exists('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', 0)):
                os.remove('%s/%s_%d.png' % (out_dir, 'debug_batch_outputs', 0))
