# ------------------------------------------------------------------------------
# Base class for loading a segmentation Dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import os

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils import data
from pathlib import Path

class BaseDataset(data.Dataset):
    """
    Base class for segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
    """
    def __init__(self,
                 root,
                 dataset_type,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.root = root
        self.split = split
        self.is_train = is_train
        self.dataset_type = dataset_type

        self.crop_h, self.crop_w = crop_size

        self.mirror = mirror
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

        self.mean = mean
        self.std = std

        self.pad_value = tuple([int(v * 255) for v in self.mean])

        # ======== override the following fields ========
        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.label_dtype = 'uint8'

        # list of image filename (required)
        self.img_list = []
        # list of label filename (required)
        self.ann_list = []
        # list of instance dictionary (optional)
        self.ins_list = []

        self.has_instance = False
        self.label_divisor = 1000

        self.raw_label_transform = None
        self.pre_augmentation_transform = None
        self.transform = None
        self.target_transform = None
    
    def findidx(self,all_list,id):
        ### Logic is wrong
        idx = None
        count =0
        for f in all_list:
            f_id = os.path.basename(f).split('_')[-1].split('.')[0]
            if id ==f_id:
                idx = count
                break 
            else:
                count+=1
        return idx

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert os.path.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        image = self.read_image(self.img_list[index], 'RGB')
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            #assert os.path.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            if self.dataset_type == 'pastis_panoptic':
                dataset_dict["image_file"] = self.img_list[index]
                ann_id =  os.path.basename(self.img_list[index]).split('_')[1]
                ann_index = self.findidx(self.ann_list,ann_id)
                #print(f'Image file id :{ann_id} Index:{index}')
                try:
                    ins_label, sem_label,zone_label,heat_map = self.read_pastis_label(self.ann_list[ann_index], self.label_dtype)
                except:
                    print(f'Image file id :{ann_id} Index:{index}')
            else:
                ins_label = self.read_label(self.ann_list[index], self.label_dtype)
        else:
            ins_label = None
        
        if self.dataset_type !='pastis_panoptic':
            raw_label = label.copy()
            if self.raw_label_transform is not None:
                raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
        
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        #name = os.path.splitext(os.path.basename(self.ann_list[ann_index]))[0]
        # TODO: how to return the filename?
        # dataset_dict['name'] = np.array(name)

        # Resize and pad image to the same size before data augmentation
        if self.pre_augmentation_transform is not None:
            image, sem_label = self.pre_augmentation_transform(image, sem_label)
            size = image.shape
            dataset_dict['size'] = np.array(size)
        else:
            dataset_dict['size'] = dataset_dict['raw_size']

        # Apply data augmentation.
        if self.transform is not None:
            image, sem_label = self.transform(image, sem_label)
        else:
            image = np.transpose(image,axes=(2,0,1))
        #import pdb
        #pdb.set_trace()   
        if self.dataset_type == 'pastis_panoptic':
            
            dataset_dict['image'] = image
            dataset_dict['semantic'] = torch.as_tensor(sem_label.astype('long'))
            dataset_dict['instance'] = torch.as_tensor(ins_label.astype('long'))
            dataset_dict['zone'] = torch.as_tensor(zone_label.astype('long'))
            dataset_dict['heat_map'] = torch.as_tensor(heat_map.astype('float32'))
        else:
            dataset_dict['image'] = image
            if not self.has_instance:
                dataset_dict['semantic'] = torch.as_tensor(raw_label.astype('long'))

        # Generate training target.
        if self.target_transform is not None:
            if self.dataset_type =="pastis_panoptic":
                label_dict = self.target_transform(sem_label,ins_label,heat_map, self.ins_list[ann_index],ann_id)
            else:
                label_dict = self.target_transform(sem_label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]

        return dataset_dict

    @staticmethod
    def read_image(file_name, format=None):
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        #image = (image - image.min())/(image.max() - image.min())
        #print(f"Max : {image.max()} Min: {image.min()}")
        return image

    @staticmethod
    def read_label(file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        label = Image.open(file_name)
        return np.asarray(label, dtype=dtype)
    
    @staticmethod
    def read_pastis_label(file_name, dtype='uint8'):
        # In some cases, `uint8` is not enough for label
        ins_label = np.asarray(Image.fromarray(np.load(file_name).astype(np.uint8),mode='L').convert('RGB'))
        root = Path(file_name).parent.parent
        file_id = file_name.split(os.sep)[-1].split('_')[-1].split('.')[0]
        sem_label = np.load(os.path.join(root,'ANNOTATIONS',f'TARGET_{file_id}.npy'))[0]
        #sem_label = np.asarray(Image.fromarray(sem_label.astype(np.uint8),mode='L').convert('RGB'))
        zone_label = np.load(os.path.join(root,'INSTANCE_ANNOTATIONS',f'ZONES_{file_id}.npy'))
        heat_map = np.load(os.path.join(root,'INSTANCE_ANNOTATIONS',f'HEATMAP_{file_id}.npy'))
        return ins_label, sem_label, zone_label,heat_map

    def reverse_transform(self, image_tensor):
        """Reverse the normalization on image.
        Args:
            image_tensor: torch.Tensor, the normalized image tensor.
        Returns:
            image: numpy.array, the original image before normalization.
        """
        dtype = image_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image_tensor.device)
        image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        image = image_tensor.mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .permute(1, 2, 0)\
                            .cpu().numpy()
        return image

    @staticmethod
    def train_id_to_eval_id():
        return None
