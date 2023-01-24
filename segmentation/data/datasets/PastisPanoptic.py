# ------------------------------------------------------------------------------
# Loads COCO panoptic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import json
import os

import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
import geopandas as gpd

from .base_dataset import BaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms, Resize, PanopticTargetGenerator, SemanticTargetGenerator
from ..transforms import PASTIS_PanopticTargetGenerator, PASTIS_SemanticTargetGenerator

_PASTIS_PANOPTIC_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'PASTIS_train': 111422,
                     'PASTIS_test': 3800},
    num_classes=20,
    ignore_label=19,
)

# Add 1 void label.
_COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 0])

_COCO_PANOPTIC_EVAL_ID_TO_TRAIN_ID = {
    v: k for k, v in enumerate(_COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID[:-1])
}

_COCO_PANOPTIC_THING_LIST = list(range(80))  # the first 80 classes are `thing` classes

PASTIS_CATEGORIES = [
                        {'id':int(0),'name':"Background","isthing": 0},
                        {'id':int(1),'name':"Meadow","isthing": 1},
                        {'id':int(2),'name':"Soft winter wheat","isthing": 1},
                        {'id':int(3),'name':"Corn","isthing": 1},
                        {'id':int(4),'name':"Winter barley", "isthing": 1},
                        {'id':int(5),'name':"Winter rapeseed", "isthing": 1},
                        {'id':int(6),'name':"Spring barley", "isthing": 1},
                        {'id':int(7),'name':"Sunflower", "isthing": 1},
                        {'id':int(8),'name':"Grapevine", "isthing": 1},
                        {'id':int(9),'name':"Beet", "isthing": 1},
                        {'id':int(10),'name':"Winter triticale", "isthing": 1},
                        {'id':int(11),'name':"Winter durum wheat", "isthing": 1},
                        {'id':int(12),'name':"Fruits,  vegetables, flowers", "isthing": 1},
                        {'id':int(13),'name':"Potatoes", "isthing": 1},
                        {'id':int(14),'name':"Leguminous fodder", "isthing": 1},
                        {'id':int(15),'name':"Soybeans", "isthing": 1},
                        {'id':int(16),'name':"Orchard", "isthing": 1},
                        {'id':int(17),'name':"Mixed cereal", "isthing": 1},
                        {'id':int(18),'name':"Sorghum", "isthing": 1},
                        {'id':int(19),'name':"Void label", "isthing": 1}
]


class PASTISPanoptic(BaseDataset):
    """
    COCO panoptic segmentation dataset.
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
        semantic_only: Bool, only use semantic segmentation label.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
    """
    def __init__(self,
                 root,
                 dataset_type,
                 split,
                 min_resize_value=641,
                 max_resize_value=641,
                 resize_factor=32,
                 is_train=True,
                 crop_size=(641, 641),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=1.0,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=10,
                 small_instance_weight=0.8,
                 **kwargs):
        super(PASTISPanoptic, self).__init__(root,dataset_type, split, is_train, crop_size, mirror, min_scale, max_scale,
                                           scale_step_size, mean, std)
        assert split in _PASTIS_PANOPTIC_INFORMATION.splits_to_sizes.keys()
        self.num_classes = _PASTIS_PANOPTIC_INFORMATION.num_classes
        self.ignore_label = _PASTIS_PANOPTIC_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 256
        self.label_dtype = np.float32
        self.thing_list = _COCO_PANOPTIC_THING_LIST

        self.metadata = gpd.read_file(os.path.join(self.root,"metadata.geojson"))
        self.metadata.index = self.metadata["ID_PATCH"].astype(int)
        self.metadata.sort_index(inplace=True)
        self.colormap = self.create_label_colormap()
        # Get image and annotation list.
        self.img_list = []
        self.ann_list = []
        self.ins_list = []
        #json_filename = os.path.join(self.root, 'annotations', 'panoptic_{}_trainId.json'.format(self.split))
        json_filename = os.path.join(self.root, '{}.json'.format(self.split))
        dataset = json.load(open(json_filename))
        
        # First sort by image id.
        images = sorted(dataset['images'], key=lambda i: i['id'])
        annotations = sorted(dataset['annotations'], key=lambda i: i['image_id'])
        for img in images:
            img_file_name = img['file_name']
            self.img_list.append(os.path.join(self.root, self.split, img_file_name))
        for ann in annotations:
            ann_file_name = ann['file_name']
            self.ann_list.append(os.path.join(
                self.root, 'panoptic_{}'.format(self.split), ann_file_name))
            self.ins_list.append(ann['segments_info'])
        
        #import pdb
        #pdb.set_trace()
        assert len(self) == _PASTIS_PANOPTIC_INFORMATION.splits_to_sizes[self.split]

        #self.pre_augmentation_transform = Resize(min_resize_value, max_resize_value, resize_factor)
        #self.transform = build_transforms(self, is_train)
        self.transform = None
        if semantic_only:
            if 'pastis' in dataset_type:
                self.target_transform = PASTIS_SemanticTargetGenerator(self.ignore_label, self.rgb2id)
            else:
                self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            if dataset_type == 'pastis_panoptic':
                self.target_transform = PASTIS_PanopticTargetGenerator(self.ignore_label, self.rgb2id,self.metadata,
                                                                self.colormap,
                                                                sigma=2, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                                small_instance_area=small_instance_area,
                                                                small_instance_weight=small_instance_weight)
            else:
                self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id,_COCO_PANOPTIC_THING_LIST,
                                                                sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                                small_instance_area=small_instance_area,
                                                                small_instance_weight=small_instance_weight)    
            # Generates semantic label for evaluation.
        if "pastis" in dataset_type:
            self.raw_label_transform = PASTIS_SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    @staticmethod
    def train_id_to_eval_id():
        return _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in COCO panoptic benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        cm = matplotlib.cm.get_cmap('tab20')
        def_colors = cm.colors
        cus_colors = [def_colors[i] for i in range(1,19)]
        cmap = ListedColormap(colors = cus_colors, name='agri',N=20)
        
        colormap = np.zeros((len(PASTIS_CATEGORIES), 3), dtype=np.uint8)
        for i, color in enumerate(cmap.colors):
            colormap[i] = (np.uint8(color[0]*255),np.uint8(color[1]*255),np.uint8(color[2]*255))
        return colormap
        
