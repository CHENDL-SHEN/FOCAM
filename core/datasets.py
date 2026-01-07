import os
import cv2
import glob
from numpy.core.fromnumeric import size
import torch

import math
import imageio
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding

from tools.general.xml_utils import read_xml
from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic
from torchvision import transforms


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data

def load_img_label_list_from_npy(img_name_list,npupath):
    cls_labels_dict = np.load(npupath, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


class Base_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir
        self.dataset = dataset

        if(self.dataset=="voc12"):
            self.image_dir = self.root_dir + 'JPEGImages/'
            self.mask_dir = self.root_dir + 'SegmentationClassAug/'
            assert os.path.exists(self.mask_dir), 'please the Voc Aug masks into'+self.mask_dir 

        elif(self.dataset=="coco"):
            if('train' in domain):
                self.image_dir = self.root_dir + 'train2014/'
                self.mask_dir = self.root_dir + 'SegmentationClass/' + 'train2014/'
            if('val' in domain):
                self.image_dir = self.root_dir + 'val2014/'
                self.mask_dir = self.root_dir + 'SegmentationClas/' + 'val2014/'
            else:
                assert "domain err"
        
        elif(self.dataset=="PubBUSI"):
            self.root_dir = '/media/ders/sdd1/XS/pipeline/dataset/BUS/Public_BUSI_v2/'
            self.image_dir = self.root_dir + 'img/'
            self.mask_dir = self.root_dir + 'label/'
        
        elif(self.dataset=="PriBUTS"):
            self.root_dir = '/media/ders/XS/dataset/BUSD/Private_BUTS_with_GT/'
            self.image_dir = self.root_dir + 'img/'
            self.mask_dir = self.root_dir + 'label_3/'
        
        elif(self.dataset=="PubDB"):
            self.root_dir = '/media/ders/sdd1/XS/pipeline/dataset/BUS/Public_DB_v2/'
            self.image_dir = self.root_dir + 'img/'
            self.mask_dir = self.root_dir + 'label/'
        
        elif(self.dataset=="PubSTU"):
            self.root_dir = '/media/ders/sdd1/XS/pipeline/dataset/BUS/Public_STU_v2/Hospital/'
            self.image_dir = self.root_dir 
            self.mask_dir = self.root_dir

        self.image_id_list = [image_id.strip() for image_id in open('/media/ders/sdd1/XS/SPCAM_FAMS/data/'+self.dataset+'/%s.txt' % domain).readlines()]
        self.label_list = load_img_label_list_from_npy(self.image_id_list,  '/media/ders/sdd1/XS/SPCAM_FAMS/data/'+self.dataset+'/cls_labels.npy')
        # self.image_id_list = [image_id.strip() for image_id in open('/media/ders/XS/SPCAM_MTKL/data/'+self.dataset+'/%s.txt' % domain).readlines()]

        self.with_id = with_id
        self.with_tags = with_tags
        self.with_mask = with_mask

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        if self.dataset=="PriBUTS":
            image = Image.open(self.image_dir + image_id + '.jpg')

        if self.dataset=="PubBUSI" or "PubDB":
            image = Image.open(self.image_dir + image_id + '.png')
    
        else:
            image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, image_id):
        label = torch.from_numpy(self.label_list[ self.image_id_list.index(image_id)])  
        label=np.insert(label,0,1)
        return label

    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_tags:
            data_list.append(self.get_tags(image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class Dataset_For_Evaluation(Base_Dataset):
    def __init__(self, root_dir, domain, transform=None, _dataset='voc12'):
        super().__init__(_dataset, root_dir, domain, with_tags=True, with_id=True, with_mask=True)
        self.transform = transform
        data = read_json('./data/VOC_2012.json')
        self.class_dic = data['class_dic']
        self.classes = data['classes']
        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name]
                                 for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, label, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image': image, 'mask': mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']

        return image, image_id, label, mask


class Dataset_For_Evaluation_PUBBUSI(Base_Dataset):
    def __init__(self, root_dir, domain, transform=None, _dataset='voc12'):
        super().__init__(_dataset, root_dir, domain, with_tags=True, with_id=True, with_mask=True)
        self.transform = transform
        # data = read_json('./data/VOC_2012.json')
        # self.class_dic = data['class_dic']
        # self.classes = data['classes']
        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name]
                                 for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, label, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image': image, 'mask': mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']

        return image, image_id, label, mask


class Dataset_with_MCGN_ABLA(Base_Dataset):
    def __init__(self, root_dir, pse_dir, domain, transform=None, _dataset='voc12'):
        super().__init__(_dataset, root_dir, domain, with_id=True, with_tags=True)
        self.transform = transform
        self.pse_dir = pse_dir
        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

        data = read_json('./data/VOC_2012.json')
        self.class_dic = data['class_dic']
        self.classes = data['classes']
        self.fn = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __getitem__(self, index):
        image, image_id, label = super().__getitem__(index)
        size = image.size
        # sal_mask = Image.open(self.pse_dir + image_id + '.png')
        oriimg = Image.open(self.pse_dir + image_id + '.jpg').convert('RGB')

        if self.transform is not None:
            input_dic = {'image': image, 'mask': oriimg}
            output_dic = self.transform(input_dic)
            image = output_dic['image']

            sal_mask = output_dic['mask']
            sal_mask = cv2.cvtColor(sal_mask.astype('uint8'), cv2.COLOR_RGB2LAB)

        return image, image_id, label, sal_mask


class Dataset_with_CAM(Base_Dataset):
    def __init__(self, root_dir, pse_dir, domain, transform=None, _dataset='voc12'):
        super().__init__(_dataset, root_dir, domain, with_id=True, with_tags=True)
        self.pse_dir = pse_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

        data = read_json('./data/VOC_2012.json')
        self.class_dic = data['class_dic']
        self.classes = data['classes']
        self.fn = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __getitem__(self, index):
        image, image_id, label = super().__getitem__(index)
        size = image.size
        sal_mask = Image.open(self.pse_dir + image_id + '.png')

        if self.transform is not None:
            input_dic = {'image': image, 'mask': sal_mask}
            output_dic = self.transform(input_dic)
            image = output_dic['image']
            sal_mask = output_dic['mask']

        return image, image_id, label, sal_mask


class Dataset_with_SEG_FULL_PUBBUSI(Base_Dataset):
    def __init__(self, root_dir, pse_dir, domain, transform=None, _dataset='voc12'):
        super().__init__(_dataset, root_dir, domain, with_id=True, with_tags=True)
        self.pse_dir = pse_dir
        self.transform = transform

    def __getitem__(self, index):
        image, image_id, label = super().__getitem__(index)
        size = image.size
        sal_mask = Image.open(self.pse_dir + image_id + '.png')

        if self.transform is not None:
            input_dic = {'image': image, 'mask': sal_mask}
            output_dic = self.transform(input_dic)
            image = output_dic['image']
            sal_mask = output_dic['mask']

        return image, image_id, label, sal_mask


