from tkinter import Label
import numpy as np
import os
import glob
import random as rd
from PIL import Image
import matplotlib.pyplot as plt
import json
from skimage.measure import label, regionprops
import geopandas as gpd
from tqdm import tqdm
import pdb
## Class for dataset information extraction
class PASTIS_info:
    def __init__(self,data_root,split):
        self.path = data_root
        self.training = split[0]
        self.validat = split[1]
        self.test = split[2]
        self.data_path = os.path.join(data_root,'DATA_S2')
        self.annot_path = os.path.join(data_root,'ANNOTATIONS')
        self.inst_path = os.path.join(data_root,'INSTANCE_ANNOTATIONS')

        self.meta_patch = gpd.read_file(os.path.join(data_root, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        
        self.data_files = glob.glob(os.path.join(self.data_path,"*.npy"))
        self.data_files.sort()
        self.all_sem = glob.glob(os.path.join(self.annot_path,'TARGET_*.npy'))
        self.all_sem.sort()
        self.all_heat = glob.glob(os.path.join(self.inst_path,'HEATMAP_*.npy'))
        self.all_heat.sort()
        self.all_ins = glob.glob(os.path.join(self.inst_path,'INSTANCES_*.npy'))
        self.all_ins.sort()
        self.all_zones = glob.glob(os.path.join(self.inst_path,'ZONES_*.npy'))
        self.all_zones.sort()
        ### Cheking if all ids have data and corresponding annotation files
        self.target_ids = [f.split('_')[-1].split('.')[0] for f in self.all_sem]
        #self.ids = [f.split('_')[-1].split('.')[0] for f in self.data_files]
        #self.ids = list(set(self.ids).intersection(set(self.target_ids)))
        self.ids = self.target_ids
        #rd.shuffle(self.ids)

        self.catagories = [
                        {'id':int(0),'name':"Background","isthing": 1},
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
                        {'id':int(19),'name':"Void label", "isthing": 1}]
        
        print("All files and labels are loaded")
    
    def save_annot(self,annot,idx,date):
         color = np.zeros(shape=(128,128,3))
         color[:,:,0]= annot%256
         color[:,:,1]= annot//256
         color[:,:,2]= annot//256//256
         annot_name = os.path.join(self.annot_saving_dir,\
                                         os.path.basename(self.all_ins[idx]).\
                                        replace('.npy',f'_{date}.png')) 
         Image.fromarray((color*255).astype(np.uint8),mode='RGB').save(annot_name)
         return np.asarray(color)

    def findidx(self,all_list,id):
        ### Logic is wrong
        idx = None
        count =0
        for f in all_list:
            f_id = f.split('_')[-1].split('.')[0]
            if id ==f_id:
                idx = count
                break 
            else:
                count+=1
        return idx
    
    def data_split(self):
        print("Generating training data")
        self.extract_info(self.ids[0:self.training],\
            os.path.join(self.path,self.path.split(os.sep)[-1]+"_train"),'PASTIS_train')
        print("Generating Validation data")
        self.extract_info(self.ids[self.training:self.training+self.validat],\
            os.path.join(self.path,self.path.split(os.sep)[-1]+"_valid"),'PASTIS_valid')
        print("Generating Test data")
        self.extract_info(self.ids[self.training+self.validat:self.training+self.validat+self.test],\
            os.path.join(self.path,self.path.split(os.sep)[-1]+"_test"),'PASTIS_test')
        
    def extract_info(self,id_list, target_dir, option):
        '''
        if isinstance(self.numbers,int):
            self.ids = rd.sample(self.ids,self.numbers)
        '''         
        if not os.path.exists(os.path.join(self.path,target_dir)):
            os.mkdir(os.path.join(self.path,target_dir))
        self.img_saving_dir = os.path.join(self.path,target_dir)
        ### COCO format has three main dictionary entry Images, Annotations and Catagories
        
        all_images =[]
        all_annot =[]
        catag_format ={}
        all_catag =[]

        for i,id in enumerate(tqdm(id_list)):
            
            idx = self.findidx(self.all_sem,id)
            idx_data = self.findidx(self.data_files,id)
            img_data = np.load(self.data_files[idx_data]).astype(np.float32)
            ## Extracting annotation information
            try:
                sem_annot = np.load(self.all_sem[idx])[0]
            except:
                print(idx)
            ins_annot = np.load(self.all_ins[idx])
            zone_annot = np.load(self.all_zones[idx])
            heat_annot = np.load(self.all_heat[idx])
            dates = dict(self.meta_patch.loc[self.meta_patch['id']==id]['dates-S2'])

            for i,frame in enumerate(img_data):
                image = np.transpose(frame[0:3,:,:],(1,2,0))
                if image.min() <0:
                    image = image + image.min()
                image = image/image.max()*255.0
                image = Image.fromarray(image.astype(np.uint8),mode='RGB')

                file_name = os.path.join(self.img_saving_dir,\
                                         os.path.basename(self.data_files[idx_data]).\
                                        replace('.npy',f'_{dates[int(id)][str(i)]}.jpg'))
                
                image.save(file_name)

                all_images.append({'file_name':file_name,'width':128,'height':128,'id':f"{id}_{dates[int(id)][str(i)]}"})
                all_segment_info =[]
                ### converting instance id to coco type instance ids
                ins_rgb = np.asarray(Image.fromarray(ins_annot.astype(np.uint8),mode='L').convert('RGB'))
                coco_ids = ins_rgb[:,:,0]+256*ins_rgb[:,:,1] + 256*256*ins_rgb[:,:,2]
                coco_ids = np.unique(coco_ids)

                for k,ins_id in enumerate(np.unique(ins_annot)):
                    if ins_id != 0:
                        ins_mask = (ins_annot==ins_id)
                        region = regionprops(ins_mask.astype(np.uint8))
                        x,y = region[-1].centroid
                        h = (ins_annot == ins_id).any(axis=-1).sum()
                        w = (ins_annot == ins_id).any(axis=-2).sum()
                        all_segment_info.append({'id':int(coco_ids[k]),'category_id':int(sem_annot[ins_mask].mean()),\
                            'bbox':(int(x),int(y),int(w),int(h)),'iscrowd':0})
                    else:
                        continue
                
            all_annot.append({'file_name':self.all_ins[idx],\
                              'image_id':f"{id}_{dates[int(id)][str(i)]}",\
                              'segments_info':all_segment_info})
        
        json_data = {'images':all_images,'annotations':all_annot,'categories':self.catagories}
        
        #print(f'Collected data:{all_format}')
        with open(os.path.join(self.path,f'{option}.json'),'w') as jfile:
            json.dump(json_data,jfile)
        return all_images

if __name__=="__main__":
    #extractslices('DATA_S2')
    obj = PASTIS_info('/home/nazib/Data/PASTIS',[2333,0,100])
    data = obj.data_split()
    print("Done")
'''
def findidx(str_list,id):
    idx =0
    count =0
    for x in str_list:
        if id == x.split('_')[1].split('.')[0]:
            idx = count
        else:
            count+=1
    return idx 
       
def normscale(data):
    if data.min()<0:
        data = data + np.abs(data.min())
    data = data/data.max()*255.0
    data = data.astype(np.uint8)
    return data

def extractslices(data_dir):
    if not os.path.isdir(os.path.join(os.getcwd(),"Pastis_images")):
        os.mkdir("Pastis_images")
    
    all_data = glob.glob(os.path.join('DATA_S2','*.npy'))
    idx = rd.randint(0,len(all_data))
    id = all_data[idx].split('_')[2].split('.')[0]
    img = np.load(all_data[idx])
    time_step = img[rd.randint(0,len(img)),:,:,:]
    ### Printing all 10 bands
    count = 0
    for frame in time_step:
        print(f"before Max:{frame.max()} before Min:{frame.min()}") 
        frame = normscale(frame)
        print(f"Max:{frame.max()} Min:{frame.min()}")        
        frame = Image.fromarray(frame).convert('RGB')
        frame.save(os.path.join(os.getcwd(),"Pastis_images",f"IMG_{id}_band_{count}.png"))
        count +=1

    ### Printing all corresponding sementic segmentations
    all_sem = glob.glob(os.path.join('ANNOTATIONS','TARGET_*.npy'))
    all_heat = glob.glob(os.path.join('INSTANCE_ANNOTATIONS','HEATMAP_*.npy'))
    all_ins = glob.glob(os.path.join('INSTANCE_ANNOTATIONS','INSTANCES_*.npy'))
    all_zones = glob.glob(os.path.join('INSTANCE_ANNOTATIONS','ZONES_*.npy'))
    sem_idx = findidx(all_sem,id)
    heat_idx = findidx(all_heat,id)
    ins_idx = findidx(all_ins,id)
    zone_idx = findidx(all_zones,id)

    img = np.load(all_sem[sem_idx])
    img = np.transpose(img,(1,2,0))
    print(f"Max:{img.max()} Min:{img.min()}")
    Image.fromarray((img*255).astype(np.uint8)).convert('RGB').\
        save(os.path.join(os.getcwd(),"Pastis_images",f"Semantics_{id}.png"))
    
    ### Printing corresponding Heatmaps
    img = np.load(all_heat[heat_idx])
    #img = np.transpose(img,(1,2,0))
    cm = plt.get_cmap('hot')
    img = cm(img) 
    print(f"Max:{img.max()} Min:{img.min()}")
    Image.fromarray((img[:,:,:3]*255).astype(np.uint8)).\
        save(os.path.join(os.getcwd(),"Pastis_images",f"Heatmap_{id}.png"))
    
    ### Printing corresponding Instances
    img = np.load(all_ins[ins_idx])
    #img = np.transpose(img,(1,2,0))
    cm = plt.get_cmap('terrain')
    img = cm(img) 
    print(f"Max:{img.max()} Min:{img.min()}")
    Image.fromarray((img[:,:,:3]*255).astype(np.uint8)).convert('RGB').\
        save(os.path.join(os.getcwd(),"Pastis_images",f"Instances_{id}.png"))
    
    ### Printing corresponding Zones
    img = np.load(all_zones[zone_idx])
    #img = np.transpose(img,(1,2,0))
    cm = plt.get_cmap('terrain')
    img = cm(img) 
    print(f"Max:{img.max()} Min:{img.min()}")
    Image.fromarray((img[:,:,:3]*255).astype(np.uint8)).convert('RGB').\
        save(os.path.join(os.getcwd(),"Pastis_images",f"Zones_{id}.png"))
'''     



        


