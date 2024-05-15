import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img
class ChannelExchange1(object):
   
    def __init__(self, gray = 2):
        pass
    def __call__(self, img):
        tmp_img = 0.3333 * img[0,:,:] + 0.3333 * img[1,:,:] + 0.3333 * img[2,:,:]
        img[0,:,:] = tmp_img
        img[1,:,:] = tmp_img
        img[2,:,:] = tmp_img
        return img
        
class PicAugmentation(object):
    def __init__(self, gray=2,layer=6,interval=1):
        self.gray = gray
        self.layer = layer
        self.interval = interval
    def __call__(self, img):
        part_len = img.size(1)//self.layer
        for i in range(self.layer//(self.interval+1)):
            idx = random.randint(0, self.gray)
            if idx==0:
                img[1,part_len*2*i:part_len*(2*i+1),:]=img[0,part_len*2*i:part_len*(2*i+1),:]
                img[2,part_len*2*i:part_len*(2*i+1),:]=img[0,part_len*2*i:part_len*(2*i+1),:]
            elif idx==1:
                img[0,part_len*2*i:part_len*(2*i+1),:]=img[1,part_len*2*i:part_len*(2*i+1),:]
                img[2,part_len*2*i:part_len*(2*i+1),:]=img[1,part_len*2*i:part_len*(2*i+1),:]
            elif idx==2:
                img[1,part_len*2*i:part_len*(2*i+1),:]=img[2,part_len*2*i:part_len*(2*i+1),:]
                img[0,part_len*2*i:part_len*(2*i+1),:]=img[2,part_len*2*i:part_len*(2*i+1),:]
        return img
    

        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None,size=(288,144)):
        data_dir = '../datasets/sysu/npy/'
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])
        self.transform_color2 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            PicAugmentation()])    
        self.transform_mpanet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10, fill=127), 
            transforms.RandomCrop(size),
            transforms.ToTensor(), 
            normalize,
            ChannelRandomErasing(probability = 0.5)
        ])
       
    def __getitem__(self, index):
        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1_0 = self.transform_color(img1) 
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2) 
        img1_2 = self.transform_color2(img1)
        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None,size=(128,384)):
        data_dir = '../datasets/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize(size, Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize(size, Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((size[1],size[0])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10, fill=127), 
            transforms.RandomCrop((size[1],size[0])),
            transforms.ToTensor(), 
            normalize,
            ChannelRandomErasing(probability = 0.5)
        ])

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1_0 = self.transform(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform(img2)


        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
class Dataloader_MEM(data.Dataset):
    def __init__(self, data_dir,  dataset=None, size=(288,144)):
        self.train_color_label = dataset.train_color_label
        self.train_thermal_label = dataset.train_thermal_label
        self.train_color_image   = dataset.train_color_image
        self.train_thermal_image = dataset.train_thermal_image
        self.choose = 0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_center = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
            ChannelExchange(gray = 2)])   
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize])
       
    def __getitem__(self, index):
        if self.choose==0:
            img1, target1 = self.train_color_image[index],  self.train_color_label[index]
            img1_0 = self.transform_test(img1)
            return img1_0, target1
        elif self.choose==1:
            img2,  target2 = self.train_thermal_image[index], self.train_thermal_label[index]
            img2 = self.transform_test(img2)
            return img2, target2
        elif self.choose==2:
            img1, target1 = self.train_color_image[index],  self.train_color_label[index]
            img1_1 = self.transform_center(img1)
            return img1_1, target1

    def __len__(self):
        if self.choose==0 or self.choose==2:
            return len(self.train_color_label)
        elif self.choose ==1:
            return len(self.train_thermal_label)
        
