'''
Note: All images should be aligend first
'''
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from mymodel import preprocess_input
import cfg

class CarvanaReader(object):
    def __init__(self, dir_images):
        self.root = dir_images
        classes = os.listdir(self.root)
        self.classes = [c for c in classes if len(listdir(join(self.root, c)))>1]
        self.idx = range(len(self.classes))
                        
    def GetTriplet(self):
        idx_class_pos = np.random.choice(self.idx, 1)[0]
        name_pos = self.classes[idx_class_pos]
        dir_pos = join(self.root, name_pos)
        [idx_img_anchor, idx_img_pos]= np.random.choice(range(len(listdir(dir_pos))), 2, replace=False)
            
        # negative classes are selected from all folders
        while True:
            idx_class_neg = np.random.choice(self.idx, 1)[0]
            if idx_class_neg != idx_class_pos:
                break    
            
        name_neg = self.classes[idx_class_neg]
        dir_neg = join(self.root, name_neg)
        idx_img_neg = np.random.choice(range(len(listdir(dir_neg))), 1)[0]

        path_anchor = join(dir_pos, listdir(dir_pos)[idx_img_anchor])
        path_pos = join(dir_pos, listdir(dir_pos)[idx_img_pos])
        path_neg = join(dir_neg, listdir(dir_neg)[idx_img_neg])
        
        return path_anchor, path_pos, path_neg
     
class MixedReader(object):
    def __init__(self, list_readers):
        self.readers = list_readers
        
    def GetTriplet(self):
        idx = np.random.randint(low=0, high=len(self.readers))
        path_anchor, path_pos, path_neg = self.readers[idx].GetTriplet()
        return path_anchor, path_pos, path_neg
        
def ReadAndResize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((cfg.image_size, cfg.image_size))
    return np.array(im, dtype="float32")

# create triplet example from LFW dataset
def TripletGenerator(reader):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(cfg.batch_size):
            path_anchor, path_pos, path_neg = reader.GetTriplet()
            img_anchor = ReadAndResize(path_anchor)
            img_pos = ReadAndResize(path_pos)
            img_neg = ReadAndResize(path_neg)
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None
        
        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label)        

def ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()
    
def TestTripletGenerator(reader):  
    # reader = LFWReader(dir_images='E:\\DM\\VGG-Face\\aligned')
    gen = TripletGenerator(reader)
    data = next(gen)
    imgs_anchor = data[0]['anchor_input']
    imgs_pos = data[0]['positive_input']
    imgs_neg = data[0]['negative_input']
    print(imgs_anchor.shape)
    print(imgs_pos.shape)
    print(imgs_neg.shape)
    
    imgs_anchor += 1
    imgs_anchor *=127.5
    imgs_pos += 1
    imgs_pos *=127.5
    imgs_neg += 1
    imgs_neg *=127.5
    
    for idx_img in range(cfg.batch_size):
        anchor = imgs_anchor[idx_img]
        pos = imgs_pos[idx_img]
        neg = imgs_neg[idx_img]

        ShowImg(anchor)
        ShowImg(pos)
        ShowImg(neg)
        break

def TestCarvana():
    reader = CarvanaReader(dir_images='E:\\DM\\Udacity\\Carvana\\Data\\aligned\\train')
    TestTripletGenerator(reader)
    
 
if __name__=='__main__':
    TestCarvana()