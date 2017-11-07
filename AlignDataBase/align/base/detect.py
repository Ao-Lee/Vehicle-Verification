import numpy as np
import cv2

YOLO_input_size = 448
classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train","tvmonitor"]
                
class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);
             
def YoloOut2Boxes(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=7):
    class_vehicle = 6
    boxes = []
    SS        =  S * S # number of grid cells
    prob_size = SS * C # class probabilities
    conf_size = SS * B # confidences for each grid cell
    
    probs = net_out[0 : prob_size]
    confs = net_out[prob_size : (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size) : ]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])
    
    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c =  confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid %  S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w =  cords[grid, b, 2] ** sqrt 
            bx.h =  cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c
            
            
            if p[class_vehicle] >= threshold:
                bx.prob = p[class_vehicle]
                boxes.append(bx)
            
    boxes.sort(key=lambda b:b.prob,reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]
    
    return boxes

def Boxes2BB(boxes, h, w):
    if len(boxes)==0:
        return 
    
    bb_box = np.zeros(shape=[len(boxes), 5])
    for i, b in enumerate(boxes):
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        
        
        # left, right, top, bot = Square(left, right, top, bot, h, w)
        bb_box[i,0] = left
        bb_box[i,2] = right
        bb_box[i,1] = top
        bb_box[i,3] = bot
        bb_box[i,4] = b.prob
    return bb_box
    
# enlarge each box to a squared shape
def Square(left, right, top, bot, max_h, max_w):
    w = right - left
    h = bot - top
    diff1 = np.abs((w-h)//2)
    diff2 = np.abs(w-h) - diff1
    left = left if w > h else left - diff1
    right = right if w > h else right + diff2
    top = top if h > w else top - diff1
    bot = bot if h > w else bot + diff2
    
    #check if box is off the margin
    
    if left < 0:
        margin = 0 - left
        left +=margin
        right += margin
        
    if right > max_w:
        margin = right - max_w
        left -= margin
        right -= margin
        
    if top < 0:
        margin = 0 - top
        top += margin
        bot += margin
    
    if bot > max_h:
        margin = bot - max_h
        top -= margin
        bot -= margin
        
    return left, right, top, bot
    
# Crop the origin image so that the output is a square   
def CropImage(image):
    assert len(image.shape)==3
    h, w, _ = image.shape
    diff1 = np.abs(w-h)//2
    diff2 = np.abs(w-h) - diff1

    w_begin = 0 if w < h else diff1
    w_end = w if w < h else w - diff2
    h_begin = 0 if h < w else diff1
    h_end = h if h < w else h - diff2
    
    image_cropped = image[h_begin:h_end, w_begin:w_end, :]
    offset_w = diff1 if w > h else 0
    offset_h = diff1 if h > w else 0
    return image_cropped, [offset_h, offset_w]

  
# Resize the image so that it fits the network inputs
# it is assumed that the input image is a square image
def ResizeImage(image, size_target=YOLO_input_size):
    assert len(image.shape)==3
    assert image.shape[0] == image.shape[1]
    size_origin = image.shape[0]
    ratio = size_origin/size_target
    image_resized = cv2.resize(image,(size_target,size_target))
    return image_resized, ratio
    
def DetectVehicle(image, model, threshold=0.17):
    image_squared, [offset_h, offset_w] = CropImage(image)
    image_resized, ratio = ResizeImage(image_squared)
    batch = np.transpose(image_resized,(2,0,1))
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = YoloOut2Boxes(out[0], threshold = threshold)
    
    if len(boxes)==0:
        return
        
    bb = Boxes2BB(boxes, YOLO_input_size, YOLO_input_size)
    # cordinates of bb is based on resized image of shape(448,448)
    # need to convert the cordinates so that it is based on original image
    bb[:,:-1] *= ratio
    bb[:,[0, 2]] += offset_w
    bb[:,[1, 3]] += offset_h

    for i in range(bb.shape[0]):
        left, right, top, bot = Square(bb[i,0], bb[i,2], bb[i,1], bb[i,3], image.shape[0], image.shape[1])
        bb[i,0] = left
        bb[i,2] = right
        bb[i,1] = top
        bb[i,3] = bot
    return bb
    

# if __name__=='__main__':
def Test():
    from model import GetModel
    from os.path import join
    import matplotlib.pyplot as plt
    
    model = GetModel()
    root = 'E:\\DM\\Udacity\\Public Security\\TieLing_tmp'
    name = '701214002172473803010830164_铁岭良工阀门门口北向南（铁抚街）.2018-02-12 133228.-.X99.Z.99.4.33.jpg'
    imagePath = join(root, name)
    image = plt.imread(imagePath)
    bb = DetectVehicle(image, model)
    bb = bb[0].astype(np.int)
    cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax1.imshow(image)
    ax2.imshow(cropped)
    

    

