import numpy as np
from scipy import misc
from .base import GetModel, DetectVehicle 
from .common import Inputs2ArrayImage, SelectLargest

def _LoadAndAlign(input, model, image_size, debug=False):
    img = Inputs2ArrayImage(input)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes = DetectVehicle(img, model)
    if bounding_boxes is None:
        # print('Unable to align')
        return
    #det = bounding_boxes[0]
    idx = SelectLargest(bounding_boxes, img_size)
    bb = bounding_boxes[idx, 0:4]
    bb = bb.astype(int)
    if not debug:
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned_images = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        return  aligned_images
    else:
        import cv2
        thick = 5
        color = (200,200,0)
        draw = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, thick)
        return draw

def GetAlignFuncByBoundingBox(output_size=224):
    m = GetModel()
    return lambda input : _LoadAndAlign(input, model=m, image_size=output_size)
    