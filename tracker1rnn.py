import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import pdb
import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import math

def detect(cfgfile, weightfile, video):
    m = Darknet(cfgfile)
    #pdb.set_trace()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))    
    use_cuda = 1
    m.cuda()

    all_boxes = []
    frames = []
    cap = cv2.VideoCapture(video)
    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img,(m.width, m.height) )
            frames.append(sized)
        else:
            break
    frames = frames[:20]
    cv2.imshow(cfgfile,frames[1])    

    for sized in frames:
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        all_boxes.append(boxes)
    return frames, all_boxes


def display_object(track, frames):
    for i, sized in enumerate(frames):
        box = track[i]
        name = None #'prediction%d.jpg' %(i)        
        draw_img = plot_boxes_cv2(sized, [box], name, class_names)
        cv2.imshow(cfgfile, draw_img)
        cv2.waitKey(500)
    
def preprocess(total_boxes, object):
    ###['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ### Get only tracks of correct object and make sure it exists in every frame...
    []
    for i in range(len(total_boxes)):
        for j in range(len(total_boxes[i])):
            if total_boxes[i][j][6] != object:
                total_boxes[i].remove(total_boxes[i][j]) 
    return total_boxes

def dist(boxes_one,boxes_two):
    transition = torch.zeros(len(boxes_one), len(boxes_two))
    for i in range(len(boxes_one)):
        for j in range(len(boxes_two)):
            transition[i][j] = math.log(1-math.sqrt((boxes_one[i][0] - boxes_two[j][0])**2 + (boxes_one[i][1] - boxes_two[j][1])**2))
    return transition

class wordrnn_tracker(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout=0.2):
        super(wordrnn_tracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers=1)
        self.linear = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_s):
        return (Variable(torch.zeros(n_layers,batch_s,hidden_size).cuda()),
         Variable(torch.zeros(n_layers,batch_s,hidden_size).cuda())) 

    def forward(self, detection, hidden):
        output, hidden = self.lstm(detection, hidden)
        output = self.linear(output)
        return output, hidden

def crop_image(img, box):
    width = img.shape[1]
    height = img.shape[0]
    x1 = int(round((box[0] - box[2]/2.0) * width))
    y1 = int(round((box[1] - box[3]/2.0) * height))
    x2 = int(round((box[0] + box[2]/2.0) * width))
    y2 = int(round((box[1] + box[3]/2.0) * height))
    crop_image = img[y1:y2, x1:x2]
    return corp_image

def optical_flow(img1,img2):
    hsv = np.zeros(img1.shape, dtype=np.uint8)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def object_tracker(boxes):
    ##Detect a single object tracker assuming all are detections of same category, that will be specified by the word.
    forward_var = torch.log(torch.Tensor([boxes[0][item][-2] for item in range(len(boxes[0]))]))
    best_tagid = [] ## All best paths
    for t in range(1,len(boxes)):
        transition = dist(boxes[t-1],boxes[t])
        next = forward_var.expand(transition.size()) + transition
        _, tag_id = torch.max(next,0)
        best_tagid.append(tag_id)
        next = torch.sum(next,0)
        #pdb.set_trace()
        forward_var = next + torch.log(torch.Tensor([boxes[t][item][-2] for item in range(len(boxes[t]))])) 
    
    ##Decode path
    path_score , tag_id = torch.max(forward_var,1)
    tag_id = tag_id[0][0]
    best_path = [tag_id]
    for back_t in reversed(best_tagid):
        tag_id = back_t[0][tag_id]
        best_path.append(tag_id)
    
    best_path.reverse()
    track = []
    for i in range(len(best_path)):
        track.append(boxes[i][best_path[i]]) 
    return track , best_path
    
if __name__ == '__main__':

    cfgfile = 'cfg/yolo.cfg'
    weightfile = 'yolo.weights'
    video = 'video1.mov'
    object = 0
    class_names = load_class_names('data/coco.names')

    frames, total_boxes = detect(cfgfile, weightfile, video)
    boxes_per_object = preprocess(total_boxes, object)
    object_track , _ = object_tracker(boxes_per_object)
    display_object(object_track, frames)          
