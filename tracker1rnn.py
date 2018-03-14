import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils_rnn import *
from darknet_rnn import Darknet
import pdb
import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import math
import copy

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

    for sized in frames:
        boxes, convrep = do_detect(m, sized, 0.5, 0.4, use_cuda)
        all_boxes.append(boxes)
    return frames, all_boxes, convrep


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
    for i in range(len(total_boxes)):
        for j in range(len(total_boxes[i])):
            if total_boxes[i][j][6] != object:
                total_boxes[i].remove(total_boxes[i][j]) 
    return total_boxes

def dist(trans_boxes):
    ## This is messy
    ##If it gets len 4 then it is T0: b1xb2  x  T1:b1xb2 but it got it in disorder so it goes 0,2,1,3 
    if len(trans_boxes)==2:
        transition = torch.zeros(len(trans_boxes[0]), len(trans_boxes[1]))
        for i in range(len(trans_boxes[0])):
            for j in range(len(trans_boxes[1])):
                transition[i][j] = math.log(1-math.sqrt((trans_boxes[0][i][0] - trans_boxes[1][j][0])**2 + \
                                                       (trans_boxes[0][i][1] - trans_boxes[1][j][1])**2))
    #else:
    #    transition = torch.zeros(len(trans_boxes[0]), len(trans_boxes[2]), len(trans_boxes[1]), len(trans_boxes[3]))
    #    for i in range(len(trans_boxes[0])):
    #        for j in range(len(trans_boxes[2])):
    #            for x in range(len(trans_boxes[1])):
    #                for y in range(len(trans_boxes[3])):
    #                    transition[i][x][j][y] += math.log(1-math.sqrt((trans_boxes[0][i][0] - trans_boxes[2][j][0])**2 + \
    #                                                                   (trans_boxes[0][i][1] - trans_boxes[2][j][1])**2))
    #                    transition[i][x][j][y] += math.log(1-math.sqrt((trans_boxes[1][x][0] - trans_boxes[3][y][0])**2 + \
    #                                                                  (trans_boxes[1][x][1] - trans_boxes[3][y][1])**2))
    return transition

class wordrnn_tracker(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, batch_s=1, dropout=0.2):
        super(wordrnn_tracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers=1)
        self.linear = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_s=1):
        return (Variable(torch.zeros(n_layers,batch_s,hidden_size).cuda()),
         Variable(torch.zeros(n_layers,batch_s,hidden_size).cuda())) 

    def forward(self, detection, hidden):
        output, hidden = self.lstm(detection, hidden)
        output = self.linear(output)
        output = F.logsigmoid(output)
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

def object_tracker(all, word, convrep, hidden_size):
    detections = []
    size_det = [] 
    for i,object in enumerate(word):
        detections.append(preprocess(all, object)) ##All detections of same category
        size_det.append(len(detections[i][0]))
    forward_var = torch.zeros(size_det)
    ## time 0
    if len(detections)==1:
        forward_var += torch.log(torch.Tensor([boxes[0][item][5] for item in range(len(boxes[0]))])))
    else:    
        for i,boxes in enumerate(detections):
            if i==0:
                forward_aux = torch.log(torch.Tensor([boxes[0][item][5] for item in range(len(boxes[0]))]))).view(-1,1)
            if i==1:
                forward_aux = torch.log(torch.Tensor([boxes[0][item][5] for item in range(len(boxes[0]))])))
            forward_aux.expand(size_det)
            forward_var += forward_aux
    
    ##Assuming at most 2 objects per word
    for i, item in enumerate(detections[0][0]):
        input = convrep[:,:,item[7],item[8]].contiguous().view(-1)
        all_models = dict()
        if len(word)>1:
            for j,item2 in enumerate(detections[1][0]):
                input = torch.cat((input,convrep[:,item2[7],item2[8]].contiguous().view(-1)),0)
                model_rnn = word_rnn_tracker(input.size(),hidden_size)
                model_rnn.cuda()
                model_rnn.zero_grad()
                hidden = model_rnn.init_hidden()
                output, hidden = model_rnn(input,hidden)
                all_models[(i,j)] =  (copy.deepcopy(model_rnn), hidden)
                forward_var[i,j] += output
        else:
            model_rnn = word_rnn_tracker(input.size(),hidden_size)
            model_rnn.cuda()
            model_rnn.zero_grad()
            hidden = model_rnn.init_hidden()
            output, hidden = model_rnn(input, hidden)
            all_models[(i)] = (copy.deepcopy(model_rnn), hidden)
            forward_var[i] += output  


    best_tagid = [] ## All best paths
    for t in range(1,len(all)):
        distances = []
        size_future = []
        for dim in len(detections):
            trans_boxes = []
            trans_boxes.append(detections[dim][t-1])
            trans_boxes.append(detections[dim][t])   
            transition = dist(trans_boxes)
            distances.append(transition)
            size_future.append(transition.size()[1])
        
        if len(detections)==2:
            future_var = torch.zeros(size_future)
            for future_i in future_var.size()[0]:
                for future_j in future_var.size()[1]:
                    current_var = copy.deepcopy(forward_var)
                    for current_i in current_var.size()[0]:
                        for current_j in current_var.size()[1]:
                            current_var[current_i][current_j] += distances[0][current_i][future_i] + \
                                                                 distances[1][current_j][future_j]
                    future_var[future_i][future_j] += current_var.sum()
                    aux ,tag_id_1 = torch.max(current_var,0)
                    _, tag_id_2 = torch.max(aux,0)
                    tag_id = (tag_id_1[tag_id_2],tag_id_2)
                    best_tagid.append(tag_id)

            forward_var = future_var    

        if len(detections)==1: 
            next = forward_var.expand(transition.size()) + transition
            _, tag_id = torch.max(next,0)
            best_tagid.append((tag_id))
            next = torch.sum(next,0)
            #pdb.set_trace()
            forward_var = next 

        ##Add the next detection probabilities
        if len(detections)==1:
       	    forward_var += torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))])))
        else: 
            for i,boxes in enumerate(detections):
       	        if i==0:
                    forward_aux = torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))]))).view(-1,1)         
                if i==1:
       	       	    forward_aux = torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))])))
       	        forward_aux.expand(size_det)
                forward_var += forward_aux

        ##Assuming at most 2 objects per word
        for i,item in enumerate(detections[0][t]):
            input = convrep[:,:,item[7],item[8]].contiguous().view(-1)
            all_models_new = dict()
            if len(word)>1:
                for j,item2 in enumerate(detections[1][t]):
                    input = torch.cat((input,convrep[:,item2[7],item2[8]].contiguous().view(-1)),0)
                    output, hidden = all_models[tag_id][0](input,all_models)
                    all_models[(i,j)] =  (copy.deepcopy(model_rnn), hidden)
                    forward_var[i,j] += output
            else:
                model_rnn = word_rnn_tracker(input.size(),hidden_size)
                model_rnn.cuda()
                model_rnn.zero_grad()
                hidden = model_rnn.init_hidden()
                output, hidden = model_rnn(input, hidden)
                all_models[(i)] = (copy.deepcopy(model_rnn), hidden)
                forward_var[i] += output




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
    hidden_size = 100
    lexicon = dict()
    lexicon['chase'] = [0,0]
    

    class_names = load_class_names('data/coco.names')

    frames, total_boxes, convrep = detect(cfgfile, weightfile, video)
    
    object_tracks, _  = object_tracker(boxes_per_object, lexicon['chase'], convrep, hidden_size)

    display_object(object_tracks, frames)


