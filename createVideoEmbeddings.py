from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
from sklearn.model_selection import train_test_split
import argparse

import cv2
import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from numpy import save
import numpy as np
from tqdm import tqdm

import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="enter 'default' if no model else enter path") 

args = parser.parse_args()
model_to_load = args.model

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=250, margin=40)
mtcnn=mtcnn.eval()

total_num_images=0

if model_to_load=='default':
# Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    print("Loading default model")
else:
    print("Loading model {0}".format(model_to_load))
    with open(model_to_load, 'rb') as f:
        resnet = torch.load(f)
        resnet.classify=False
        resnet.eval()


with open('./config.json') as f:
    config = json.load(f)

#Read the images
train_data_path = config['paths']['data']['video']['train']
print("Training Path {0}".format(train_data_path))
output_path = config['paths']['output']
print("Output Path {0}".format(output_path))

files = os.listdir(train_data_path)
paths = []
video_info = []

limit_videos = bool(config['limit_videos'])
count = config['videos_count']

if limit_videos:
    print("Limiting the number of videos to process to: {0}".format(count))
else:
    print("Processing all videos")
    
counter = 0

for video in tqdm(files):
    video_dict = {}
    video_dict['name'] = video
    video_dict['frames'] = []
    video_cap = cv2.VideoCapture(train_data_path+video)
    success=True
    frames = []
    frame_nums = []
    framerate = video_cap.get(cv2.CAP_PROP_FPS)
    resolution = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_counter = 0
    while(True):
        success, frame = video_cap.read()
        if not success:
            break
        if frame_counter%framerate==0:
            frames.append(frame)
            frame_nums.append(frame_counter+1)
        frame_counter+=1
        

    
    if limit_videos:
        if counter==count:
            break

    counter+=1


    print("Image set of length {0} for {1}".format(len(frames), video))

    embeddings = []

    pre_frames = frames
    frames = []
    pre_frame_nums = frame_nums
    frame_nums = []

    with torch.no_grad():
        for index,img in tqdm(enumerate(pre_frames)):

        # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(img)
            
            if(img_cropped!=None):
                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = resnet(img_cropped.unsqueeze(0))
                embeddings.append(img_embedding)
                frame_nums.append(pre_frame_nums[index])
            

    embeddings = [i[0].detach().numpy() for i in embeddings]
    del pre_frames
    video_dict['embeddings'] = embeddings
    video_dict['frame_num'] = frame_nums
    total_num_images+= len(video_dict['embeddings'])
    video_info.append(video_dict)

print("No. of videos parsed: {0}".format(len(video_info)))
print("Total no. of images gathered: {0}".format(total_num_images))

with open("video_database.pkl", "wb") as f:
    pickle.dump(video_info, f)




