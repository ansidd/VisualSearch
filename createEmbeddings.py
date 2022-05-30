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



def create_embeddings(train_data_path, output_path, mtcnn, resnet, args, config, train=True):

    folders = os.listdir(train_data_path)
    labels = []
    paths = []
    images = []

    limit_images = bool(config['limit_images'])
    count = config['images_count']

    if limit_images:
        print("Limiting the number of images to: {0}".format(count))
    else:
        print("Processing with all images")

    counter = 0

    for person in tqdm(folders):
        files = os.listdir(train_data_path+person)
        if limit_images and train:
            if counter==count:
                break
        #if len(files)<4:
            #print("Skipped {0} because class had less than 4 images".format(person))
            #continue
        for img in files:
            try:
                images.append(cv2.imread(train_data_path+person+'/'+img))
            except Exception as e:
                print(e)
                continue
            labels.append(person)
            paths.append(train_data_path+person+'/'+img)
            counter+=1
            if limit_images and train:
                if counter==count:
                    break


    print("Image set of length: ", len(images))

    embeddings = []
    pre_labels = labels
    labels = []

    pre_images = images
    images = []

    pre_paths = paths
    paths = []

    with torch.no_grad():
        for index,img in tqdm(enumerate(pre_images)):

          # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(img)

            if(img_cropped!=None):
                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = resnet(img_cropped.unsqueeze(0))
                labels.append(pre_labels[index])
                images.append(pre_images[index])
                paths.append(pre_paths[index])
                embeddings.append(img_embedding)

            else:
                print("No face detected in {0} skipping image".format(pre_paths[index]))
                #img = cv2.resize(img, (250,250))
                #img = np.moveaxis(img, -1, 0)
                #img = torchvision.transforms.functional.to_tensor(img)
                #img_embedding = resnet(img.unsqueeze(0))
                #For images where face is not detected


    X = [i[0].detach().numpy() for i in embeddings]

    df = pd.DataFrame({'embeddings':X,'label':labels})
    label_counts = df['label'].value_counts()
    #valid_labels = list((label_counts[label_counts>2]).index)
    #valid_indices = df['label'].isin(valid_labels)

    #print("Total no. of valid labels: {0}".format(len(set(valid_labels))))
    #print(type(valid_labels),valid_labels)

    X = np.stack(X, axis=0)
    #X= X[valid_indices, :]

    labels = np.array(labels)
    paths = np.array(paths)
    #paths = paths[valid_indices]
    #labels = labels[valid_indices]

    labels_w_paths = np.stack([labels, paths], axis = 1)

    print("No. of valid face images: ", len(df['label']))
    print('Size of embedding for an image: ', df['embeddings'].iloc[0].shape)

    #X_train, X_test, y_train, y_test = train_test_split(X, labels_w_paths, test_size=0.3, stratify=labels, shuffle=True)

    if train:
        save(output_path+'embeddings_train.npy', X)
        save(output_path+'labels_train.npy', labels_w_paths)
    else:
        save(output_path+'embeddings_test.npy', X)
        save(output_path+'labels_test.npy', labels_w_paths)
        
def start():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="enter 'default' if no model else enter path") 

    args = parser.parse_args()
    model_to_load = args.model

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=250, margin=40)
    mtcnn=mtcnn.eval()

    if model_to_load=='default':
    # Create an inception resnet (in eval mode):
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
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
    train_data_path = config['paths']['data']['train']
    test_data_path = config['paths']['data']['test']
    
    print("Training Path {0}".format(train_data_path))
    output_path = config['paths']['output']
    print("Output Path {0}".format(output_path))
    
    create_embeddings(train_data_path, output_path, mtcnn, resnet, args, config, train=True)
    create_embeddings(test_data_path, output_path, mtcnn, resnet, args, config, train=False)
    

    torch.save(mtcnn, config["paths"]["models"]["mtcnn"])
    print("MTCNN model saved at {0}".format(config["paths"]["models"]["mtcnn"]))
    torch.save(resnet, config["paths"]["models"]["resnet"])
    print("Resnet model saved at {0}".format(config["paths"]["models"]["resnet"]))

if __name__=="__main__":
    start()

