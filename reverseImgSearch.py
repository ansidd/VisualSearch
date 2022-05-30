from sklearn.neighbors import KNeighborsClassifier
import json
import cv2
import torch
import pandas as pd
import numpy as np
import sys



def search(img_path):
    with open('./config.json') as f:
        config = json.load(f)

    output_path = config['paths']['output']

    embeddings_train = np.load(output_path+'embeddings_train.npy', allow_pickle=True)
    labels_w_path_train = np.load(output_path+ 'labels_train.npy', allow_pickle=True)


    labels_train = labels_w_path_train[:,0]
    n_classes = len(set(labels_train))

    knnc = KNeighborsClassifier(n_neighbors=15)
    knnc.fit(embeddings_train, labels_train)

    
    img = cv2.imread(img_path)

    mtcnn = torch.load(config['paths']['models']['mtcnn'])
    mtcnn.eval()
    resnet = torch.load(config['paths']['models']['resnet'])
    resnet.eval()

    print('KNN Classifier is used to predict the classes of the images')

    if img_path=='test':
        embeddings_test = np.load(output_path+'embeddings_test.npy', allow_pickle=True)
        labels_w_path_test = np.load(output_path+'labels_test.npy', allow_pickle=True)

        labels_true = labels_w_path_test[:,0]
        labels_predicted = knnc.predict(embeddings_test)

        acc = (labels_predicted==labels_true).mean()
        print("Accuracy:", acc )
        return acc, labels_predicted
    else:
        try:
            img_cropped = mtcnn(img)
            img_embedding = resnet(img_cropped.unsqueeze(0))
            pred = knnc.predict(img_embedding.detach().numpy())
            print("Results: {0}".format(pred))
            return 0, [pred]
        except Exception as e:
            print("Error while detecting face in the image: {0}".format(e))
            return None, None
        
def main():
    img_path = sys.argv[1]
    search(img_path)
    
if __name__=="__main__":
    main()
