# Reverse Visual Image and Video Search

 
# Code Setup
 Steps to setup the code:
 1. Clone the repository
 2. Download the data.zip, output.zip and models.zip from [here](https://drive.google.com/drive/folders/1lUt7YG975h2GkoRM88snxsLH1jBgAqIK?usp=sharing), extract these folders and place them in the repo along side the folder **milvus**
 3. Install Milvus using docker as in: [Install Milvus](https://milvus.io/docs/install_standalone-docker.md)
 4. Install **pymilvus** like in [Install pymilvus](https://milvus.io/docs/install-pymilvus.md)
 5. Start **Milvus** using **sudo docker-compose up -d** as done in https://milvus.io/docs/install_standalone-docker.md
 6. Install **streamlit** python library for spinning up the webapp using the command: "pip install streamlit"

For Image Search:  
7. Run "python3 -m streamlit run imgSearchSL.py" to spin up the Reverse Visual Search web app.

For Video Search:  
8. Run "python3 reverseImgSearchMilvusForVideo.py [img-path]". This will return list of videos the system has found a matching frame for the query image.


Other Resources:  
9. The iPython Notebook "FaceNet FineTune" was used to train the pretrained model on the custom dataset "./data/lfw_lite" (a version of LFW that has a smaller number of classes with balanced no. of images across all classes by downsampling and upsampling)


## What is it?

Reverse visual Search is a package of two functionalities: Image search from a) Image Database (ISI) and b) Video Database (ISV). By "Image Search" I mean querying where the query is an image. Given an image we want images/videos in the database that "match" the image. We make this happen we use Image Processing and Deep Learning Techniques and vector databases.

## How in the world?

When we query for example in SQL results that satisfy query conditions are returned; this works on text/numeric data. For an Image this simple equality and regex checks would not suffice. A trick we use here is to convert the images into a representative n-length vector. This vector form of the image would have the property to tell apart similar images and also quanitfy similarity through simple mathematical metrics here it would be the euclidean distance. These vectors of images are called embeddings. Similar images would have less euclidean distance between their embeddings. Now how do we generate these embeddings for an image? Enter Deep Learning. It is a neat trick. We first train a Deep Learning model to classify on face images, the classes being individuals. Once we obtain a good classifier, we remove the last layer of the network. Now when we pass in an image to the network it would give out a vector of n length at the end. This is the mysterious embeddings we were talking about earlier. How would these embeddings capture similarity in the images? Thats the awesomeness of tDeep Learning. The intuition is that when we train the classifier to predict the classes of a persongiven an image of a face wthe classififer would learn to analyse an image and capture the features of a face. Thus the vector outputs of the penultimate layer for similar images (those that would be predicted as the same class) would all be close to each other thus having similar euclidean distance between them would be small.

 For the **Video Search**, the YouTubeFaces dataset did not have videos but rather frames and annotations of the personalities in the frames. Also the dataset was 24GB when zipped making it difficult to work with on the AWS instance. The instance ran out of memory when the dataset was unzipped. So instead of YouTubeFaces dataset we worked with the [UCF dataset] a smaller video dataset. For Video Search System we extracted frames from videos, one frame per second. These frames were then fed into the FacenEt model to obtain the embeddings. These embeddings were then added to the Milvus DB. Each frame had the attributes of frame number in the video as well as the video name and path specified. Thus when given a query image, Milvus would return the vlosest similar frames and their attributes thus allowing us to find the video with the person matching the query image.


## Data

 LFW Dataset: http://vis-www.cs.umass.edu/lfw/
	
## Results

The performance of the system is measured by taking the average of portion of hits in top20 matching results for Reverse Image Search using Milvus using the embeddings as the vectors.

These three metrics are reported in the images shown below:

### Average Portion of Top 20 Images
FaceNet, Average Portion of Top 20 Images Correctly Identified:  
![FaceNet](./Results/Facenet_top20_metric.png?raw=true "FaceNet Top 20 Image Classification Accuracy")

### Reverse Image Search on 10 Test Images  
#### FaceNet, 10 Test Images:
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.39.13%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.39.36%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.39.56%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.42.51%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.43.11%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.45.51%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.46.15%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.46.36%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  
![FaceNet](./Results/facenet_10_test_images/Screen%20Shot%202022-05-02%20at%203.47.02%20PM.png?raw=true "FaceNet Top 20 Image Search Results")  

### Video Search Results:
![VideoSearch](https://github.com/lukezhu1/rev-img-search/blob/85a46b4f256e1a1239728deea9826e7a9f29bcd5/Results/VideoSearch%20Results.png "Video Search Results") 


## References

[1] https://arxiv.org/pdf/1806.08896.pdf
[2] https://ieeexplore.ieee.org/document/9202716
[3] https://reader.elsevier.com/reader/sd/pii/S2666307421000073?token=50E69FDC2DDCB5CA2EDAE1455D7F1F3D4DC78A0D8D22A8027F9D0DC2B09EA8A6515C13C194AB8A5173A7E18463E81051&originRegion=us-east-1&originCreation=20220421232109

