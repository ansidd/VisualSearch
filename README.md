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

When we query for example in SQL we satisfy query conditions that work on text/numeric data. For an Image this simple equality and regex checks would not suffice. A trick we use here is to conver the images into a n-length vector. This vector form of the image would have the property to tell apart similar images and also quanitfy similarity through simple mathematical metrics here it would be the euclidean distance. These vectors of images are called embeddings. Similar images would have less euclidean distance between their embeddings. Now how do we generate these embeddings for an image? Enter Deep Learning. It is a neat trick. We first train a Deep Learning model to classify on face images, the classes being individuals. Once we obtain a good classifier, we remove the last layer of the network. Now when we pass in an image to the network it would give out a vector of n length at the end. This is the mysterious embeddings we were talking about earlier. How would these embeddings capture similarity in the images? Thats the awesomeness of tDeep Learning. The intuition is that when we train the classifier to predict the classes of a persongiven an image of a face wthe classififer would learn to analyse an image and capture the features of a face. Thus the vector outputs of the penultimate layer for similar images (those that would be predicted as the same class) would all be close to each other thus having similar euclidean distance between them would be small.

## 1. Introduction

 The original idea of the project was to be able to take an already existing reverse image search model and find ways to improve it. The CNN is trained on the LFW dataset and generates a feature vector, or embedding, for each image. The embeddings are then stored in a vector database like knn Elastic Search or Milvus. A user can search for similar images by inputting an image query into the model to generate a new embedding, then search the vector database for the image with the nearest embedding calculated by Euclidean distance. In order to understand how face recognition works and what makes certain models perform better than others, we tested a baseline ResNet50 model pre-trained on ImageNet against FaceNet. 

## 2. Related Work

### Elastic Search

 Using a query of vectors that come from each input image we are able to send it into a function that calculates a similarity value between other vectors. We call these vectors “embeddings”. Elastic search is usually done with the nearest neighbors method which is a framework that extracts n closest points to the original point passed in for understanding. There are two phases that go into searching the image based on the embeddings created [1]. This refers to the retrieval of features and the ranking of them as well. Using a ranking system we can easily identify how prominent a feature is on a face to add more weightage to it.

### Evaluation of Google Reverse Image Search

 Seeing as Google is retrospectively world widely popular, an analysis of performance would prove to be beneficial in understanding how models induce feature detection. Looking at the results of different types of images fed into the Google model, we can deduce it performs better on certain types of images. Although such a popular model it seems that there is some “[trouble recognizing] the same images” [2]. The study done to conclude this was using simple images of objects and uploading them to the reverse search bar then clicking on the similar images button in order to understand what the model identifies in each image.

### Facial Expression with Resnet50

 Facial recognition tests are done to identify the emotions of a person. This requires the same intensity of facial feature extraction and classification as facial matching as it needs to use a clustering algorithm to find similar faces as the picture being passed as input. Using a CNN a kernel is able to be “convolved with the input image” which allows a feature map to be created [3]. The study also uses Resnet50 and deduces a feature pattern that can be extracted through the use of increased layers [3].


## 3. Data

 We are using a photo database filled with images of peoples’ faces. We are using the LFW Dataset, which contains 13,233 images of 5749 people. We decided to preprocess the dataset using MTCNN so we can effectively extract the faces. The crop allows us to have just the face take up most of the image space which allows us to have less of a background to distract the model in the case there are other prominent features in the background.

## 4. Methods

 In order to analyze an input photo, we use deep neural networks to create a vector called an embedding for each image in the image database LFW. These embeddings are stored in a vector database. The input photo is also converted into an embedding by the same network, and that embedding is compared against each embedding stored in the vector database. Our definition for similar embeddings is based on Euclidean distance. The shorter the distance between two embeddings, the more similar the two images are.  

 The embeddings are numerically real-valued vectors. In our case, we used vectors of 1024 elements. These embeddings are abstractions of images that are used to characterize and quantify features in an image, and more specifically, faces. The transformation of an RGB image into a single vector is a type of dimension reduction. The resulting embedding vectors can be used to quantify similarity between any two images by calculating the Euclidean distance of the two vectors.  

![Equation](https://latex.codecogs.com/svg.image?\bg{white}d(p,q)&space;=&space;\sqrt{\sum_{i}^{n}&space;(q_i&space;-&space;p_i)^2})
$$ d(p,q) = \sqrt{\sum_{i}^{n} (q_i - p_i)^2} $$ 

 There are several options for the vector database, such as Elastic Search’s elasticknn, Weaviate, or Milvus. For our project we went ahead and used milvus due to its robustness. The vector databases allow storage of our image embeddings and fast searching. Milvus uses a Euclidean distance metric to return the most similar results. 

 We also explored two types of mathematical formulas in order to decide which would produce the best results. After exploring the cosine and euclidean distance equations when determining value between the embeddings, we found that euclidean should perform better because it is actually used for image detection whereas cosine is usually used for text analysis problems.  

 Our baseline model is a ResNet50 CNN model that is pre-trained on ImageNet, an image database of hundreds of thousands of labeled images. In order to apply this model to the new dataset, LFW, we applied transfer learning on the model and retrained it with LFW. This model then generates a feature vector for each image in the LFW database, which is stored in a vector database for later retrieval and comparisons. A queried image is fed into the model to create its own embedding, which is used to search for the most similar embedding by Euclidean distance, and thus, select the most similar image.  

 We also decided to use Facenet as our improved model because of the idea of the Siamese Network. The parallel neural network allows for the verification of the face to be done in  a more efficient manner. We calculate the distance between two feature vectors in order to get to our target which is which person the face really belongs to. By using this, we can take any two images and label them as either identical or different in order to simplify the similarities between the two faces being compared against.   For Facenet we actually adjusted the model by removing the last layer, this allows us to actually search the image in question with the rest of the database for similar features as opposed to leaving the last layer and just assigning a label to each image, in our case being a name attached to the face. The reasoning behind removing the last layer includes wanting to actually train the weights and also use a more dense layer for the transfer learning.  
 
 For the **Video Search**, the YouTubeFaces dataset did not have videos but rather frames and annotations of the personalities in the frames. Also the dataset was 24GB when zipped making it difficult to work with on the AWS instance. The instance ran out of memory when the dataset was unzipped. So instead of YouTubeFaces dataset we worked with the [UCF dataset] a smaller video dataset. For Video Search System we extracted frames from videos, one frame per second. These frames were then fed into the FacenEt model to obtain the embeddings. These embeddings were then added to the Milvus DB. Each frame had the attributes of frame number in the video as well as the video name and path specified. Thus when given a query image, Milvus would return the vlosest similar frames and their attributes thus allowing us to find the video with the person matching the query image.

	

## 5. Experiments

To improve the performance of the models we attempted a number of experiments:
1. FaceNet: The baseline model that we used was a ResNet50 Neural Network. This model gave decent performance. FaceNet is a model that performs better on face data related tasks. We took a pretrained FaceNet model that was trained on VGGFace2 dataset. It performs better than Resnet50.
2. Dataset Augmentation: We observed that the dataset was imbalanced. Majority of the classes had one image while there were classes that had 500 images. We augmented the dataset by applying transformations such as grayscaling, rotation, flipping and duplicating. 
3. Downsampling: Since the dataset is really big, processing and training the models was taking up significant time. We reduced the size of the dataset to have 10 samples per class. This avoided bias in the model towards classes with higher samples.
4. Transfer Learning: The ResNet50 model was trained on the image database ImageNet, whereas the FaceNet model was trained on the standard dataset VGGFace2. We wanted to improve the performance of the models by training the models on the LFW dataset with the pretrained weights and transfer learning. This did not result in a better accuracy. The learning was done as an Image Classification task.

## 6. Results

We calculated three metrics to evaluate the performance of the baseline Resnet50 Model and the improved FaceNet model:  

1. Accuracy of image classification on test set using KNN model with the embeddings generated by the model.
2. Average of portion of hits in top20 matching results for Reverse Image Search using Milvus using the embeddings as the vectors.
3. Result of Reverse Image Search on the 10 Test Images specified in the problem statement
These three metrics are reported in the images shown below:

### Image Classificatino Accuracy  
ResNet50 Image Classification Accuracy:  
![ResNet50](./Results/Resnet50_Classification_acc.png?raw=true "ResNet50 Image Classification Accuracy")

FaceNet Image Classification Accuracy:  
![FaceNet](./Results/Facenet_KNN_Classification_acc.png?raw=true "FaceNet Image Classification Accuracy")

### Average Portion of Top 20 Images
ResNet50, Average Portion of Top 20 Images Correctly Identified:  
![ResNet50](./Results/Resnet_50_Em_top20_Metric.png?raw=true "ResNet50 Top 20 Image Classification Accuracy")

FaceNet, Average Portion of Top 20 Images Correctly Identified:  
![FaceNet](./Results/Facenet_top20_metric.png?raw=true "FaceNet Top 20 Image Classification Accuracy")

### Reverse Image Search on 10 Test Images  
#### ResNet50, 10 Test Images:  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.18.12%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.18.44%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.19.18%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.19.56%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.20.39%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.21.42%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.22.18%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.22.55%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  
![ResNet50](./Results/resnet50_10_test_images/Screen%20Shot%202022-05-02%20at%204.23.26%20PM.png?raw=true "ResNet50 Top 20 Image Search Results")  

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




## 7. Conclusion

 We developed a Reverse Visual Search system that would return top 20 similar images given a query image. We used the embeddings generated by neural network models and a vector search database that uses these embeddings. The baseline neural network, a ResNet50 model, and the superior model, FaceNet, were selected for this task. We experimented with a number of techniques to improve the performance of these models. The results of the system with the two different models in the backend was shown above. Overall, FaceNet showed considerable improvement over the ResNet50 baseline. 


## References

[1] https://arxiv.org/pdf/1806.08896.pdf
[2] https://ieeexplore.ieee.org/document/9202716
[3] https://reader.elsevier.com/reader/sd/pii/S2666307421000073?token=50E69FDC2DDCB5CA2EDAE1455D7F1F3D4DC78A0D8D22A8027F9D0DC2B09EA8A6515C13C194AB8A5173A7E18463E81051&originRegion=us-east-1&originCreation=20220421232109

