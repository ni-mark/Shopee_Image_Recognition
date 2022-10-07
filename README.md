# Shopee_Image_Recognition
Capstone 3 Project for Springboard. Image recognition and categorization project using python PyTorch package.

Motivation 

Shopee is an e-commerce brand with a lowest-price guarantee program and is looking to improve product categorization and detect marketplace spam through image and text recognition. This project focuses on the image recognition side of product matching. The goal is to build a model that predicts and categorizes images into their designated label groups. To create image classification models to detect label grouping PyTorch is used to create and initiate the neural network. 

Data

The data comprised of two image folders, train and test, and two csv files corresponding to each folder. The train dataset contains 32412 unique images, with 5 columns: posting id, image, image phash, title, and label group. The test dataset contains 3 images with the same columns excluding label group, as it is the target prediction. 

Exploratory Data Analysis (EDA)

Given the large number of images and unique label groups, only the train dataset was used and split into test, train, and validation sets during the ‘Prepare Data for Modeling’ section. 
In the train dataset, image shape was consistent across images at (1024,1024,3) in RGB.
The EDA consisted of exploring each column in relationship to other columns and images. 

Label Group

There are 11,014 unique label groups. The largest label group contained 51 images. 7 unique label groups had the maximum of 51 images. The majority of label groups, 63%, contained 2 images. When visualizing a max label group, it contained duplicate and similar images with various image automations and different images but with the same product. There are also images of with the same product in different marketing campaigns/styles.

Data Cleaning 

As the target prediction is label group, the size of 11,014 unique classes in label group is of concern as it will take too long for google colab pro to process all the images and classes in the neural network. This concern is compounded by the majority of label groups is comprised of 2 images. Therefore, in order to build a model for image classification on PyTorch the train image dataset was reduced to the top 21 label groups. The reduced dataset comprised of 899 images, of which 72 were duplicates and were not included as their image and paths were identical, thus leaving a total of 827 images to train, validate, and split. 

Prepare Data for Modeling 

With my data mounted to the colab notebook, the first step was to create a list of image paths to the image folder. Afterwards a data frame containing just the image name and label groups was created. This was done to map each label group (class) to an associated index . The map is used to generate predictions later with the associated label group. 

The first function that was made was to initialize and generate a PyTorch dataset and to feed it into a data loader. Specifically this functions calls an image with its associated label group and converts to the image to PIL, which is needed for tensor data transformations. The image is saved as the input transformed as a tensor. Next the associated label group is saved as the output and converted into a tensor. Both the input and the output are sent to the device, which in this case is GPU using Colab Pro. Furthermore, in this section, it is possible to toggle image augmentation, which would randomly arguments an image with a 50% change in either a random grayscale, horizontal or vertical flip. 

The Second function calls the dataset defined above and splits the data into train (80%), validation (10%), and test sets (10%). Next, the split datasets are loaded using PyTorch’s Dataloader, with batch size made easily adjustable. 

Model Fundamentals

3 models were used in this project to deploy the neural network. 

1. Convolution Neural Network 
	- 3 blocks of 2d convolution, ReLU function, and Max pool 2d layer
	- 1 flatten layer 
	- 2 Linear dense layers
2. Efficient net b0 without weights 
	- Efficient net is based on a CNN 
	- This model adds more layers and complexity to the model that aim to increase 				prediction of labels
3. Transfer Learning: Efficient net b0 with pre-trained weights

Every model is compiled with:
	-  Criterion =  cross entropy loss
	-  Optimizer = stochastic gradient descent 
	-  metrics = accuracy 
  
  
Optimal model: Efficientnet with pre-trained weights 

After adjusting batch size and other parameters of learning rate and weight decay. The biggest factor in increasing the final learning rate with the pre-trained model was to increase the batch size to 75.

Thee model maintained a train and validation accuracy above 90% and reached a final prediction accuracy of 89.4%. This is to say that the model is able to predict the category 89% of the images to the proper label group. 
