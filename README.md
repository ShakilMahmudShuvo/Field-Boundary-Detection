# Field-Boundary-Detection-Challenge
In this challenge, the objective is to design machine learning algorithms for classifying crop field boundaries using multispectral observations. In other words, proposed  model is expected to accurately segment out boundary masks delineating areas where there is crop field versus no crop field.
# Environment
The notebook `Field Boundary Detection.ipynb` can be run locally using the `requirements.txt` file on Python 3.8.13 with a minimum RAM : 8GB.
<br>
`inference.py` can be run to Inference mask images from the private test set. The inference images contain the same name as it is already given and the folder structure is the same as provided.
<br>
`inference` directory contains the inference masks generated using the `inference.py` script.

# Approach
This is one of the trickiest computer vision problems I have encountered. I tried to learn everything I could about this task, and enjoyed a lot
## The Data
Here, we are dealing with a classic Satellite Image Time Series (SITS) problem. The labels were digitized for the months of March, April (on season), August (off season), October, November and December of 2021. 
The data are tiled into 256x256 chips adding up to 56 tiles. Within those 56 tiles 1226 individual field boundaries have been identified. The dataset has been split into training and private-test chips (51 in the train and 6 in the private-test).

Sattelite imagery for 4 bands [B01, B02, B03, B04] mapped to a common 3.7m spatial resolution grid for 6 timestamps.
I have used the training chips to train my model.

## Image Preprocessing
The intuition behind my choice of image pre-processing was aimed at primarily creating weakly delineated boundaries in the images to enable the models gain better visual perception of the fields and also to offer a better supervised learning procedure. The end-goal is geared towards making the label masks reasonably detectable within their corresponding images.
The idea can be laid down in steps below:

1. Grab a field

2. Apply a square root function on a field to take care of outliers

3. Compute the mean and standard deviation of result in step 2

4. Standardize result in step 3 per-channel. Thus, for each channel, subtract out the mean and divide result by the standard deviation
AND Thats all it takes!
## Model
The model is a Unet-based architecture using pretrained Efficientnet-B7 as encoder.

Surprisingly, it offered good consistency over the loss, F1 and Recall metrics across both the train and validation sets.

The data was split based on a custom segmentation-based stratified split method.

Training is done utilizing AdamW optimizer with a learning rate of 1e-4 and weight decay of 1e-5, for 200 epochs.
My compute did not permit larger batch sizes. Additional data augmentation ideas could not be applied as my notebook was repeatedly crashing with only a custom mixup augmentation due to GPU memory shortage.
## Results
This is training VS Validation loss curve after training 200 epochs:
![loss](https://github.com/phreak1703007/Field-Boundary-Detection/assets/62479964/ca7eb022-2063-4024-862a-ac2d9c88de9f)
<br>
This is training VS Validation F1-Score curve after training 200 epochs:
![f1](https://github.com/phreak1703007/Field-Boundary-Detection/assets/62479964/34905062-05e2-4238-9c23-42ecc9f36601)
<br>
Based on the graph, I can say that the model performed well and was generalized enough to handle new or unseen data.

## Inference 
`inference.py` can be run to Inference mask images from the private test set. The inference images contain the same name as it is already given and the folder structure is the same as provided.
Here is a sample inference mask image:
![inference](https://github.com/phreak1703007/Field-Boundary-Detection/assets/62479964/aa3388a7-6474-4c01-aca7-883e7dc092f7)
<br>

# Finally, A summary of my experiments ( failed + successful )
I approached the problem from various perspectives, experimenting with different methods, structures, and approaches until I ultimately achieved the desired outcomes.
## Model Ablations
1. Standard U-Net 
2. Different pretrained model backbones — Efficient net variants(B1-B7), Resnet34, Resnet50, Seresnet34, VGG16 and VGG19-bn.
3. Residual Unet
4. Channel Attention Based U net
## Loss Function Ablations
1. Binary Crossentropy
2. Focal loss
3. Dice loss
4. Combo loss — dice loss + focal
5. Combo loss — dice + binary crossentropy ( Tried different weighting configs before landing on 0.9dice + 0.1bce)
## Optimizers
1. Adam
2. AdamW
3. SGD

## Insights
Generally speaking, the extent to which I explored all of the above experimental configurations was highly decided by my compute configuration. It would be really nice if I could experiment more but on a much more powerful machine!

A great deal of learning experience it was, with numerous uncharted territories that one could explore.
