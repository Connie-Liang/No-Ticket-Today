<img src="https://secureservercdn.net/45.40.150.81/xpm.570.myftpupload.com/wp-content/uploads/2019/11/Parking-enforcement-2.png?time=1609235610" width="700" height="450">

# No Ticket Today!

An object detection project to spot pesky Parking Enforcement vehicles before they spot you!

    Tools: [Sixgill Sense, Dectectron2, Torch, Torchvision, Pyyaml, Pycocotools, Pillow, Numpy, os, json, cv2, random, Tensorboard]
    
"I love parking enforcement!!", said *no one ever*.

Simply seeing those mini, slow-cruising vehicles stalk up and down the streets of San Francisco hunting for prey is enough to induce a small amount of anxiety. 

It's a necessary job for sure, but maybe I'd like to know exactly when one of those silent, seemingly ubiquitious vehicles has arrived right in front of *my* car. 

# Let's see how we can do that!

**From here I will refer to Parking Enforcement vehicles as 'PE vehicles' for brevity**

1. Image Acquisition: Obtain images that include both PE vehicles and non-PE subjects.
2. Image Processing: For each image in the dataset, mark a 'bounding box' around the PE vehicle.
3. Object Detection: Using Detectron2, train the model on the training set of 1352 images.
4. Results: Test and evaluate the model results using Tensorboard and a holdout validation set.


# 1. Image Acquisition

I thought about using a script to query Google image results for 'PE vehicles', however, several problems arose: 

   (a) What do I mean by 'PE vehicle'? In different countries or cities, PE vehicles can vary in size and shape. Some more advanced PE vehicles, for example, look more like a smart car. In contrast, the specific shape I am looking for is the pervasive rectangular/parallelogram type PE vehicle (see differentiation below).
   
   (b) I wanted my model to be able to identify PE vehicles from near or far, and with or without other distracting objects in the picture, which meant the PE vehicle could located anywhere on the image. If I used a script to pull images and then cropped them to the same size (because the images often have to be the same size to be trained), they may or may not correctly include the PE vechicle.
   
So, I decided instead to pull frames from videos and use them as 'images'. This offered more advantages: abundance of images since I can control how many frames to grab per second, a variety of angles of the PE vehicle, a variety of backgrounds with differing objects in the images, and the images would be in the same size.

From YouTube, I found 34 videos featuring PE vehicles. I had trouble downloading the videos directly, so I used the 'Kazam' application to screen record them. Then I used the application 'VLC Media Player' to run each video through and set the amount of frames to grab.

My resulting dataset was 1700 images. Below is an example of one of the images:

<img src="https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step1_image.png" width="600" height="350"><img src="https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step1_image.png" width="600" height="350">


# 2. Image Processing

Time for bounding boxes! There are various free sites that you can build bounding boxes with. We need to annotate the PE vehicles in each of our images so our model down the line slowly "learns" what to look for.

The annotation tool I used was called Sixgill Sense. You can imagine the more images you have, the more annotating you'll have to do...my hand was cramping by the time I finished!

I saved my finished annotations and images in a COCO format, which is one of popular formats to save and process image data. This format ensures my annotations are compatible with the model type I will later use to train for object detection, which is the COCO faster rcnn model from modelZOO. 

My resulting files were: an 'img' folder with ALL the images in my dataset, and 3 separate json files for the dataset split into training, testing, and validation. Each json file contains a giant dictionary containing information about one of the images, such as its height, width, and bounding box coordinates. 

Below are two images showing the bounding boxes drawn around the PE vehicles, as well as an example of what the resulting annotated files look like in COCO format:

[![Annotated_Image1](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step2_image1.png)](#)
[![Annotated_Image2](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step2_image2.png)](#)

# 3. Object Detection

To build my PE vehicle object detection model, I used Detectron2. Detectron2 is a Facebook vision library that works in hand with Pytorch for object detection, image segmentation, keypoint detection, and panoptic segmentation (segmentation and classification combined).

I also utilized ModelZoo's "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" framework in training my data. "COCO-Detection" specifies the format the image processing was saved, for the purpose of object detection, and "faster_rcnn" is a type of region based convolutional neural network that is used for deep learning. The framework works by selective search and generating a limited number of ranked proposal regions on an image where it thinks there is an object to detect.
    
Here is an image showing the framework of how an rcnn works:


# 4. Results

I got an Average Precision (AP) score of 82%. My confidence threshold was set at 90%, meaning the model will only categorize the object if the model determines with 90% or higher confidence that the object is a PE vehicle. 

[![Success1](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image1.png)](#)
[![Success2](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image2.png)](#)
[![Success3](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image3.png)](#)
[![Success4](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image4.png)](#)
[![Success5](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image5.png)](#)
[![Success6](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_image6.png)](#)

## Interesting Observations:

I noticed that while my model correctly predicted the front, side, multiple PE vehicles in an image, and even a partial part of a PE very well, it seemed to be having more trouble with the back part of the vehicle. More specifically, it had trouble differentiating between the back of a PE vehicle and the back of a regular car, so while it would typically still labeled the PE vehicle, it would also incorrectly label the car as a PE vehicle. 

[![Mislabeled](https://github.com/Connie-Liang/No-Ticket-Today/tree/main/image_examples/step4_mistake.png)](#)

From these mistakenly labeled images, I noticed further that these mistakes occured more frequently when:
- a. the angle of the image made the back of both vehicles seem equivalent in size, even though in reality the size of a PE vehicle is significantly smaller than a regular car
- b. there was less space between the PE vehicle and regular car

This is partly understandable as the back of a PE vehicle contains many similar features and structures as that of a regular car (tailights, wheels, boxy structure, back windshield) and is also less unique than the fronts or sides in shape.

## How Would We Improve Our Model?

To improve our model, we should increase our dataset to include more images of the backside of PE vehicles and regular cars in the same image. We could also play with our parameter thresholds, such as increasing it to a higher confidence value to eliminate less confident detections. As a further idea, I could train "cars" as its own separate class and modify the model to differentiate between more classes.


# Ideas for What's Next?

1. I would like to apply this model to a video, so bounding boxes would follow the detected objects as they moved. This would allow more "live use" practicality and emulate the real world situation of a PE vehicle pulling up by my car.

2. I would like my model to detect more classes, such as officers or cars. Some parking enforcement is done by foot instead of by these types of PE vehicles, so if I could train my model to recognize uniformed officers, this could help close that gap. Creating a separate class for "cars" might also help the model learn to better differentiate between the backside of a regular car vs a PE vehicle. 

    The challenge I foresee for both these improvements would be requiring an excess of data. What constitutes a non-PE vehicle/regular car is so much more broad than the specific shape and type of PE vehicle the model is trained to find. As for the officers, it'll be hard to differentiate between officer vs regular pedestrian, or parking officer vs police officer vs business security officer, especially with different types and colors of uniforms.


# Acknowledgements

Thank you to:
- Michelle Hoogenhoat for her troubleshooting support
- Sage Elliot for Sixgill Sense and Detectron2 outline
