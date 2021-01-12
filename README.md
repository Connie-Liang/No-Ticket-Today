<img src="https://secureservercdn.net/45.40.150.81/xpm.570.myftpupload.com/wp-content/uploads/2019/11/Parking-enforcement-2.png?time=1609235610" width="700" height="400">

# No Ticket Today!

An object detection project to spot pesky Parking Enforcement vehicles before they spot you!

    Tools: [Sixgill Sense, Dectectron2, Torch, Torchvision, Pyyaml, Pycocotools, Pillow, Numpy, os, json, cv2, random, Tensorboard]
    
"I love parking enforcement!!", said no one ever.

Simply seeing those mini, paragonic, slow-cruising vehicles stalk up and down the streets of San Francisco for prey is enough to induce a small amount of anxiety. 

It's a necessary job for sure, but maybe I'd like to know exactly when one of those silent, seemingly ubiquitious vehicles has arrived right in front of *my* car. 

# Let's see how we can do that!

**From here I will refer to Parking Enforcement vehicles as 'PE vehicles' for brevity

1. Image Acquisition: Obtain images that include both PE vehicles and non-PE subjects.
2. Image Processing: For each of your images, mark a 'bounding box' around the PE vehicle.
3. Object Detection:


# 1. Image Acquisition

I thought about using a script to query Google image results for 'PE vehicles', however, several problems arose: 

   (a) What do I mean by 'PE vehicle'? In different countries or cities, PE vehicles can vary in size and shape. Some more advanced PE vehicles, for example, look more like a smart car. In contrast, the specific shape I am looking for is the pervasive rectangular/parallelogram type PE vehicle (see differentiation below).
   
   (b) I wanted my model to be able to identify PE vehicles from near or far, and with or without other distracting objects in the picture, which meant the PE vehicle could located anywhere on the image. If I used a script to pull images and then cropped them to the same size (because the images often have to be the same size to be trained), they may or may not correctly include the PE vechicle.
   
So, I decided instead to pull frames from videos and use them as 'images'. This offered more advantages: abundance of images since I can control how many frames to grab per second, a variety of angles of the PE vehicle, a variety of backgrounds with differing objects in the images, and the images would be in the same size.

From YouTube, I found 34 videos featuring PE vehicles. I had trouble downloading the videos directly, so I used the 'Kazam' application to screen record them. Then I used the application 'VLC Media Player' to run each video through and set the amount of frames to grab.

My resulting dataset was 1700 images.

# 2. Image Processing

