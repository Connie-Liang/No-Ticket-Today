def show_random_images(number_of_random_images):
  '''
  Let's make sure that our dataset images and annotations are reading in correctly by showing three random images from our training set and their corresponding bounding boxes.
  Below, the code says: we'd like to create a dictionary variable to store information of our images.
  Then, pull 'x' number of random images from that dictionary and for each of them, read it in using open cv, and plot and show the bounding boxes.
  You can run this as many times as you want.

  Input:
      number of random images to show (int)
  Output:
      images with bounding boxes
  '''

  my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
  dataset_dicts = DatasetCatalog.get("my_dataset_train")

  for image in random.sample(dataset_dicts, number_of_random_images):
      img = cv2.imread(image["file_name"])
      visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5) #the negative indexing reformats the 'BGR' pixels to correct format of 'RGB'
      out = visualizer.draw_dataset_dict(image)
      cv2_imshow(out.get_image()[:, :, ::-1])

# set meta data for validation set (we'll use this later for testing)
test_metadata = MetadataCatalog.get("my_dataset_test")