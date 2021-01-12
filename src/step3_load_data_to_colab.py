# LOADING IN DATA TO GOOGLE COLAB
from google.colab import drive
from detectron2.data.datasets import register_coco_instances

def load_data_in_colab(train_file_path, validation_file_path, test_file_path, project_folder_path):
  '''
  If you are using google colab, you should mount your data to your google drive for continuous access without having to reupload data after a timeout message.
  Then after uploading the img file and 3 json files to Google Drive, pass in the pathways to each file. Make sure the file paths are encased in quotes.

  Inputs:
      directory: path to directory with images/json files (as a string)
  Output:
      'success' message
  '''

  drive.mount("/content/drive")

  register_coco_instances("my_dataset_train", {}, train_file_path, project_folder_path)
  register_coco_instances("my_dataset_val", {}, validation_file_path, project_folder_path)
  register_coco_instances("my_dataset_test", {}, test_file_path, project_folder_path)
  print(f"Success! Your data is mounted to Google Drive.")