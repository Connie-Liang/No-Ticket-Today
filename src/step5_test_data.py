# we are now going to load this model that we trained so we can use it on test set

def test_data(model_type, number_classes, threshold):

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(model_type)) # has to be same as your starting model architecture
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # instead of using the predefined weights, we want to tell it to use our saved model weights (transfer learning)
  cfg.DATASETS.TEST = ("my_dataset_test", ) # loading in our test dataset
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes #if you have more than one class remember to update this
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # will only draw a bounding box if the guess is above "x" percent confidence

  predictor = DefaultPredictor(cfg) # this is our variable to pass in an image and get some result


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))