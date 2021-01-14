# unfortunately detectron2 doesn't have a built in way to do validation as you're training the model so
# the below code will feed in the evaluation dataset into the model as it's being trained

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class MyTrainer(DefaultTrainer):
  @classmethod
  '''
  because detectron2 doesn't have a built in way to do validation as you're training the model, this is a custom way to do it.
  so if an output folder is not yet created (iterations not done running), then this code will keep feeding validation.
  '''
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, cfg, True, output_folder)



def train_data(model_type, learning_rate, iterations, number_classes, eval_message):

  cfg = get_cfg() # start detectron2's default config
  cfg.merge_from_file(model_zoo.get_config_file(model_type)) # we are going to load in our model type
  cfg.DATASETS.TRAIN = ("my_dataset_train",) # <- don't forget comma at end, otherwise it will throw an error. it's expecting a tuple datatype to run

  cfg.DATALOADER.NUM_WORKERS = 0 # higher number means more subprocesses to use for data loading. default is 0
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_type) # transfer learning: let training initialize from the above architecture that and use those weights included
  cfg.SOLVER.IMS_PER_BATCH = 2 # image per batch per GPU
  cfg.SOLVER.BASE_LR = learning_rate
  cfg.SOLVER.MAX_ITER = iterations
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # this is saying how many instances per image we are going to look at. default is 512 but we don't need that many
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #this is saving our model during training in a newly made output folder

  cfg.DATASETS.TEST = ("my_dataset_val",) # testing on our validation
  cfg.TEST.EVAL_PERIOD = eval_message # this says every 'x' cycles, run our model on validation dataset and give us an output (should show improvement)
  trainer = MyTrainer(cfg)

  trainer.resume_or_load(resume=False)
  trainer.train()