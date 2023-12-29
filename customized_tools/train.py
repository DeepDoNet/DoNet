import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import DuringTrainAmodalEvaluator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--annTrainFile', type=str, required=True, help="path to the train json file")
parser.add_argument('--imgTrainFile', type=str, required=True, help="path to the train image file")
parser.add_argument('--annValFile', type=str, required=True, help="path to the validation json file")
parser.add_argument('--imgValFile', type=str, required=True, help="path to the validation image file")
parser.add_argument('--configFile', type=str, default="COCO-InstanceSegmentation/mask_triple_branch_rcnn_R_50_FPN_3x_heads_attention.yaml", help="path to the config file")
parser.add_argument('--outputDir', type=str, required=True, help="path to the output directory, note that this is the directory that stores the weights file, and the evaluation results will be stored in <outputDir>/results")

args = parser.parse_args()
    
'''
to run:
python train.py 
    --annTrainFile <path to the train json file>
    --imgTrainFile <path to the train image file>
    --annValFile <path to the validation json file> 
    --imgValFile <path to the validation image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
'''

annTrainFile = args.annTrainFile
imgTrainFile = args.imgTrainFile    
annValFile = args.annValFile
imgValFile = args.imgValFile
configFile = args.configFile
outputDir = args.outputDir

register_coco_instances("amodal_coco_train", {},annTrainFile , imgTrainFile)
register_coco_instances("amodal_coco_val", {}, annValFile, imgValFile)

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(model_zoo.get_config_file(configFile))
cfg.DATASETS.TRAIN = ("amodal_coco_train",)
cfg.DATASETS.TEST = ("amodal_coco_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.SOLVER.BASE_LR = 0.001 # 0.0005  # pick a good LR
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.STEPS = (40000,50000)
cfg.SOLVER.MAX_ITER = 60000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.TEST.EVAL_PERIOD = 10000
cfg.VIS_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.OUTPUT_DIR = outputDir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg)

evaluator = DuringTrainAmodalEvaluator("amodal_coco_val", cfg, False, output_dir=cfg.OUTPUT_DIR + "/Output/" + cfg.DATASETS.TEST[0])

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return evaluator


trainer = Trainer(cfg)

trainer.resume_or_load(resume=False)
trainer.train()