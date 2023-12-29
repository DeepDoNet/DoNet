from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--annTestFile', type=str, required=True, help="path to the test json file")
parser.add_argument('--imgTestFile', type=str, required=True, help="path to the test image file")
parser.add_argument('--configFile', type=str, default="COCO-InstanceSegmentation/mask_triple_branch_rcnn_R_50_FPN_3x_heads_attention.yaml", help="path to the config file")
parser.add_argument('--outputDir', type=str, required=True, help="path to the output directory, note that this is the directory that stores the weights file, and the evaluation results will be stored in <outputDir>/results")
parser.add_argument('--weightsFile', type=str, default="model_final.pth", help="name of the weights file, note that the absolute path is <outputDir>/<weightsFile>")

args = parser.parse_args()
    
'''
to run:
python evaluate.py 
    --annTestFile <path to the test json file> 
    --imgTestFile <path to the test image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
    --weightsFile <name of the weights file>
'''
    
annTestFile = args.annTestFile
imgTestFile = args.imgTestFile
configFile = args.configFile
outputDir = args.outputDir
weightsFile = args.weightsFile
register_coco_instances("amodal_coco_test", {}, annTestFile, imgTestFile)

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(model_zoo.get_config_file(configFile))
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.OUTPUT_DIR = outputDir
cfg.DATASETS.TEST = ("amodal_coco_test",)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightsFile)
# set the testing threshold for this model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
#  evaluate its performance using AP metric implemented in COCO API.
from detectron2.evaluation import AmodalEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = AmodalEvaluator("amodal_coco_test", cfg, False,
                            output_dir=os.path.join(cfg.OUTPUT_DIR, "results"))
val_loader = build_detection_test_loader(cfg, "amodal_coco_test")
inference_on_dataset(trainer.model, val_loader, evaluator)