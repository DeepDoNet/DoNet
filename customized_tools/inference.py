from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
import os
import cv2
import matplotlib.pyplot as plt
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
python inference.py 
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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.OUTPUT_DIR = outputDir
cfg.DATASETS.TEST = ("amodal_coco_test",)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightsFile)
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("amodal_coco_val")
 
for d in dataset_dicts:  
    im = cv2.imread(d["file_name"])
    visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("amodal_coco_val"), scale=0.5)
    vis = visualizer.draw_dataset_dict(d,"segmentation")
    plt.imsave(os.path.join(outputDir, "results", d["file_name"].split("/")[-1]+'_gt'), vis.get_image()[:, :, ::-1])


    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                metadata=MetadataCatalog.get("amodal_coco_val"), 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"),'pred_masks')
    plt.imsave(os.path.join(outputDir, "results", d["file_name"].split("/")[-1]), v.get_image()[:, :, ::-1])