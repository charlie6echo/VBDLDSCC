import tensorflow as tf
from numpy import mean
import os
import sys
import random
os.chdir('./Mask_RCNN/Layout')
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs/layout20210702T0231")

from Mask_RCNN.Layout import layout
config = layout.LayoutConfig()
LAYOUT_DIR = os.path.join(ROOT_DIR, "/home/Dataset")

class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

dataset = layout.LayoutDataset()
dataset.load_layout(LAYOUT_DIR, "val") #train or val
dataset.prepare()
datasetids = random.sample(list(dataset.image_ids),20)

def evaluate_model( model, config):
    APs = list(); 
    ARs = list();
    F1_scores = list(); 
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        
        #scaled_image = modellib.mold_image(image, config)
        #sample = expand_dims(scaled_image, 0)
        yhat = model.detect([image], verbose=0)
        r = yhat[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
        ARs.append(AR)
        F1_scores.append((2* (mean(precisions) * mean(recalls)))/(mean(precisions) + mean(recalls)))
        APs.append(AP)

    mAP = mean(APs)
    mAR = mean(ARs)
    return mAP, mAR, F1_scores


with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)


for rt,drs,mdls in os.walk(MODEL_DIR):
    for mdlf in sorted(mdls):
        mdlpath = os.path.join(rt,mdlf)
        if mdlpath.split('.')[-1]!="h5":
            continue
        #print("Loading weights ",mdl)
        model.load_weights(mdlpath , by_name=True)
        mAP, mAR, F1_score = evaluate_model( model, config)
        F1_score_2 = (2 * mAP * mAR)/(mAP + mAR)
        print("model: ",mdlf,"mAP (mean average precision): %.4f" % mAP,"mAR: (mean average recall): %.4f" % mAR, 'F1-score : ', F1_score_2)
