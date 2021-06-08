"""
Mask R-CNN
Train on the toy layout dataset.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 layout.py train --dataset=/path/to/layout/dataset --weights=coco --layers='heads' --lr=0.001 --epochs=30

    # Resume training a model that you had trained earlier
    python3 layout.py train --dataset=/path/to/layout/dataset --weights=last --layers='heads' --lr=0.001 --epochs=30

    # Train a new model starting from ImageNet weights
    python3 layout.py train --dataset=/path/to/layout/dataset --weights=imagenet --layers='heads' --lr=0.001 --epochs=30
    
    
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class LayoutConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "layout"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + (txt+math+img+tbl)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class LayoutDataset(utils.Dataset):

    def load_layout(self, dataset_dir, subset):
        """Load a subset of the Layout dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 4 classes to add.
        #text','Equation','Table','Image'
        self.add_class("layout", 1, "Text")
        self.add_class("layout", 2, "Math")
        self.add_class("layout", 3, "Table")
        self.add_class("layout", 4, "Image")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        bbxs =[]
        list(map(bbxs.extend, [[bbx for  bbx in fls if  bbx.endswith(".bbx")  ]  for _,_,fls in os.walk(dataset_dir)  ]))
        
        
        for bb in bbxs:
            dir_name = bb.split('.')[0]
            annotations = json.load(open(os.path.join(dataset_dir, bb)))
            #val = annotations.values() # don't need the dict keys
            annotations_vals = list(annotations.values())
            imglst = list(annotations_vals[2].keys())
            
            for i in range(len(imglst)):
            
                image_name = (list(imglst)[i])
                #print(image_name)
                annt_pp = annotations['images'][image_name]['annotations']#Annotation Per page
                coords = [list(annt_pp[r]['bbox'].values() ) for r in range(len(annt_pp))]#co-ordinates for layout
                label = [(annt_pp[r]["label"]) for r in range(len(annt_pp))]#label for layout
                label_dict={'Text':1,'Math':2,'Table':3,'Image':4}
                label_ids = [label_dict[a] for a in label]
                
                #print(len(label)==len(coords))#cross check  no. of labels w.r.t coords
                
                image_path = os.path.join(dataset_dir,dir_name, image_name)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
    
                self.add_image(
                    "layout",
                    image_id=image_name,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    rectangle=coords,
                    class_ids = label_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a ND array of class IDs of the instance masks.
        """
        # If not a layout dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "layout":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "layout":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['class_ids']
        mask = np.zeros([info["height"],info["width"], len(info["rectangle"]) ],
                        dtype=np.uint8)

        print(info["path"])
        for k, p in enumerate(info["rectangle"]):
            
            # Get indexes of pixels inside the polygon and set them to 1
            x1= int(p[0]*info["width"])
            x2= int(p[1]*info["width"])
            y1= int(p[2]*info["height"])
            y2= int(p[3]*info["height"])
            #print(info["path"],x1,x2,y1,y2,p)
	    # We dont't want polygon mask so we are giving Co-ords of BBox Layout
            rr, cc = skimage.draw.rectangle((x1,y1), (x2,y2))
            #print(info["path"],i,p[0])
            #rr, cc = skimage.draw.polygon([p[0]*info["width"],p[1]*info["width"]], [p[2]*info["height"],p[3]*info["height"]])

            mask[cc , rr,k] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "layout":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LayoutDataset()
    dataset_train.load_layout(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LayoutDataset()
    dataset_val.load_layout(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect layouts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/layout/dataset/",
                        help='Directory of the Layout dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--layers', required=True,
                        metavar="layers to be trained ",
                        help="layers to be trained upon")
    parser.add_argument('--lr', required=True,
                        metavar="Learning Rate ",
                        help="Learning rate used for Training")
    parser.add_argument('--epochs', required=True,
                        metavar="epochs ",
                        help="epochs for Training")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Layers: ", args.layers)
    print("LR: ", args.lr)
    print("Epochs: ", args.epochs)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LayoutConfig()
    else:
        class InferenceConfig(LayoutConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'Create a new commnad'".format(args.command))
