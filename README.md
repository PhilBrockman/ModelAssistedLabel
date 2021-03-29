# Model-asisted Labeling with YOLOv5
> custom image set annotation with a model's help


![base64 splash](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/modelassistedlabel%20splash.jpg?raw=true)

## Background

My exercise equipment doesn’t connect to a network. But I still want the "smart workout experience" when I'm using a "dumb" rowing machine.

Maybe if I point my webcam at the equipment’s LCD output, I can make my computer identify and interpret useful information. Perfect! I’ll just utilize object detection to determine the location and identity of digits on the machine’s readout. 

First question -- just a tiny one -- how do you do that?  

After wading through several guides, I found [Roboflow's YOLOv5 tutorial]( https://models.roboflow.com/object-detection/yolov5). They provide a hands-on and accessible experience in machine learning. But unfortunately, I couldn't progress immediately on my specific project. Instead, I had to build my own dataset.

As I labeled digits on image after image, tedium tore at me. I envisioned using the YOLOv5 model predictions as "pre-labels". I sleuth through [Ultralytic's](https://github.com/ultralytics/yolov5) original project, and I build wrappers around `detect.py` and `train.py`. I determine that my vision could be a reality.

This repository contains the tools that let me "pre-label" my images before sending them off for human inspection and correction. I also provide labeled and unlabeled images to demonstrate the tools.

* `Image Repo/`
 - `labeled/`
    + `Final Roboflow Export (841)/` 
      + 841 labeled image dataset
 - `unlabeled/`
    + `21-3-22 rowing (200) 7:50-12:50/`
      - 200 images with direct lighting from one light source
    + `21-3-22 rowing (200) 1:53-7:00/` 
      - 200 images with direct lighting from one light source and intermittent glare
    + `21-3-18 rowing 8-12/` 
      - 200 images with direct light and ambient lamps turned on




## Getting Started

{% include tip.html content='[Open In Colab](https://colab.research.google.com/github/PhilBrockman/ModelAssistedLabel/blob/master/index.ipynb) (and enable GPU)' %}

```
# clone this repository
!git clone https://github.com/PhilBrockman/ModelAssistedLabel.git
%cd "ModelAssistedLabel"
```

    Cloning into 'ModelAssistedLabel'...
    remote: Enumerating objects: 463, done.[K
    remote: Counting objects: 100% (463/463), done.[K
    remote: Compressing objects: 100% (146/146), done.[K
    remote: Total 4440 (delta 332), reused 439 (delta 312), pack-reused 3977[K
    Receiving objects: 100% (4440/4440), 210.50 MiB | 13.74 MiB/s, done.
    Resolving deltas: 100% (1355/1355), done.
    Checking out files: 100% (2381/2381), done.
    /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel


### Expected Inputs

-  **labeled images**
    + All of the images and labels must be in a common folder (subfolders allowed).
    + Labels must be in [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885).
{% include note.html content='Image/label pairs are based on their base filename. For example `image.jpg/image.txt` would be paired as would `other_image5.jpg/other_image5.txt`.' %}





```
# these images have already been labeled
labeled_images = "Image Repo/labeled/Final Roboflow Export (841)"
```

  - **unlabeled images**

```
# these images need to be annotated
unlabeled_images = "Image Repo/unlabeled/21-3-22 rowing (200) 7:50-12:50"
```

### Expected Output

* Folder that contains: 
    - `images/`
      + a copy of every image in **Unlabeled Data**
    - `labels/`
      + result of running object detection on each image
    - `class labels.txt` to preserve the identity of the classes
    - (if not using pre-trained weights) a results folder produced by Ultralytic's `train.py` on the **Labeled Data** 


Start by building the folder structure for the output.

```
from ModelAssistedLabel.config import Defaults
import os

project_name = "seven segment digits - "
export_folder = Defaults._itername(project_name)

print(export_folder)
# make the export folder
os.mkdir(export_folder)

# make the images and labels subfolders
for resource_folder in ["images", "labels"]:
  os.mkdir(os.path.join(export_folder, resource_folder))
```

    seven segment digits - 1


### Configure Defaults

Several values are stored by the `Defaults` class. Any value can be overridden (and new values can be added. Make sure to `save()` any changes! (changes are written to the `config_file`

```
d = Defaults()
```

#### changing a Default value

```
print(" -- Defined Keys: --")
print("\n".join([x for x in d.__dict__.keys()]))
```

     -- Defined Keys: --
    config_file
    root
    split_ratio
    data_yaml
    resource_map
    trainer_template


Speciy the absolute path of the root directory.

```
!pwd
```

    /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel


```
d.root = "/content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/"
```

Save changes and enter root directory

```
d.save()
d.to_root()
```

    moving to /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/


#### cloning YOLOv5

```
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo

%cd yolov5
# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

%cd ..
```

    Cloning into 'yolov5'...
    remote: Enumerating objects: 7, done.[K
    remote: Counting objects: 100% (7/7), done.[K
    remote: Compressing objects: 100% (7/7), done.[K
    remote: Total 5532 (delta 1), reused 0 (delta 0), pack-reused 5525
    Receiving objects: 100% (5532/5532), 8.15 MiB | 7.60 MiB/s, done.
    Resolving deltas: 100% (3776/3776), done.
    /content/drive/My Drive/vision.philbrockman.com/ModelAssistedLabel/yolov5
    /content/drive/My Drive/vision.philbrockman.com/ModelAssistedLabel


#### define class map


I have `nc = 10` classes and their `names` are all string types.


```
d.data_yaml = """train: ../train/images
val: ../valid/images

nc: 10
names: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']"""

d.save()
```

Make sure that the class names are accessible:

```
d.get_class_names()
```




    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']



Store the class labels with index 0 on line 1, index 1 on line 2, and so on.

```
with open(os.path.join(export_folder, "label_map.txt"), "w") as label_map:
  label_map.writelines("\n".join(d.get_class_names()))
```

## Generating Weights


Make sure GPU is enabled.

```
if torch.cuda.is_available():
  print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) ))
  d.to_root() #step up a level
else:
   raise Exception("enable GPU")
```

    Setup complete. Using torch 1.8.0+cu101 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)
    moving to /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/


### Preparing Filesystem

The `Generation` class helps convert an unordered folder of images and labels into a format compatible with YOLOv5. By default, the train/valid/test split is set to 70%/20%/10%.

```
from ModelAssistedLabel.fileManagement import Generation

backup_dir = "archive/Generation/zips"

g = Generation(repo=labeled_images, 
               out_dir=backup_dir,
               verbose=True)

g.set_split()
g.get_split()
```




    [{'train': 589}, {'valid': 169}, {'test': 83}]



Backups are built into this system. As an intermediate step, the split must be written to disk in `g.out_dir`.


```
zipped = g.write_split_to_disk(descriptor=export_folder)
```

    
    dirs ['./train', './valid', './test']
    yaml archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03/data.yaml
    subdir train
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03
    subdir valid
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03
    subdir test
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03
    os.listdir ['train', 'valid', 'test', 'data.yaml']


Next, the images need to be written in a way so that the Ultralytics repository can understand their content. The `Autoweights` class both organizes data and create weights. Running an "initialize" command makes changes to the disk.

```
from ModelAssistedLabel.train import AutoWeights
#configure a basic AutoWeights class instance
aw = AutoWeights(name=export_folder, out_dir=backup_dir)

# create train/valid/test split from a bag of labeled images (recusively seek out images/labels)
aw.initialize_images_from_zip(zipped)
```

    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03/train' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03/valid' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03/test' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-46-03/data.yaml' .


Peep on the sizes of the train/valid/test groups.

```
aw.traverse_resources()
```

    train/images
    	 > 589 files
    train/labels
    	 > 589 files
    valid/images
    	 > 169 files
    valid/labels
    	 > 169 files
    test/images
    	 > 83 files
    test/labels
    	 > 83 files
    File:  data.yaml


### Running `train.py`

With the images written to disk, we can run the Ultralytics training algorithm. I loved watching the progress fly by in real time on the original `train.py`. Fortunately, the Ultralytics folk write the results file to disk so the model's training data is still accessible!
{% include note.html content='this output has already been calculated and stored in `pre-trained/results` for convenience.' %}

```
%%time

aw.generate_weights(epochs=2000, yaml_data=Defaults().trainer_template)
```

    CPU times: user 1min 3s, sys: 10.8 s, total: 1min 14s
    Wall time: 4h 57min 26s





    'yolov5/runs/train/seven segment digits - 1/'



The results folder is stored as an attribute as well, and it has a lot of data stored therein.

```
import os
aw.last_results_path, len(os.listdir(aw.last_results_path))
```




    ('yolov5/runs/train/seven segment digits - 1/', 22)



However, the weights are stored in a subfolder called (aptly) "weights".

```
os.listdir(aw.last_results_path + "/weights")
```




    ['last.pt', 'best.pt']



## Machine-assisted Labeling

### Labeling a New Set of Images

The `Viewer` class needs weights and class labels to operate.

```
from ModelAssistedLabel.detect import Viewer

# access the folder of results from the AutoWeights instance
results_folder = aw.last_results_path

# I'm choosing to use the best weight.
weight_path = results_folder + "/weights/best.pt"

# Viewer needs a set of weights and an array of labels for the detected object types
v = Viewer(weight_path, Defaults().get_class_names())
```

    Fusing layers... 


Let's us look through the computer's eyes at the images.

With the existing dataset, the model performs best under direct lighting.

```
%matplotlib inline 
import random, glob

images = glob.glob(f"./{unlabeled_images}/*.jpg")

for image in random.sample(images,5):
  print(image)
  v.plot_for(image)
```


    Output hidden; open in https://colab.research.google.com to view.


In this small sample, the model performs quite well. Nonetheless, manual review of all images is needed to remove overlapping predictions, categorization errors, and absent boxes.

```
results = []
for image in images:
  results.append(v.predict_for(image))
```

### Exporting Annotated Images

Ensure that image/label pairs have a common root filename and collect relevant files in a single folder.

```
import random, PIL, shutil
salt = lambda: str(random.random())[2:]

for result in results:
  #generate a likely-to-be-unique filename
  shared_root = Defaults._itername(f"{project_name}-{salt()}")

  #save the image to the outfile
  image = PIL.Image.open(result["image path"])
  image.save(os.path.join(export_folder, "images", f"{shared_root}.jpg"))

  #save the predictions to the outfile
  predictions = result["predictions"]
  with open(os.path.join(export_folder, "labels", f"{shared_root}.txt"), "w") as prediction_file:
    prediction_file.writelines("\n".join([x["yolov5 format"] for x in predictions]))
```

Lastly, if results need to get saved, make sure they get saved.

```
moved = False #set a flag

try: 
  if os.path.exists(aw.last_results_path):
    # `aw` exists and it has been executed 
    print(f"Moving yolov5 results: {aw.last_results_path}")
    shutil.move(aw.last_results_path, export_folder)

    #flip the flag
    moved = True
except NameError:
  pass

if not(moved):
  # either the AutoWeigts didn't pan out or it wasn't used
  print("No weights to save")
```

    No weights to save


I labeled dozens upon dozens and dozens of images with Roboflow and would recommend their free annotation service! However, to be transparent, I developed [an annotator](https://github.com/PhilBrockman/autobbox) in React that better suited my physical needs.

## Wrap Up

I have uncovered a camera and lighting positioning that allows for my model to read the LCD at with high fidelty. I'm using object detection as a form of OCR and it's working!

I see three main areas for development with this project. The first would be bolstering the dataset (and staying in the machine learning space). The second would be logic interpreting parsed data (building the "smart" software).

The third area of development is refactoring. I made a decision early on to hard to hardcode the path to training and validation images. It's worth revisiting the way `Defaults.data_yaml` is constructed.



### Note on the Dataset

This dataset of 841 images is a mishmash. There's images from a different rowing machine and also from  [this](https://github.com/SachaIZADI/Seven-Segment-OCR) repo. Some scenes are illuminated with sunlight. Others have been cropped to include only the LCD. Digits like 7, 8, and 9 are underrepresented.


### Recording from Laptop

This is my setup for how I'm currently fixing the position my black-shelled laptop while recording: [pic1](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-laptop-mount.jpg), [pic2](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-computer-capture.jpg), [pic3](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/DIY-capture.jpeg?raw=true).

I use the `_capture.ipynb` notebook to capture images on a bit of a delay to prevent repeat images from cluttering the dataset. For me, it was much easier to get recording working from a local notebook than from a Colab notebook but YMMV.



### About Me

I am finishing my fourth year as a public school teacher in Kentucky. This summer, I am moving to the Bay Area to pursue a career in tech. When I’m not coding, I enjoy playing violin! Reach me at phil.brockman@gmail.com.


