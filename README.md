# Model-asisted Labeling with YOLOv5
> custom image set annotation with a model's help


![base64 splash](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/modelassistedlabel%20splash.jpg?raw=true)

## Background

My exercise equipment, despite even being electronic, doesn’t connect to a network.

But if I instead point my webcam at the equipment’s LCD output, I can make a machine learn to identify and interpret useful information. Perfect! I’ll just utilize object detection to determine the location and identity of the machine’s analog readout. 

First question, just a tiny one, how do you do that?  

After wading through several guides, I found [Roboflow's YOLOv5 tutorial]( https://models.roboflow.com/object-detection/yolov5). They helped provide a hands-on and accessible experience in machine learning.

Unfortunately, I didn't have much luck with existing models being able to readily parse digits. Instead, I decided to start building my own dataset.

I shouldn't have been caught off-guard by the tedium of manually annotating images. As my mind starts to drift, I wonder if I’m a reCAPTCHA interface that’s gained sentience, and I break through. If I label enough digits, I can train a YOLO model to tell me what it sees. I can then take that information and pre-label my images with those predictions. 

The pieces come together.  I can focus on writing code while I use Roboflow to sort, generate, and deliver my images. I sleuth through [Ultralytic's](https://github.com/ultralytics/yolov5) original project and build wrappers around the essential functions in `detect.py` and `train.py`.

This repository contains the tools that let me "pre-label" my images before sending them off for human inspection and correction.

I use the `Viewer` class to 
{% include note.html content='In `./Image Repo` I provide access to 841 labeled images (lumped in one folder) and 600 unlabeled images (seperated into three sets of 200 images - lighting condition is the same within each run, but differs between runs). ' %}




## Getting Started

{% include tip.html content='[Open In Colab](https://colab.research.google.com/github/PhilBrockman/ModelAssistedLabel/blob/master/index.ipynb)' %}

```
# clone this repository
!git clone -b future_forward https://github.com/PhilBrockman/ModelAssistedLabel.git
%cd "ModelAssistedLabel"
```

    Cloning into 'ModelAssistedLabel'...
    remote: Enumerating objects: 424, done.[K
    remote: Counting objects: 100% (424/424), done.[K
    remote: Compressing objects: 100% (120/120), done.[K
    remote: Total 4401 (delta 305), reused 414 (delta 299), pack-reused 3977[K
    Receiving objects: 100% (4401/4401), 207.35 MiB | 14.02 MiB/s, done.
    Resolving deltas: 100% (1328/1328), done.
    Checking out files: 100% (2375/2375), done.
    /content/drive/My Drive/Coding/vision.philbrockman.com/ModelAssistedLabel


### Expected Inputs:

-  **labeled images**
    + All of the images and labels must be in a common folder (subfolders allowed).
    + Labels must be in [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885).
{% include note.html content='Image/label pairs are based on their base filename. For example `image.jpg/image.txt` would be paired as would `other_image5.jpg/other_image5.txt`.' %}





```
# these images have already been labeled
labeled_images   = "Image Repo/labeled/Final Roboflow Export (841)"
```

  - **unlabeled images**

```
# these images need to be labeled
unlabeled_images = "Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00"
```

### Expected Output:

* **Folder** that contains: 
    - `images/`
      + a copy of every image in **Unlabeled Data**
    - `labels/`
      + result of running object detection on each image
    - a results folder produced by Ultralytic's `train.py` on the **Labeled Data** (if not using pre-trained weights)
    - `class labels.txt` to preserve the identity of the classes


```
from ModelAssistedLabel.config import Defaults
import os

project_name = "seven segment digits - "
export_folder = Defaults._itername(project_name)

print(export_folder)
```

    seven segment digits - 1


```
# make the export folder
os.mkdir(export_folder)

# make the images and labels subfolders
for resource_folder in ["images", "labels"]:
  os.mkdir(os.path.join(export_folder, resource_folder))
```

## Configure defaults

Several values are stored by the `Defaults` class. Any value can be overridden (and new values can be added. Make sure to `save()` any changes!

```
d = Defaults()
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

    /content/drive/My Drive/Coding/vision.philbrockman.com/ModelAssistedLabel


```
d.root = "/content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/"
```

Save any changes and enter root directory

```
d.save()
d.to_root()
```

    moving to /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/


I borrow the instructions to set up the Ultralytics repo from [the Roboflow tutorial]( https://models.roboflow.com/object-detection/yolov5). (If I'd be allowed one undo on this project, I wish I would have intially forked this project from that tutorial.)

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
    remote: Total 5532 (delta 1), reused 0 (delta 0), pack-reused 5525[K
    Receiving objects: 100% (5532/5532), 8.15 MiB | 8.33 MiB/s, done.
    Resolving deltas: 100% (3776/3776), done.
    /content/drive/My Drive/vision.philbrockman.com/ModelAssistedLabel/yolov5
    /content/drive/My Drive/vision.philbrockman.com/ModelAssistedLabel


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


    [autoreload of utils failed: Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/IPython/extensions/autoreload.py", line 247, in check
        superreload(m, reload, self.old_objects)
    ModuleNotFoundError: spec not found for the module 'utils'
    ]


The names of my classes are digits. Under the hood, the YOLOv5 model is working of the index of the class, rather than the human-readable name. Consequently, the identities of each class index must be supplied.

```
class_idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
```

## Processing input

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



```
zipped = g.write_split_to_disk(descriptor=export_folder)
```

    
    dirs ['./train', './valid', './test']
    yaml archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37/data.yaml
    subdir train
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37
    subdir valid
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37
    subdir test
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37
    os.listdir ['train', 'valid', 'test', 'data.yaml']


Next, the images need to be written in a way so that the Ultralytics repository can understand their content. The `Autoweights` class both organizes data and create weights. Running an "initialize" command makes changes to the disk.

```
from ModelAssistedLabel.train import AutoWeights
#configure a basic AutoWeights class instance
aw = AutoWeights(name=export_folder, out_dir=backup_dir)

# create train/valid/test split from a bag of labeled images (recusively seek out images/labels)
aw.initialize_images_from_zip(zipped)
```

    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37/train' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37/valid' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37/test' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 03-56-37/data.yaml' .


Peep on the sizes of the train/valid/test groups.

```
aw.traverse_resources()
```

    train/images
    	 > 769 files
    train/labels
    	 > 769 files
    valid/images
    	 > 305 files
    valid/labels
    	 > 305 files
    test/images
    	 > 159 files
    test/labels
    	 > 159 files
    File:  data.yaml


## Generate Weights

With the images written to disk, we can run the Ultralytics training algorithm. I loved watching the progress fly by in real time on the original `train.py`. Fortunately, the Ultralytics folk write the results file to disk so the model's training data is still accessible!

```
from ModelAssistedLabel.train import Trainer
```

```
%%time

aw.generate_weights(epochs=2000, data_yaml=d.trainer_template)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-32-044047f7d798> in <module>()
    ----> 1 get_ipython().run_cell_magic('time', '', '\naw.generate_weights(epochs=2000, data_yaml=d.trainer_template)')
    

    /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py in run_cell_magic(self, magic_name, line, cell)
       2115             magic_arg_s = self.var_expand(line, stack_depth)
       2116             with self.builtin_trap:
    -> 2117                 result = fn(magic_arg_s, cell)
       2118             return result
       2119 


    <decorator-gen-53> in time(self, line, cell, local_ns)


    /usr/local/lib/python3.7/dist-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        186     # but it's overkill for just that one bit of state.
        187     def magic_deco(arg):
    --> 188         call = lambda f, *a, **k: f(*a, **k)
        189 
        190         if callable(arg):


    /usr/local/lib/python3.7/dist-packages/IPython/core/magics/execution.py in time(self, line, cell, local_ns)
       1187         if mode=='eval':
       1188             st = clock2()
    -> 1189             out = eval(code, glob, local_ns)
       1190             end = clock2()
       1191         else:


    <timed eval> in <module>()


    TypeError: generate_weights() got an unexpected keyword argument 'data_yaml'


The results folder is stored as an attribute as well, and it has a lot of data stored therein.

```
import os
aw.last_results_path, len(os.listdir(aw.last_results_path))
```

However, the weights are stored in a subfolder called (aptly) "weights".

```
os.listdir(aw.last_results_path + "/weights")
```

View the last couple lines 

```
with open(aw.last_results_path + "results.txt") as results_file:
  results = results_file.readlines()
print("Epoch   gpu_mem       box       obj       cls     total    labels  img_size")
results[-5:]
```

## Labeling a new set of images

And the `Viewer` class doesn't care how recently your weights were generated so you can plug in existing weights.

```
from ModelAssistedLabel.detect import Viewer

# access the folder of results from the AutoWeights instance
results_folder = aw.last_results_path

# I'm choosing to use the best weight.
weight_path = results_folder + "/weights/best.pt"

# Viewer needs a set of weights and an array of labels for the detected object types
v = Viewer(weight_path, class_idx)
```

Selects all images in the unlabeled folder and let's us look through the computer's eyes at the images.

```
%matplotlib inline 
import random, glob

images = glob.glob(f"./{unlabeled_images}/*.jpg")

for image in random.sample(images,5):
  v.plot_for(image)
```

```
results = []
for image in images:
  results.append(v.predict_for(image))
```

## Exporting annotated images

Store the class labels with index 0 on line 1, index 1 on line 2, and so on.

```
with open(os.path.join(export_folder, "label_map.txt"), "w") as label_map:
  label_map.writelines("\n".join(class_idx))
```

Ensure that image/label pairs have a common root filename

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

#check if weights were generated
if aw is not None and os.path.exists(aw.last_results_path):
  print(f"Moving yolov5 results folder: {aw.last_results_path}")
  shutil.move(aw.last_results_path, export_folder)
else:
  print("No weights to save")
```

At this point I would have uploaded this set of image/label pairs to Roboflow for correction and annotation. As the model grows more accurate, I would alter camera position or lighting until the model started stumbling again. I want to be keeping the model on its toes!

To be transparent, I developed a [custom React annotator](https://github.com/PhilBrockman/autobbox) that better suited my needs.

I labeled dozens upon dozens and dozens of images with Roboflow and would recommend their free annotation service! 

## Wrap up

My original goal of "smartifying" my rowing machine is closer than ever. 

It is possible to parse workout information (thought currently, I only have access to a maximum of 4 digits). I wonder if the model could keep up if there were 20+ digits to capture.

I know that lighting and camera position have an effect on accuracy. Here's how I'm holding my computer steady as I modify the lighting: [standing](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-laptop-mount.jpg), [floor 1](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-computer-capture.jpg), [floor 2](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/DIY-capture.jpeg?raw=true).

Here are 3 runs captured under different lighting conditions:
* `21-3-22 rowing (200) 7:50-12:50` (direct lighting from one light source)
* `21-3-22 rowing (200) 1:53-7:00` (direct lighting from one light source with glare)
* `21-3-18 rowing 8-12 ` (direct light and ambient lamps turned on)
{% include note.html content='All unlabeled images were taken inside a blacked-out room. The are stored in `Image Repo/unlabeled/`' %}





### Lingering Questions

My labeled images are disorderly. There's data from other rowing machines and from [a kind *stranger*'s github repo](https://github.com/SachaIZADI/Seven-Segment-OCR). Some images have been cropped to only include the display. Did having varied data slow me down overall? Or did it make the models more robust? 

