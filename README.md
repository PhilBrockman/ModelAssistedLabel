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
    remote: Enumerating objects: 463, done.[K
    remote: Counting objects: 100% (463/463), done.[K
    remote: Compressing objects: 100% (146/146), done.[K
    remote: Total 4440 (delta 332), reused 439 (delta 312), pack-reused 3977[K
    Receiving objects: 100% (4440/4440), 210.50 MiB | 13.74 MiB/s, done.
    Resolving deltas: 100% (1355/1355), done.
    Checking out files: 100% (2381/2381), done.
    /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel


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

    /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel


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
    remote: Total 5532 (delta 1), reused 0 (delta 0), pack-reused 5525
    Receiving objects: 100% (5532/5532), 8.15 MiB | 7.60 MiB/s, done.
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


## Generate Weights

With the images written to disk, we can run the Ultralytics training algorithm. I loved watching the progress fly by in real time on the original `train.py`. Fortunately, the Ultralytics folk write the results file to disk so the model's training data is still accessible!

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



View the last couple lines 

```
with open(aw.last_results_path + "results.txt") as results_file:
  results = results_file.readlines()
print("Epoch   gpu_mem       box       obj       cls     total    labels  img_size")
results[-5:]
```

    Epoch   gpu_mem       box       obj       cls     total    labels  img_size





    [' 1995/1999      1.8G   0.02351   0.01559  0.006725   0.04583       125       416    0.9915    0.9908    0.9929    0.8725   0.02014   0.01494  0.004582\n',
     ' 1996/1999      1.8G   0.02363   0.01608  0.006827   0.04654       150       416    0.9915    0.9909     0.993    0.8726   0.02014   0.01495  0.004582\n',
     ' 1997/1999      1.8G    0.0242   0.01487  0.007266   0.04633       161       416    0.9914    0.9909     0.993    0.8725   0.02014   0.01494  0.004582\n',
     ' 1998/1999      1.8G   0.02356   0.01581  0.006952   0.04632       102       416    0.9915    0.9909     0.993    0.8726   0.02014   0.01493  0.004582\n',
     ' 1999/1999      1.8G   0.02305   0.01591  0.006753   0.04571       185       416    0.9915    0.9909    0.9929    0.8722   0.02014   0.01492  0.004582\n']



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

    Fusing layers... 


Selects all images in the unlabeled folder and let's us look through the computer's eyes at the images.

```
%matplotlib inline 
import random, glob

images = glob.glob(f"./{unlabeled_images}/*.jpg")

for image in random.sample(images,5):
  v.plot_for(image)
```


    Output hidden; open in https://colab.research.google.com to view.


```
results = []
for image in images:
  results.append(v.predict_for(image))
```

    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/0.jpg: >>> [{'predictions': ['0 0.0328125 0.280556 0.065625 0.55 0.530924']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/1.jpg: >>> [{'predictions': ['6 0.416406 0.243056 0.01875 0.0722222 0.416284']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/10.jpg: >>> [{'predictions': ['0 0.0335938 0.280556 0.0671875 0.547222 0.475971']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/100.jpg: >>> [{'predictions': ['9 0.415625 0.25 0.021875 0.0833333 0.748945', '3 0.442969 0.253472 0.01875 0.0763889 0.763084', '6 0.493359 0.257639 0.0179687 0.0791667 0.852358', '1 0.467578 0.254861 0.0210938 0.0791667 0.897831']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/101.jpg: >>> [{'predictions': ['2 0.491406 0.256944 0.0203125 0.0777778 0.608123', '8 0.491016 0.257639 0.0210938 0.0791667 0.733652', '3 0.442578 0.252778 0.0195312 0.075 0.746644', '1 0.467187 0.255556 0.0203125 0.0777778 0.873402']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/102.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/103.jpg: >>> [{'predictions': ['3 0.441016 0.25 0.0195312 0.0777778 0.472052', '9 0.415234 0.252083 0.0226563 0.0847222 0.720322', '1 0.491016 0.259028 0.0210938 0.0819444 0.736842', '2 0.466797 0.254861 0.0195312 0.0791667 0.846333']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/104.jpg: >>> [{'predictions': ['9 0.415625 0.249306 0.021875 0.0819444 0.535528', '3 0.442187 0.252083 0.0203125 0.0763889 0.704703', '2 0.491406 0.257639 0.0203125 0.0819444 0.833294', '2 0.467187 0.253472 0.01875 0.0791667 0.890521']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/105.jpg: >>> [{'predictions': ['9 0.414844 0.249306 0.021875 0.0819444 0.623336', '4 0.490234 0.256944 0.0195312 0.0833333 0.645447', '3 0.442187 0.251389 0.0203125 0.0777778 0.781916', '2 0.467187 0.253472 0.01875 0.0791667 0.870233']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/106.jpg: >>> [{'predictions': ['0 0.034375 0.277778 0.06875 0.525 0.537866', '9 0.414062 0.250694 0.021875 0.0819444 0.60361', '2 0.466406 0.255556 0.01875 0.0805556 0.69404', '3 0.441797 0.252778 0.0195312 0.0777778 0.766312', '5 0.489453 0.258333 0.0195312 0.0805556 0.910713']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/107.jpg: >>> [{'predictions': ['9 0.414844 0.25 0.021875 0.0833333 0.52519', '3 0.441797 0.252778 0.0195312 0.0777778 0.76314', '2 0.467187 0.255556 0.01875 0.0777778 0.805961', '7 0.489844 0.257639 0.0203125 0.0819444 0.855903']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/108.jpg: >>> [{'predictions': ['3 0.467578 0.255556 0.0210938 0.075 0.51304', '9 0.467578 0.255556 0.0195312 0.075 0.555125', '9 0.416016 0.251389 0.0226563 0.0888889 0.693813', '3 0.442187 0.252778 0.0203125 0.075 0.772623', '9 0.491016 0.259028 0.0210938 0.0791667 0.822141']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/109.jpg: >>> [{'predictions': ['9 0.416016 0.252083 0.0226563 0.0875 0.704407', '3 0.442578 0.253472 0.0195312 0.0763889 0.728376', '3 0.467187 0.255556 0.021875 0.075 0.744717', '6 0.494922 0.258333 0.0164062 0.075 0.796405']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/11.jpg: >>> [{'predictions': ['0 0.0328125 0.269444 0.0640625 0.525 0.606911']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/110.jpg: >>> [{'predictions': ['9 0.415234 0.25 0.0226563 0.0833333 0.499961', '3 0.467187 0.254167 0.0203125 0.075 0.72737', '3 0.442187 0.252083 0.0203125 0.0763889 0.762814', '2 0.492188 0.256944 0.0203125 0.0833333 0.798111']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/111.jpg: >>> [{'predictions': ['9 0.414844 0.25 0.021875 0.0833333 0.649057', '3 0.467578 0.255556 0.0210938 0.075 0.77575', '3 0.442187 0.254167 0.0203125 0.075 0.795152', '3 0.492578 0.259028 0.0210938 0.0791667 0.844923']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/112.jpg: >>> [{'predictions': ['0 0.0332031 0.284722 0.0664062 0.536111 0.450711', '9 0.414844 0.25 0.021875 0.0833333 0.542981', '3 0.467187 0.255556 0.01875 0.075 0.615227', '3 0.442187 0.252778 0.0203125 0.0777778 0.809913', '5 0.489844 0.258333 0.0203125 0.0805556 0.860976']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/113.jpg: >>> [{'predictions': ['9 0.415625 0.252778 0.021875 0.0888889 0.591632', '3 0.466406 0.255556 0.0203125 0.075 0.650121', '3 0.441797 0.254167 0.0195312 0.075 0.749403', '6 0.491406 0.257639 0.01875 0.0763889 0.834653']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/114.jpg: >>> [{'predictions': ['3 0.467187 0.254861 0.0203125 0.0736111 0.561552', '2 0.491406 0.257639 0.0203125 0.0791667 0.577433', '8 0.491016 0.256944 0.0210938 0.0805556 0.580385', '9 0.416016 0.251389 0.0226563 0.0888889 0.715196', '3 0.442187 0.252083 0.0203125 0.0763889 0.798831']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/115.jpg: >>> [{'predictions': ['5 0.466016 0.255556 0.0210938 0.0805556 0.483577', '4 0.466406 0.254861 0.0203125 0.0791667 0.622532', '9 0.415234 0.248611 0.0226563 0.0833333 0.662227', '3 0.442187 0.252083 0.0203125 0.0791667 0.742917', '9 0.490625 0.258333 0.021875 0.0805556 0.861698']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/116.jpg: >>> [{'predictions': ['9 0.415234 0.249306 0.0226563 0.0847222 0.627369', '3 0.442187 0.252083 0.0203125 0.0763889 0.735727', '4 0.465625 0.252083 0.0203125 0.0819444 0.81888', '1 0.491016 0.256944 0.0210938 0.0805556 0.892418']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/117.jpg: >>> [{'predictions': ['9 0.416016 0.247917 0.0226563 0.0847222 0.498683', '4 0.466797 0.252083 0.0195312 0.0819444 0.600431', '3 0.442578 0.251389 0.0195312 0.0777778 0.691495', '2 0.491406 0.254861 0.0203125 0.0819444 0.828464']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/118.jpg: >>> [{'predictions': ['9 0.416797 0.250694 0.0226563 0.0902778 0.67716', '4 0.466406 0.252778 0.0203125 0.0805556 0.736248', '3 0.442578 0.251389 0.0210938 0.0777778 0.743895', '4 0.490625 0.25625 0.0203125 0.0819444 0.868969']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/119.jpg: >>> [{'predictions': ['9 0.415234 0.25 0.0226563 0.0833333 0.415597', '5 0.465625 0.254861 0.0203125 0.0819444 0.675315', '3 0.442187 0.252778 0.0203125 0.0777778 0.727142', '5 0.489844 0.258333 0.0203125 0.0805556 0.878106']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/12.jpg: >>> [{'predictions': ['0 0.03125 0.283333 0.0609375 0.522222 0.509543']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/120.jpg: >>> [{'predictions': ['4 0.465234 0.25625 0.0195312 0.0791667 0.442731', '5 0.464844 0.25625 0.0203125 0.0791667 0.496586', '9 0.490234 0.259028 0.0210938 0.0791667 0.557364', '7 0.490234 0.259028 0.0210938 0.0791667 0.686725', '9 0.415234 0.252778 0.0226563 0.0888889 0.718939', '3 0.441016 0.251389 0.0195312 0.0777778 0.786524']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/121.jpg: >>> [{'predictions': ['5 0.414062 0.253472 0.021875 0.0819444 0.613668', '9 0.464844 0.259722 0.021875 0.0805556 0.626974', '9 0.489062 0.2625 0.021875 0.0805556 0.734973', '4 0.439063 0.255556 0.0203125 0.0805556 0.80066']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/122.jpg: >>> [{'predictions': ['9 0.464844 0.260417 0.021875 0.0819444 0.787542', '4 0.439063 0.256944 0.0203125 0.0805556 0.801793', '9 0.414453 0.25625 0.0226563 0.0875 0.814777', '6 0.494141 0.261806 0.0148437 0.0763889 0.83072']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/123.jpg: >>> [{'predictions': ['5 0.441016 0.254861 0.0210938 0.0819444 0.694802', '5 0.416016 0.252778 0.0226563 0.0916667 0.758493', '4 0.440625 0.250694 0.0203125 0.0819444 0.824021', '9 0.467187 0.25625 0.0203125 0.0791667 0.848847', '2 0.491406 0.259028 0.0203125 0.0819444 0.86706']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/124.jpg: >>> [{'predictions': ['5 0.441016 0.255556 0.0210938 0.0833333 0.567948', '5 0.415234 0.252778 0.0226563 0.0916667 0.661084', '4 0.440625 0.251389 0.0203125 0.0833333 0.826705', '3 0.491406 0.259028 0.0203125 0.0763889 0.852668', '9 0.466797 0.25625 0.0210938 0.0791667 0.861994']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/125.jpg: >>> [{'predictions': ['5 0.441406 0.254861 0.021875 0.0847222 0.666777', '5 0.416016 0.252778 0.0226563 0.0916667 0.791452', '9 0.467187 0.25625 0.021875 0.0791667 0.797964', '4 0.441016 0.250694 0.0210938 0.0819444 0.813131', '5 0.489844 0.258333 0.0203125 0.0805556 0.837982']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/126.jpg: >>> [{'predictions': ['5 0.415234 0.252778 0.0226563 0.0888889 0.70852', '4 0.440625 0.252083 0.0203125 0.0819444 0.813636', '9 0.467187 0.256944 0.0203125 0.0805556 0.854431', '6 0.493359 0.259028 0.0179687 0.0791667 0.85452']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/127.jpg: >>> [{'predictions': ['2 0.491406 0.258333 0.0203125 0.0777778 0.467275', '5 0.441016 0.254861 0.0210938 0.0819444 0.511005', '5 0.415625 0.252778 0.0234375 0.0916667 0.729991', '4 0.440625 0.252083 0.0203125 0.0819444 0.786182', '9 0.467187 0.256944 0.0203125 0.0777778 0.851754']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/128.jpg: >>> [{'predictions': ['5 0.440625 0.254861 0.0203125 0.0819444 0.499951', '5 0.415234 0.253472 0.0226563 0.0902778 0.689276', '9 0.489844 0.259028 0.0203125 0.0791667 0.697994', '4 0.440234 0.250694 0.0195312 0.0819444 0.798594']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/129.jpg: >>> [{'predictions': ['5 0.441016 0.254861 0.0210938 0.0819444 0.500308', '5 0.415234 0.252778 0.0226563 0.0888889 0.609302', '4 0.440234 0.251389 0.0195312 0.0833333 0.818733', '1 0.490625 0.260417 0.0203125 0.0819444 0.863188']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/13.jpg: >>> [{'predictions': ['0 0.0347656 0.279861 0.0679687 0.540278 0.489764']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/130.jpg: >>> [{'predictions': ['5 0.441016 0.254861 0.0210938 0.0819444 0.663044', '3 0.491406 0.259722 0.0203125 0.0777778 0.732905', '4 0.440234 0.251389 0.0195312 0.0833333 0.770491', '5 0.415625 0.252778 0.021875 0.0888889 0.783725']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/131.jpg: >>> [{'predictions': ['5 0.441797 0.253472 0.0210938 0.0819444 0.679462', '4 0.441016 0.25 0.0195312 0.0833333 0.698165', '5 0.416016 0.252083 0.0226563 0.0902778 0.802477']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/132.jpg: >>> [{'predictions': ['5 0.441406 0.253472 0.0203125 0.0819444 0.448009', '6 0.471484 0.254861 0.0164062 0.0736111 0.554809', '5 0.415625 0.252083 0.021875 0.0902778 0.730235', '6 0.491406 0.256944 0.01875 0.0777778 0.733498', '4 0.440234 0.25 0.0195312 0.0833333 0.772164']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/133.jpg: >>> [{'predictions': ['5 0.441406 0.253472 0.0203125 0.0819444 0.636207', '5 0.415625 0.252083 0.021875 0.0902778 0.647305', '4 0.440625 0.25 0.0203125 0.0833333 0.742214', '7 0.490625 0.257639 0.0203125 0.0819444 0.793201']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/134.jpg: >>> [{'predictions': ['0 0.0324219 0.282639 0.0648438 0.548611 0.409083', '5 0.414062 0.252083 0.021875 0.0847222 0.589766', '1 0.466016 0.257639 0.0210938 0.0791667 0.718571', '9 0.490625 0.260417 0.021875 0.0791667 0.856464', '4 0.439844 0.252083 0.0203125 0.0819444 0.86927']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/135.jpg: >>> [{'predictions': ['9 0.415234 0.25 0.0226563 0.0833333 0.691182', '6 0.495703 0.258333 0.0148437 0.075 0.775117', '4 0.440625 0.25 0.0203125 0.0833333 0.830354', '1 0.466797 0.254861 0.0210938 0.0791667 0.880215']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/136.jpg: >>> [{'predictions': ['5 0.415625 0.252083 0.021875 0.0902778 0.683699', '2 0.491406 0.258333 0.0203125 0.0805556 0.817453', '1 0.467187 0.255556 0.0203125 0.0805556 0.851592', '4 0.440625 0.250694 0.0203125 0.0819444 0.854993']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/137.jpg: >>> [{'predictions': ['5 0.416016 0.252083 0.0226563 0.0902778 0.788427', '3 0.492188 0.257639 0.0203125 0.0736111 0.830645', '4 0.440625 0.25 0.0203125 0.0833333 0.868591', '1 0.467187 0.254167 0.0203125 0.0777778 0.869087']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/138.jpg: >>> [{'predictions': ['1 0.466406 0.254861 0.0203125 0.0791667 0.616201', '4 0.440625 0.25 0.0203125 0.0833333 0.784155', '5 0.415625 0.252083 0.021875 0.0902778 0.806727', '5 0.490234 0.259028 0.0210938 0.0819444 0.840643']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/139.jpg: >>> [{'predictions': ['9 0.491016 0.258333 0.0210938 0.0805556 0.509927', '5 0.416016 0.251389 0.0226563 0.0916667 0.716302', '4 0.440625 0.248611 0.0203125 0.0833333 0.814009', '1 0.467187 0.254861 0.0203125 0.0791667 0.865454']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/14.jpg: >>> [{'predictions': ['0 0.0324219 0.282639 0.0632813 0.554167 0.441108']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/140.jpg: >>> [{'predictions': ['0 0.0316406 0.2875 0.0617188 0.55 0.415913', '2 0.491406 0.257639 0.0203125 0.0791667 0.435928', '5 0.416016 0.250694 0.0226563 0.0902778 0.740381', '8 0.491016 0.257639 0.0210938 0.0819444 0.806298', '4 0.440625 0.249306 0.0203125 0.0819444 0.83453', '1 0.466797 0.254861 0.0210938 0.0791667 0.868339']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/141.jpg: >>> [{'predictions': ['6 0.497266 0.257639 0.0148437 0.0763889 0.594222', '4 0.441016 0.247222 0.0195312 0.0861111 0.679794', '5 0.416406 0.249306 0.021875 0.0902778 0.764394', '2 0.468359 0.250694 0.0195312 0.0819444 0.81154']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/142.jpg: >>> [{'predictions': ['5 0.416016 0.249306 0.0210938 0.0847222 0.53737', '4 0.440625 0.248611 0.0203125 0.0833333 0.830463', '2 0.466797 0.252778 0.0195312 0.0805556 0.835491', '1 0.491016 0.258333 0.0210938 0.0833333 0.886099']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/143.jpg: >>> [{'predictions': ['5 0.416016 0.252083 0.0226563 0.0902778 0.660482', '3 0.492188 0.256944 0.0203125 0.0777778 0.829653', '4 0.440625 0.248611 0.0203125 0.0833333 0.83757', '2 0.467578 0.252083 0.0195312 0.0819444 0.861442']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/144.jpg: >>> [{'predictions': ['5 0.489062 0.260417 0.0203125 0.0819444 0.504921', '0 0.0328125 0.279861 0.0640625 0.531944 0.523078', '4 0.489062 0.259028 0.0203125 0.0819444 0.562677', '2 0.465234 0.256944 0.0195312 0.0805556 0.617502', '4 0.439844 0.252778 0.0203125 0.0833333 0.807876', '9 0.414453 0.254861 0.0226563 0.0875 0.81793']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/145.jpg: >>> [{'predictions': ['5 0.414453 0.252778 0.0210938 0.0833333 0.522315', '9 0.414844 0.254167 0.021875 0.0888889 0.558095', '2 0.465625 0.258333 0.01875 0.0805556 0.804543', '6 0.492188 0.259722 0.01875 0.0777778 0.807951', '4 0.439844 0.253472 0.0203125 0.0819444 0.854412']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/146.jpg: >>> [{'predictions': ['5 0.415625 0.252083 0.021875 0.0902778 0.761773', '2 0.467187 0.254861 0.01875 0.0791667 0.781848', '4 0.440625 0.25 0.0203125 0.0833333 0.807098', '7 0.490234 0.257639 0.0210938 0.0819444 0.880993']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/147.jpg: >>> [{'predictions': ['4 0.440625 0.247222 0.0203125 0.0833333 0.643946', '3 0.467578 0.252778 0.0195312 0.075 0.772553', '9 0.491016 0.257639 0.0210938 0.0791667 0.7827']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/148.jpg: >>> [{'predictions': ['5 0.416406 0.246528 0.021875 0.0847222 0.561238', '6 0.495703 0.255556 0.0179687 0.0777778 0.768193', '4 0.440625 0.247222 0.0203125 0.0833333 0.781241', '3 0.467578 0.251389 0.0210938 0.075 0.819548']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/149.jpg: >>> [{'predictions': ['5 0.415234 0.250694 0.0226563 0.0902778 0.686033', '3 0.466406 0.252083 0.0203125 0.0763889 0.696063', '2 0.490625 0.256944 0.0203125 0.0805556 0.820772', '4 0.439844 0.248611 0.0203125 0.0833333 0.872039']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/15.jpg: >>> [{'predictions': ['0 0.0339844 0.290972 0.0679687 0.559722 0.444698']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/150.jpg: >>> [{'predictions': ['9 0.414844 0.250694 0.021875 0.0902778 0.552354', '3 0.466406 0.252083 0.0203125 0.0736111 0.706835', '4 0.439844 0.248611 0.0203125 0.0833333 0.809988', '3 0.491406 0.256944 0.021875 0.0777778 0.861746']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/151.jpg: >>> [{'predictions': ['5 0.414844 0.250694 0.021875 0.0902778 0.4768', '3 0.465234 0.254167 0.0195312 0.075 0.68376', '5 0.489062 0.256944 0.0203125 0.0805556 0.866906', '4 0.439063 0.248611 0.0203125 0.0833333 0.88623']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/152.jpg: >>> [{'predictions': ['5 0.415234 0.250694 0.0226563 0.0902778 0.560546', '3 0.466406 0.254167 0.0203125 0.075 0.776057', '7 0.490234 0.25625 0.0210938 0.0819444 0.875154', '4 0.439844 0.249306 0.0203125 0.0819444 0.892946']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/153.jpg: >>> [{'predictions': ['0 0.0328125 0.276389 0.0640625 0.544444 0.418655', '5 0.414844 0.249306 0.021875 0.0847222 0.448614', '2 0.491406 0.25625 0.0203125 0.0791667 0.484924', '8 0.490234 0.255556 0.0210938 0.0805556 0.658421', '3 0.466016 0.252778 0.0195312 0.075 0.698357', '4 0.439844 0.249306 0.0203125 0.0819444 0.886327']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/154.jpg: >>> [{'predictions': ['5 0.415234 0.250694 0.0226563 0.0902778 0.595315', '6 0.495703 0.256944 0.0148437 0.075 0.721271', '5 0.464844 0.252083 0.0203125 0.0819444 0.741625', '4 0.439844 0.248611 0.0203125 0.0833333 0.875832']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/155.jpg: >>> [{'predictions': ['5 0.414844 0.251389 0.021875 0.0888889 0.684258', '4 0.464453 0.253472 0.0210938 0.0819444 0.716626', '1 0.490625 0.258333 0.021875 0.0805556 0.894428', '4 0.439844 0.25 0.0203125 0.0833333 0.894446']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/156.jpg: >>> [{'predictions': ['5 0.415234 0.250694 0.0226563 0.0902778 0.556791', '3 0.490625 0.25625 0.0203125 0.0763889 0.883798', '4 0.440234 0.248611 0.0210938 0.0833333 0.892433']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/157.jpg: >>> [{'predictions': ['9 0.414062 0.249306 0.021875 0.0847222 0.463265', '5 0.414453 0.25 0.0226563 0.0861111 0.550859', '4 0.465234 0.252083 0.0210938 0.0819444 0.799759', '4 0.489844 0.25625 0.0203125 0.0819444 0.872286', '4 0.439844 0.249306 0.0203125 0.0847222 0.897688']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/158.jpg: >>> [{'predictions': ['5 0.464844 0.253472 0.0203125 0.0819444 0.500673', '5 0.415234 0.251389 0.0226563 0.0888889 0.638368', '6 0.491797 0.25625 0.0179687 0.0763889 0.811067', '4 0.439844 0.249306 0.0203125 0.0819444 0.886714']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/159.jpg: >>> [{'predictions': ['5 0.414844 0.249306 0.021875 0.0847222 0.476222', '5 0.464844 0.254861 0.0203125 0.0819444 0.569472', '4 0.465234 0.253472 0.0210938 0.0819444 0.749218', '7 0.490234 0.256944 0.0210938 0.0833333 0.860763', '4 0.440234 0.249306 0.0210938 0.0819444 0.896604']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/16.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/160.jpg: >>> [{'predictions': ['5 0.415234 0.251389 0.0226563 0.0916667 0.678887', '9 0.466406 0.254861 0.0203125 0.0791667 0.826533', '9 0.489844 0.258333 0.021875 0.0805556 0.86428', '5 0.440234 0.249306 0.0210938 0.0819444 0.903562']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/161.jpg: >>> [{'predictions': ['5 0.414844 0.250694 0.021875 0.0902778 0.775541', '6 0.495703 0.258333 0.0148437 0.075 0.791485', '9 0.466406 0.254861 0.0203125 0.0791667 0.795379', '5 0.439844 0.250694 0.0203125 0.0819444 0.917712']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/162.jpg: >>> [{'predictions': ['5 0.414844 0.252083 0.021875 0.0902778 0.639694', '2 0.490625 0.259028 0.0203125 0.0819444 0.781947', '9 0.465625 0.25625 0.0203125 0.0791667 0.815748', '5 0.439844 0.25 0.0203125 0.0805556 0.912402']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/163.jpg: >>> [{'predictions': ['3 0.489062 0.259028 0.0203125 0.0763889 0.488803', '5 0.414062 0.250694 0.021875 0.0847222 0.505361', '9 0.414062 0.25 0.021875 0.0833333 0.632208', '9 0.465234 0.25625 0.0210938 0.0791667 0.777596', '5 0.439844 0.250694 0.0203125 0.0819444 0.916783']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/164.jpg: >>> [{'predictions': ['0 0.0328125 0.279167 0.0640625 0.555556 0.503743', '5 0.415234 0.251389 0.0226563 0.0916667 0.695783', '9 0.466016 0.254167 0.0210938 0.0805556 0.798413', '5 0.489453 0.257639 0.0210938 0.0819444 0.840198', '5 0.440234 0.249306 0.0210938 0.0819444 0.907073']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/165.jpg: >>> [{'predictions': ['5 0.415234 0.251389 0.0226563 0.0916667 0.779934', '7 0.490234 0.257639 0.0210938 0.0819444 0.848766', '9 0.466016 0.254861 0.0210938 0.0791667 0.859755', '5 0.440234 0.249306 0.0210938 0.0819444 0.916485']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/166.jpg: >>> [{'predictions': ['9 0.414844 0.248611 0.021875 0.0833333 0.416806', '2 0.490625 0.258333 0.0203125 0.0805556 0.477528', '5 0.415234 0.251389 0.0226563 0.0916667 0.710012', '9 0.466406 0.255556 0.0203125 0.0777778 0.821935', '5 0.439844 0.25 0.0203125 0.0805556 0.912385']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/167.jpg: >>> [{'predictions': ['6 0.469922 0.255556 0.0164062 0.075 0.734133', '5 0.414844 0.252083 0.021875 0.0902778 0.737497', '6 0.494141 0.258333 0.0164062 0.075 0.767374', '5 0.439844 0.249306 0.0203125 0.0819444 0.89647']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/168.jpg: >>> [{'predictions': ['5 0.415625 0.250694 0.021875 0.0902778 0.774226', '5 0.440234 0.247917 0.0195312 0.0819444 0.883466', '1 0.490234 0.259028 0.0210938 0.0819444 0.88975']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/169.jpg: >>> [{'predictions': ['5 0.414844 0.250694 0.021875 0.0902778 0.77136', '3 0.491016 0.257639 0.0210938 0.0791667 0.844699', '5 0.439844 0.249306 0.0203125 0.0819444 0.897057']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/17.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/170.jpg: >>> [{'predictions': ['5 0.415625 0.250694 0.021875 0.0902778 0.717985', '5 0.439844 0.249306 0.0203125 0.0819444 0.898353']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/171.jpg: >>> [{'predictions': ['6 0.471484 0.254167 0.0179687 0.075 0.618823', '6 0.490625 0.255556 0.01875 0.0777778 0.783213', '5 0.415625 0.250694 0.021875 0.0902778 0.822995', '5 0.440234 0.249306 0.0195312 0.0819444 0.895591']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/172.jpg: >>> [{'predictions': ['5 0.414844 0.252083 0.021875 0.0902778 0.687553', '5 0.439453 0.249306 0.0195312 0.0819444 0.911065']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/173.jpg: >>> [{'predictions': ['1 0.465234 0.256944 0.0210938 0.0777778 0.679153', '5 0.414844 0.252083 0.021875 0.0902778 0.749061', '9 0.489844 0.259722 0.021875 0.0805556 0.839398', '5 0.439844 0.25 0.0203125 0.0805556 0.915814']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/174.jpg: >>> [{'predictions': ['5 0.415625 0.25 0.021875 0.0916667 0.767872', '1 0.466016 0.253472 0.0210938 0.0791667 0.877861', '1 0.491016 0.259028 0.0210938 0.0819444 0.881671', '5 0.439844 0.247917 0.0203125 0.0819444 0.90065']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/175.jpg: >>> [{'predictions': ['5 0.415625 0.251389 0.021875 0.0888889 0.672202', '1 0.465625 0.254861 0.0203125 0.0791667 0.821184', '2 0.490625 0.257639 0.0203125 0.0819444 0.825492', '5 0.439844 0.248611 0.0203125 0.0805556 0.908554']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/176.jpg: >>> [{'predictions': ['5 0.414844 0.250694 0.021875 0.0902778 0.747697', '1 0.466016 0.254167 0.0210938 0.0805556 0.822749', '4 0.489844 0.257639 0.0203125 0.0819444 0.865554', '5 0.439844 0.249306 0.0203125 0.0819444 0.909206']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/177.jpg: >>> [{'predictions': ['9 0.414062 0.248611 0.021875 0.0833333 0.455035', '1 0.465625 0.254861 0.0203125 0.0791667 0.541002', '5 0.414062 0.249306 0.021875 0.0847222 0.619048', '5 0.489453 0.258333 0.0210938 0.0805556 0.885181', '5 0.439844 0.249306 0.0203125 0.0819444 0.917361']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/178.jpg: >>> [{'predictions': ['7 0.490234 0.258333 0.0210938 0.0805556 0.60435', '5 0.414844 0.250694 0.021875 0.0902778 0.720329', '1 0.465625 0.254861 0.0203125 0.0791667 0.787759', '5 0.439844 0.248611 0.0203125 0.0805556 0.91396']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/179.jpg: >>> [{'predictions': ['5 0.415234 0.251389 0.0226563 0.0916667 0.668751', '8 0.490234 0.258333 0.0210938 0.0805556 0.781107', '1 0.465625 0.255556 0.0203125 0.0777778 0.82918', '5 0.439844 0.25 0.0203125 0.0805556 0.915755']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/18.jpg: >>> [{'predictions': ['1 0.416406 0.246528 0.0203125 0.0763889 0.425353']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/180.jpg: >>> [{'predictions': ['5 0.414453 0.252083 0.0226563 0.0902778 0.667557', '6 0.494922 0.258333 0.0148437 0.075 0.733071', '2 0.466016 0.254167 0.0195312 0.0805556 0.820734', '5 0.439844 0.249306 0.0203125 0.0819444 0.876945']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/181.jpg: >>> [{'predictions': ['2 0.465234 0.256944 0.0195312 0.0805556 0.610175', '9 0.414062 0.252778 0.021875 0.0888889 0.718542', '1 0.489453 0.259028 0.0210938 0.0819444 0.8491', '5 0.439063 0.251389 0.0203125 0.0805556 0.908739']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/182.jpg: >>> [{'predictions': ['9 0.414062 0.25 0.021875 0.0833333 0.421012', '5 0.414062 0.25 0.021875 0.0833333 0.558475', '3 0.490234 0.257639 0.0210938 0.0763889 0.828696', '2 0.466016 0.254167 0.0195312 0.0805556 0.829036', '5 0.439844 0.249306 0.0203125 0.0819444 0.908518']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/183.jpg: >>> [{'predictions': ['2 0.465234 0.255556 0.0195312 0.0805556 0.691984', '9 0.414453 0.252778 0.0226563 0.0888889 0.698392', '5 0.488281 0.259722 0.0203125 0.0805556 0.792294', '5 0.439453 0.250694 0.0195312 0.0819444 0.893807']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/184.jpg: >>> [{'predictions': ['2 0.466016 0.255556 0.0195312 0.0805556 0.722424', '9 0.414062 0.253472 0.021875 0.0875 0.722945', '6 0.491016 0.259028 0.0179687 0.0791667 0.826105', '5 0.439453 0.251389 0.0195312 0.0805556 0.896368']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/185.jpg: >>> [{'predictions': ['2 0.491797 0.256944 0.0195312 0.0777778 0.5122', '8 0.489844 0.255556 0.0203125 0.0833333 0.659221', '5 0.415625 0.250694 0.021875 0.0902778 0.750235', '2 0.466797 0.252778 0.0195312 0.0805556 0.868032', '5 0.439844 0.247917 0.0203125 0.0819444 0.893572']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/186.jpg: >>> [{'predictions': ['3 0.466406 0.25625 0.01875 0.0736111 0.461817', '5 0.415234 0.250694 0.0226563 0.0930556 0.650718', '9 0.490234 0.259028 0.0210938 0.0791667 0.874015', '5 0.439844 0.248611 0.0203125 0.0805556 0.886372']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/187.jpg: >>> [{'predictions': ['9 0.414062 0.248611 0.021875 0.0833333 0.408996', '5 0.414844 0.251389 0.021875 0.0888889 0.567436', '1 0.490625 0.259028 0.0203125 0.0819444 0.73783', '3 0.466016 0.254167 0.0195312 0.075 0.778559', '5 0.439453 0.249306 0.0210938 0.0819444 0.909735']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/188.jpg: >>> [{'predictions': ['3 0.465625 0.254861 0.0203125 0.0763889 0.600105', '5 0.415234 0.251389 0.0226563 0.0888889 0.631589', '2 0.490625 0.257639 0.0203125 0.0819444 0.86504', '5 0.439844 0.249306 0.0203125 0.0819444 0.901923']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/189.jpg: >>> [{'predictions': ['4 0.489844 0.256944 0.0203125 0.0805556 0.44551', '3 0.466016 0.254861 0.0195312 0.0736111 0.580933', '5 0.415234 0.252083 0.0226563 0.0902778 0.7793', '5 0.439844 0.25 0.0203125 0.0805556 0.901729']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/19.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/190.jpg: >>> [{'predictions': ['0 0.0328125 0.282639 0.0640625 0.554167 0.415644', '5 0.414844 0.250694 0.021875 0.0902778 0.658387', '3 0.466406 0.254861 0.01875 0.0736111 0.685325', '5 0.439844 0.249306 0.0203125 0.0819444 0.892058']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/191.jpg: >>> [{'predictions': ['0 0.0328125 0.2875 0.0640625 0.544444 0.402637', '3 0.466016 0.256944 0.0195312 0.075 0.574981', '5 0.414453 0.252083 0.0226563 0.0902778 0.767725', '7 0.490234 0.258333 0.0210938 0.0805556 0.856493', '5 0.439453 0.250694 0.0210938 0.0819444 0.920978']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/192.jpg: >>> [{'predictions': ['5 0.464844 0.256944 0.0203125 0.0805556 0.418299', '9 0.414062 0.25 0.021875 0.0833333 0.472152', '0 0.0320312 0.277083 0.0625 0.534722 0.500885', '5 0.415234 0.252083 0.0226563 0.0902778 0.555494', '9 0.489453 0.259028 0.0210938 0.0791667 0.833306', '5 0.440234 0.250694 0.0210938 0.0819444 0.908911']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/193.jpg: >>> [{'predictions': ['4 0.465625 0.252083 0.0203125 0.0819444 0.559989', '5 0.416016 0.25 0.0226563 0.0916667 0.60566', '6 0.496094 0.256944 0.0140625 0.075 0.662695', '5 0.465625 0.253472 0.0203125 0.0819444 0.728129', '5 0.440625 0.247917 0.0203125 0.0819444 0.870818']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/194.jpg: >>> [{'predictions': ['5 0.414844 0.251389 0.021875 0.0888889 0.734659', '2 0.490234 0.25625 0.0195312 0.0819444 0.871983', '5 0.439844 0.249306 0.0203125 0.0819444 0.89644']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/195.jpg: >>> [{'predictions': ['4 0.464063 0.253472 0.0203125 0.0791667 0.439217', '5 0.415234 0.252083 0.0226563 0.0902778 0.601774', '3 0.490625 0.256944 0.0203125 0.075 0.872235', '5 0.439844 0.249306 0.0203125 0.0819444 0.906536']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/196.jpg: >>> [{'predictions': ['5 0.464844 0.25625 0.0203125 0.0819444 0.589971', '5 0.415234 0.252083 0.0226563 0.0902778 0.658692', '5 0.439844 0.249306 0.0203125 0.0819444 0.902189', '5 0.489062 0.258333 0.0203125 0.0805556 0.905443']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/197.jpg: >>> [{'predictions': ['9 0.414062 0.253472 0.021875 0.0902778 0.606996', '6 0.491406 0.258333 0.01875 0.0777778 0.823942', '5 0.439453 0.251389 0.0195312 0.0805556 0.911321']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/198.jpg: >>> [{'predictions': ['9 0.414062 0.25 0.021875 0.0833333 0.499768', '5 0.414844 0.252083 0.021875 0.0902778 0.575845', '4 0.464844 0.254861 0.0203125 0.0791667 0.746769', '8 0.489453 0.256944 0.0210938 0.0805556 0.825187', '5 0.439844 0.250694 0.0203125 0.0819444 0.916206']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/199.jpg: >>> [{'predictions': ['5 0.414062 0.250694 0.021875 0.0847222 0.487667', '9 0.414844 0.252778 0.021875 0.0888889 0.526367', '6 0.444141 0.255556 0.0179687 0.0805556 0.564233', '9 0.465625 0.25625 0.0203125 0.0791667 0.699249', '9 0.489453 0.259722 0.0210938 0.0805556 0.853971']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/2.jpg: >>> [{'predictions': ['6 0.417188 0.242361 0.01875 0.0736111 0.42554']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/20.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/21.jpg: >>> [{'predictions': ['6 0.467969 0.253472 0.0171875 0.0736111 0.413683']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/22.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/23.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/24.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/25.jpg: >>> [{'predictions': ['6 0.467969 0.253472 0.01875 0.0708333 0.487878']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/26.jpg: >>> [{'predictions': ['6 0.492578 0.257639 0.0179687 0.0736111 0.541864', '6 0.468359 0.251389 0.0164062 0.0694444 0.565136']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/27.jpg: >>> [{'predictions': ['6 0.442969 0.244444 0.01875 0.0722222 0.409742', '6 0.470313 0.250694 0.01875 0.0708333 0.575003']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/28.jpg: >>> [{'predictions': ['6 0.491406 0.258333 0.01875 0.0722222 0.413778', '6 0.467578 0.252778 0.0164062 0.0694444 0.476028']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/29.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/3.jpg: >>> [{'predictions': ['6 0.416797 0.241667 0.0195312 0.075 0.457185']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/30.jpg: >>> [{'predictions': ['0 0.0324219 0.279167 0.0632813 0.538889 0.425715', '6 0.491797 0.258333 0.0179687 0.0722222 0.441006', '6 0.468359 0.252778 0.0164062 0.0694444 0.460474']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/31.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/32.jpg: >>> [{'predictions': ['6 0.492188 0.256944 0.01875 0.0722222 0.484996']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/33.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/34.jpg: >>> [{'predictions': ['6 0.489844 0.259028 0.01875 0.0736111 0.472251']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/35.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/36.jpg: >>> [{'predictions': ['6 0.491016 0.256944 0.0179687 0.0722222 0.54149']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/37.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/38.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/39.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/4.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/40.jpg: >>> [{'predictions': ['6 0.469141 0.254167 0.0195312 0.0722222 0.403822']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/41.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/42.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/43.jpg: >>> [{'predictions': ['6 0.491797 0.256944 0.0179687 0.0722222 0.41309']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/44.jpg: >>> [{'predictions': ['6 0.469531 0.254167 0.01875 0.0722222 0.414751']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/45.jpg: >>> [{'predictions': ['6 0.467969 0.254167 0.0171875 0.0722222 0.450332']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/46.jpg: >>> [{'predictions': ['6 0.469531 0.254167 0.01875 0.0722222 0.461863']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/47.jpg: >>> [{'predictions': ['6 0.468359 0.254167 0.0179687 0.0722222 0.424109']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/48.jpg: >>> [{'predictions': ['6 0.469531 0.253472 0.0171875 0.0708333 0.437441']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/49.jpg: >>> [{'predictions': ['6 0.46875 0.253472 0.0171875 0.0708333 0.431172', '6 0.492188 0.257639 0.01875 0.0708333 0.446441']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/5.jpg: >>> [{'predictions': ['0 0.0332031 0.280556 0.0648438 0.527778 0.431499']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/50.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/51.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/52.jpg: >>> [{'predictions': ['6 0.492188 0.256944 0.01875 0.0722222 0.416191']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/53.jpg: >>> [{'predictions': ['6 0.476172 0.25 0.0179687 0.0638889 0.40095', '6 0.443359 0.24375 0.0195312 0.0680556 0.436533']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/54.jpg: >>> [{'predictions': ['6 0.491016 0.259028 0.0179687 0.0736111 0.451997']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/55.jpg: >>> [{'predictions': ['6 0.491016 0.258333 0.0179687 0.0722222 0.530453']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/56.jpg: >>> [{'predictions': ['6 0.491016 0.258333 0.0179687 0.0722222 0.500523']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/57.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/58.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/59.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/6.jpg: >>> [{'predictions': ['0 0.0328125 0.274306 0.0640625 0.526389 0.511489']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/60.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/61.jpg: image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/62.jpg: >>> [{'predictions': ['6 0.491797 0.258333 0.0179687 0.0722222 0.464498', '6 0.467969 0.252778 0.0171875 0.0694444 0.481104']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/63.jpg: >>> [{'predictions': ['6 0.489844 0.2625 0.0203125 0.0805556 0.527296', '6 0.466406 0.25625 0.01875 0.0791667 0.563262', '5 0.414844 0.252083 0.021875 0.0875 0.797202']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/64.jpg: >>> [{'predictions': ['5 0.415625 0.250694 0.021875 0.0847222 0.469272', '6 0.492188 0.258333 0.0171875 0.0777778 0.728418', '2 0.441797 0.254167 0.0195312 0.0833333 0.741133', '2 0.466406 0.253472 0.01875 0.0791667 0.819917']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/65.jpg: >>> [{'predictions': ['5 0.415625 0.252778 0.021875 0.0888889 0.623984', '2 0.441797 0.254861 0.0195312 0.0819444 0.736859', '2 0.490234 0.259028 0.0195312 0.0819444 0.80444', '2 0.466406 0.254861 0.01875 0.0819444 0.868972']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/66.jpg: >>> [{'predictions': ['5 0.415234 0.252083 0.0210938 0.0875 0.426197', '2 0.441797 0.253472 0.0195312 0.0819444 0.758322', '2 0.466406 0.254167 0.01875 0.0805556 0.803234']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/67.jpg: >>> [{'predictions': ['9 0.414844 0.254861 0.021875 0.0875 0.792049', '2 0.440234 0.253472 0.0195312 0.0819444 0.816941', '5 0.489062 0.261111 0.0203125 0.0805556 0.87344']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/68.jpg: >>> [{'predictions': ['5 0.415625 0.25 0.021875 0.0833333 0.451701', '2 0.441797 0.252778 0.0195312 0.0833333 0.745301', '2 0.467187 0.254861 0.01875 0.0791667 0.804396', '7 0.490625 0.258333 0.0203125 0.0805556 0.82509']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/69.jpg: >>> [{'predictions': ['5 0.415625 0.250694 0.021875 0.0847222 0.556853', '2 0.491797 0.258333 0.0195312 0.0777778 0.635582', '2 0.441797 0.252778 0.0195312 0.0805556 0.759009', '2 0.467187 0.254167 0.01875 0.0805556 0.864282']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/7.jpg: >>> [{'predictions': ['0 0.0332031 0.28125 0.0648438 0.540278 0.44109']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/70.jpg: >>> [{'predictions': ['5 0.415625 0.249306 0.021875 0.0847222 0.52293', '2 0.442187 0.254167 0.01875 0.0805556 0.606425', '6 0.495703 0.258333 0.0164062 0.075 0.703863', '3 0.467187 0.254167 0.0203125 0.075 0.763363']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/71.jpg: >>> [{'predictions': ['5 0.414844 0.250694 0.021875 0.0847222 0.465772', '2 0.491797 0.258333 0.0210938 0.0805556 0.521264', '1 0.491797 0.259028 0.0210938 0.0819444 0.710908', '3 0.466797 0.254167 0.0195312 0.075 0.717472', '2 0.442187 0.253472 0.01875 0.0791667 0.725653']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/72.jpg: >>> [{'predictions': ['5 0.414844 0.25 0.021875 0.0861111 0.562175', '2 0.441797 0.253472 0.0195312 0.0791667 0.732331', '3 0.466797 0.254167 0.0210938 0.075 0.812031', '3 0.491406 0.257639 0.0203125 0.0763889 0.85392']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/73.jpg: >>> [{'predictions': ['5 0.490625 0.258333 0.0203125 0.0805556 0.476213', '3 0.467187 0.254167 0.01875 0.075 0.516816', '5 0.415625 0.249306 0.021875 0.0847222 0.554975', '2 0.442187 0.253472 0.01875 0.0791667 0.677563']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/74.jpg: >>> [{'predictions': ['3 0.466797 0.254861 0.0179687 0.0736111 0.643335', '2 0.441016 0.251389 0.0195312 0.0805556 0.788152', '6 0.492188 0.258333 0.01875 0.0777778 0.859165']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/75.jpg: >>> [{'predictions': ['2 0.491797 0.257639 0.0195312 0.0763889 0.489247', '5 0.414844 0.250694 0.021875 0.0847222 0.526641', '3 0.466797 0.254861 0.0195312 0.0736111 0.631197', '2 0.441016 0.250694 0.0195312 0.0819444 0.791436']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/76.jpg: >>> [{'predictions': ['5 0.414062 0.250694 0.021875 0.0847222 0.410352', '5 0.464844 0.256944 0.0203125 0.0805556 0.504446', '2 0.441406 0.251389 0.0203125 0.0833333 0.765702', '9 0.490625 0.260417 0.021875 0.0791667 0.894502']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/77.jpg: >>> [{'predictions': ['5 0.465625 0.254861 0.0203125 0.0819444 0.489079', '5 0.415625 0.251389 0.021875 0.0888889 0.557909', '2 0.441797 0.250694 0.0195312 0.0819444 0.676088', '4 0.465625 0.253472 0.0203125 0.0819444 0.706635', '1 0.491797 0.259028 0.0210938 0.0819444 0.772315']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/78.jpg: >>> [{'predictions': ['5 0.415625 0.25 0.021875 0.0833333 0.437373', '2 0.441016 0.250694 0.0195312 0.0819444 0.718955', '2 0.491406 0.257639 0.0203125 0.0819444 0.806392']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/79.jpg: >>> [{'predictions': ['0 0.0347656 0.286111 0.0679687 0.561111 0.421883', '5 0.414844 0.250694 0.021875 0.0847222 0.437693', '4 0.465234 0.254861 0.0195312 0.0819444 0.616751', '2 0.442187 0.254167 0.0203125 0.0805556 0.625252', '4 0.490625 0.258333 0.0203125 0.0805556 0.751569']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/8.jpg: >>> [{'predictions': ['0 0.0328125 0.275 0.065625 0.519444 0.581705']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/80.jpg: >>> [{'predictions': ['2 0.442578 0.253472 0.0195312 0.0819444 0.598805', '5 0.415625 0.251389 0.021875 0.0888889 0.694197', '5 0.465625 0.254861 0.0203125 0.0819444 0.748185', '5 0.490234 0.258333 0.0210938 0.0805556 0.835795']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/81.jpg: >>> [{'predictions': ['9 0.491016 0.257639 0.0210938 0.0819444 0.418503', '5 0.416406 0.247917 0.021875 0.0847222 0.542749', '4 0.466406 0.252083 0.0203125 0.0819444 0.638337', '2 0.442578 0.252083 0.0195312 0.0819444 0.656531', '7 0.491016 0.25625 0.0210938 0.0819444 0.663753', '5 0.466406 0.252778 0.0203125 0.0805556 0.712766']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/82.jpg: >>> [{'predictions': ['4 0.466406 0.250694 0.0203125 0.0819444 0.503141', '5 0.416797 0.248611 0.0210938 0.0888889 0.680757', '2 0.491797 0.256944 0.0195312 0.0805556 0.692726', '2 0.442578 0.250694 0.0195312 0.0819444 0.693113', '8 0.491797 0.254861 0.0210938 0.0819444 0.839635']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/83.jpg: >>> [{'predictions': ['9 0.416797 0.25 0.0226563 0.0888889 0.656334', '3 0.442969 0.252083 0.0203125 0.0763889 0.768929', '6 0.496484 0.257639 0.0164062 0.0763889 0.804986', '9 0.468359 0.254167 0.0210938 0.0805556 0.84157']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/84.jpg: >>> [{'predictions': ['9 0.415234 0.249306 0.0226563 0.0819444 0.670994', '3 0.442578 0.253472 0.0210938 0.0763889 0.765678', '9 0.467187 0.255556 0.0203125 0.0777778 0.781437', '1 0.491797 0.258333 0.0210938 0.0805556 0.88715']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/85.jpg: >>> [{'predictions': ['9 0.416016 0.248611 0.0226563 0.0833333 0.742383', '3 0.442969 0.252083 0.0203125 0.0763889 0.775914', '9 0.467969 0.254167 0.0203125 0.0777778 0.875131', '3 0.492578 0.256944 0.0210938 0.075 0.877358']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/86.jpg: >>> [{'predictions': ['4 0.491016 0.25625 0.0210938 0.0819444 0.656851', '9 0.416016 0.248611 0.0226563 0.0833333 0.782337', '3 0.443359 0.251389 0.0210938 0.0777778 0.784245', '9 0.467969 0.253472 0.0203125 0.0791667 0.834896']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/87.jpg: >>> [{'predictions': ['3 0.442578 0.254167 0.0195312 0.0777778 0.739658', '9 0.416016 0.251389 0.0226563 0.0888889 0.807205', '9 0.468359 0.25625 0.0210938 0.0791667 0.831611', '6 0.494141 0.258333 0.0179687 0.0777778 0.849766']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/88.jpg: >>> [{'predictions': ['8 0.491016 0.258333 0.0210938 0.0805556 0.468699', '2 0.491406 0.257639 0.0203125 0.0791667 0.570692', '9 0.415234 0.25 0.0226563 0.0833333 0.686543', '3 0.442187 0.253472 0.0203125 0.0763889 0.763689', '9 0.467187 0.25625 0.0203125 0.0763889 0.856012']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/89.jpg: >>> [{'predictions': ['9 0.415234 0.252778 0.0226563 0.0888889 0.80504', '9 0.489453 0.259722 0.0210938 0.0805556 0.81971', '3 0.441797 0.254167 0.0210938 0.0777778 0.82736']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/9.jpg: >>> [{'predictions': ['0 0.034375 0.272917 0.0671875 0.515278 0.623043']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/90.jpg: >>> [{'predictions': ['3 0.442578 0.252778 0.0210938 0.0777778 0.774965', '9 0.416797 0.252083 0.0226563 0.0902778 0.805013', '1 0.491797 0.258333 0.0210938 0.0833333 0.883738']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/91.jpg: >>> [{'predictions': ['6 0.471875 0.254167 0.015625 0.075 0.511787', '9 0.414844 0.250694 0.021875 0.0819444 0.552591', '2 0.491797 0.258333 0.0195312 0.0833333 0.694827', '3 0.442187 0.252778 0.0203125 0.0777778 0.732888']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/92.jpg: >>> [{'predictions': ['0 0.0339844 0.288194 0.0679687 0.545833 0.433119', '4 0.490234 0.257639 0.0195312 0.0791667 0.452927', '9 0.414844 0.251389 0.021875 0.0833333 0.710365', '3 0.442187 0.252778 0.0203125 0.0777778 0.804282']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/93.jpg: >>> [{'predictions': ['9 0.415234 0.253472 0.0226563 0.0875 0.773935', '3 0.442187 0.254167 0.0203125 0.0777778 0.783402', '5 0.489453 0.258333 0.0195312 0.0805556 0.891346']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/94.jpg: >>> [{'predictions': ['3 0.442187 0.252778 0.0203125 0.0777778 0.79807', '7 0.491016 0.258333 0.0210938 0.0805556 0.839247', '9 0.415625 0.252778 0.0234375 0.0888889 0.847377']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/95.jpg: >>> [{'predictions': ['2 0.491016 0.258333 0.0195312 0.0805556 0.598596', '3 0.442187 0.252778 0.0203125 0.075 0.780456', '9 0.416016 0.252083 0.0226563 0.0875 0.797674']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/96.jpg: >>> [{'predictions': ['9 0.415625 0.252778 0.021875 0.0888889 0.715757', '3 0.442187 0.254167 0.01875 0.0777778 0.742117', '6 0.496094 0.258333 0.015625 0.075 0.840804', '1 0.467187 0.25625 0.021875 0.0791667 0.894684']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/97.jpg: >>> [{'predictions': ['9 0.415625 0.25 0.021875 0.0833333 0.507089', '3 0.442578 0.252778 0.0195312 0.075 0.751245', '2 0.491406 0.25625 0.0203125 0.0791667 0.838688', '1 0.467187 0.253472 0.0203125 0.0791667 0.896689']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/98.jpg: >>> [{'predictions': ['9 0.414844 0.25 0.021875 0.0833333 0.4445', '3 0.442578 0.252778 0.0195312 0.075 0.782793', '3 0.491406 0.258333 0.0203125 0.075 0.861713', '1 0.467187 0.254167 0.0203125 0.0777778 0.882078']}]
    image 1/1 /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00/99.jpg: >>> [{'predictions': ['9 0.416797 0.25 0.0226563 0.0888889 0.636944', '3 0.442578 0.252083 0.0195312 0.0763889 0.753047', '5 0.491016 0.258333 0.0210938 0.0805556 0.820021', '1 0.467187 0.254861 0.0203125 0.0791667 0.887824']}]


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

    Moving yolov5 results folder: yolov5/runs/train/seven segment digits - 1/


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

