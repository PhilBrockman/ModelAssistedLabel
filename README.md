# Model-asisted Labeling with YOLOv5
> bootstrapping image annotation


![base64 splash](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/modelassistedlabel%20splash.jpg?raw=true)

## Background

Object detection is great! ... if your labeled dataset already exists. I wanted to use machine learning to turn my regular rowing machine into a "smart" rowing machine (specifically: I want to track my workout stats).

Unfortunately, I was unable to find a suitable existing set of labeled LCD digits.

After working through [a Roboflow tutorial]( https://models.roboflow.com/object-detection/yolov5), I started to use Roboflow to annotate and store my images. 

And I hated annotating my images by hand. Once the models began making reasonable guesses, I enlisted the model's help in labeling new images. This repository is the result of these efforts.

(Later on, I developed a [custom React annotator](https://github.com/PhilBrockman/autobbox) as a curiousity. However, I labeled dozens upon dozens of images with Roboflow and would recommend their free annotation service.)

## Getting Started

```python
#Fresh colab installation:

!git clone https://github.com/PhilBrockman/ModelAssistedLabel.git
%cd "ModelAssistedLabel"
!pip install -i https://test.pypi.org/simple/ ModelAssistedLabel==1.1
```

### Expected Inputs:
* Both 
  - **labeled images**
      + All of the images and labels must be in a common folder (subfolders allowed).
      + labels must be in [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885).
{% include note.html content='Image/label pairs are based on their base filename. For example `image.jpg/image.txt` would be paired as would `other_image5.jpg/other_image5.txt`.' %}
* And:
  - **unlabeled images**





```python
# these images have already had the images labeled and verified by a human
labeled_images   = "./Image Repo/labeled/Final Roboflow Export (841)"

unlabeled_images_repos = [] #including a variety of lighting from three seperate recordingns
unlabeled_images_repos.append("Image Repo/unlabeled/21-3-18 rowing 8-12 /")
unlabeled_images_repos.append("Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00")
unlabeled_images_repos.append("Image Repo/unlabeled/21-3-22 rowing (200) 7:50-12:50")

unlabeled_images = unlabeled_images_repos[2] #for the sake of example, I'm selecting the images with the best lighting
```

### Expected Output:

* ***ZIP file*** that contains: 
    - `images/`
      + a copy of every image in **Unlabeled Data**
    - `labels/`
      + result of running object detection on each image
    - a results folder produced by Ultralytic's `train.py` on the **Labeled Data** (if not using pre-trained weights)
    - `classmap.yaml` to preserve the identity of the classes


## Configure defaults

Several values are stored by the `Defaults` class. Any value can be overridden (and new values can be added. Make sure to `save()` any changes!

```python
from ModelAssistedLabel.config import Defaults

d= Defaults()
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

```python
!pwd
```




    '/content'



```python
d.root = "/content/drive/MyDrive/Coding/ModelAssistedLabel/"
```

Save any changes

```python
d.save()
```

Enter root directory

```python
import os
os.chdir(Defaults().root)
```

Clone yolov5 repo and install requirements.

```python
Defaults.prepare_YOLOv5()
```

    Setup complete. Using torch 1.8.0+cu101 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)


## Processing input

Next, the images need to be written in a way so that the Ultralytics repository can understand their content. The `Autoweights` class both organizes data and create weights. Running an "initialize" command makes changes to the disk.

```python
ls
```

     00_config.ipynb         [0m[01;34mdocs[0m/                'ModelAssistedLabel config.json'
     01_split.ipynb         [01;34m'Image Repo'[0m/         'modelassistedlabel splash.jpg'
     02_train.ipynb          index.ipynb          [01;34m'pre-trained weights'[0m/
     03_detect.ipynb         [01;34mipynb_tests[0m/          README.md
    '_capture input.ipynb'   LICENSE               settings.ini
     CONTRIBUTING.md         Makefile              setup.py
     data.yaml               MANIFEST.in           _Synch.ipynb
     docker-compose.yml      [01;34mModelAssistedLabel[0m/   [01;34myolov5[0m/


```python
from ModelAssistedLabel.train import AutoWeights

datadump="ipynb_tests/index"

aw = AutoWeights(name="<index>", out_dir=datadump, MAX_SIZE=200)
aw.initialize_images_from_bag(labeled_images)
aw.traverse_resources()
```

    
    dirs ['./train', './valid', './test']
    yaml ipynb_tests/index/Final Roboflow Export (841)<index> 21-03-25 12-30-41/data.yaml
    subdir train
    	outdir ipynb_tests/index/Final Roboflow Export (841)<index> 21-03-25 12-30-41
    subdir valid
    	outdir ipynb_tests/index/Final Roboflow Export (841)<index> 21-03-25 12-30-41
    subdir test
    	outdir ipynb_tests/index/Final Roboflow Export (841)<index> 21-03-25 12-30-41
    os.listdir ['train', 'valid', 'test', 'data.yaml']
    train/images
    	 > 140 files
    train/labels
    	 > 140 files
    valid/images
    	 > 40 files
    valid/labels
    	 > 40 files
    test/images
    	 > 20 files
    test/labels
    	 > 20 files
    File:  data.yaml


## Generate Weights

With the images written to disk, we can run the Ultralytics training algorithm. On this dataset, I found 1200 epochs to be a reasonable stopping point but using even longer training times are not uncommon.

```python
%%time
aw.generate_weights(1000)
```

    CPU times: user 10.5 s, sys: 1.45 s, total: 11.9 s
    Wall time: 45min 23s





    'yolov5/runs/train/<index>'



The results folder is stored as an attribute as well, and it has a lot of data stored therein.

```python
aw.last_results_path, len(os.listdir(aw.last_results_path))
```




    ('yolov5/runs/train/<index>', 20)



However, the weights are stored in a subfolder called (aptly) "weights". I use `best.pt`.

```python
os.listdir(aw.last_results_path + "/weights")
```




    ['last.pt', 'best.pt']



View the last couple lines 

```python
with open(aw.last_results_path + "/results.txt") as results_file:
  results = results_file.readlines()
print("Epoch   gpu_mem       box       obj       cls     total    labels  img_size")
results[-5:]
```

    Epoch   gpu_mem       box       obj       cls     total    labels  img_size





    ['   995/999     1.82G   0.02979   0.02355   0.01262   0.06595       119       416    0.9787    0.9698    0.9861    0.8327   0.02502   0.01936  0.008843\n',
     '   996/999     1.82G   0.02952   0.02375   0.01236   0.06562       124       416    0.9785    0.9677    0.9861    0.8326   0.02496   0.01922  0.008919\n',
     '   997/999     1.82G   0.03078   0.02463   0.01184   0.06725       162       416    0.9719    0.9679    0.9859    0.8301   0.02492   0.01924  0.008982\n',
     '   998/999     1.82G   0.03055   0.02504   0.01201    0.0676       148       416     0.973    0.9663    0.9859    0.8312   0.02488   0.01942   0.00898\n',
     '   999/999     1.82G   0.03112   0.02214   0.01227   0.06553       146       416    0.9731    0.9666    0.9857    0.8301   0.02482   0.01951  0.009014\n']



## Labeling a new set of images

The names of my classes are digits. Under the hood, the YOLOv5 model is working of the index of the class, rather than the human-readable name. Consequently, the identities of each class index must be supplied.

```python
#aw.last_results_path + "/weights/best.pt"
from ModelAssistedLabel.detect import Viewer

class_idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
v = Viewer("pre-trained weights/21-2-25 1k-digits YOLOv5-weights.pt", class_idx)
```

    Fusing layers... 


```python
import random

images = [os.path.join(unlabeled_images, x) for x in os.listdir(unlabeled_images)]
```

```python
%matplotlib inline 
for image in random.sample(images,3):
  v.plot_for(image)
```

    image 1/1 /content/drive/MyDrive/Coding/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 7:50-12:50/27.jpg: >>> [{'predictions': ['0 0.441406 0.385417 0.0140625 0.0708333 0.834958', '2 0.413672 0.379167 0.0195312 0.0777778 0.893516', '7 0.389453 0.376389 0.0210938 0.0777778 0.90789', '9 0.364844 0.372917 0.021875 0.0791667 0.912621']}]



![png](docs/images/output_37_1.png)


    image 1/1 /content/drive/My Drive/Coding/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 7:50-12:50/136.jpg: >>> [{'predictions': ['0 0.419141 0.377778 0.0148437 0.075 0.61542', '0 0.36875 0.370833 0.01875 0.0805556 0.804835', '0 0.397656 0.376389 0.015625 0.075 0.825409', '8 0.436719 0.382639 0.01875 0.0763889 0.894479']}]



![png](docs/images/output_37_3.png)


    image 1/1 /content/drive/My Drive/Coding/ModelAssistedLabel/Image Repo/unlabeled/21-3-22 rowing (200) 7:50-12:50/143.jpg: >>> [{'predictions': ['7 0.437891 0.380556 0.0195312 0.0777778 0.547772', '0 0.397656 0.375694 0.015625 0.0708333 0.758558', '0 0.369141 0.371528 0.0164062 0.0763889 0.805282', '1 0.414453 0.377778 0.0210938 0.0805556 0.907629']}]



![png](docs/images/output_37_5.png)


## Exporting annotated images

```python
project_name = "seven segment digits"
outzip = Defaults._itername(project_name)
os.mkdir(outzip)
print(outzip)
```

    seven segment digits3


Store the class labels with index 0 on line 1, index 1 on line 2, and so on.

```python
with open(os.path.join(outzip, "label_map.txt"), "w") as label_map:
  label_map.writelines("\n".join(class_idx))
```

Ensure that image/label pairs have a common root filename

```python
import random, PIL
salt = lambda: str(random.random())[2:]

for result in results:
  #generate a likely-to-be-unique filename
  shared_root = Defaults._itername(f"{project_name}-{salt()}")

  #save the image to the outfile
  image = PIL.Image.open(result["image path"])
  image.save(os.path.join(outzip, f"{shared_root}.jpg"))

  #save the predictions to the outfile
  predictions = result["predictions"]
  with open(os.path.join(outzip, f"{shared_root}.txt"), "w") as prediction_file:
    prediction_file.writelines("\n".join([x["yolov5 format"] for x in predictions]))
```

## Next Steps

After letting the YOLOv5 model take a stab at labeling, I would then adjust these predictions manually before absorbing them to the training data. While I built (an admittedly janky) labeler to perform my touchups, There are certaintly a number of other anntotation tool available.

I've only used one commerical annotation tool and that would be Roboflow's annotator. Roboflow was a great tool for me to use when I was starting off.
