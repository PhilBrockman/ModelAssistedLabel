# Model-asisted Labeling with YOLOv5
> bootstrapping image annotation


# Background
Object detection is great! ... if your labeled dataset already exists. I wanted to use machine learning to turn my regular rowing machine into a "smart" rowing machine (specifically: I want to track my workout stats).

Unfortunately, I was unable to find a suitable existing set of labeled LCD digits.

After working through [a Roboflow tutorial]( https://models.roboflow.com/object-detection/yolov5), I started to use Roboflow to annotate and store my images. 

I hated annotating my images by hand. Once my model began making reasonable guesses, I resolved to enlist the model's help in labeling new images. (I ended up building a [key-driven image labeler](https://github.com/PhilBrockman/autobbox) to modify my model's predictions, but that codebase is no longer being maintained.)

I found the files responsible for training and employing models and wrote wrappers around the relevant calls

## Expected Inputs:
* Either:
  - **weight** file OR
  - **labeled images**
      + All of the images and labels must be in a common folder (subfolders allowed).
      + labels must be in [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60).
* And:
  - **unlabeled images**
{% include note.html content='Image/label pairs are based on their base filename. For example `image.jpg/image.txt` would be paired as would `other_image5.jpg/other_image5.txt`.' %}

## Expected Output:

* ***ZIP file*** that contains: 
    - `images/`
      + a copy of every image in **Unlabeled Data**
    - `labels/` (folder
      + result of running object detection on each image
    - a results folder produced by Ultralytic's `train.py` on the **Labeled Data**
    - `classmap.yaml` to preserve the identity of the classes


# Preparing Repository

Start by cloning https://github.com/ultralytics/yolov5.

```python
!nbdev_build_lib
```

    Converted 00_config.ipynb.
    Converted 01_split.ipynb.
    Converted 02_train.ipynb.
    Converted 03_detect.ipynb.
    Converted index.ipynb.


```python
from ModelAssistedLabel.config import Defaults
import os

# enter root directory
os.chdir(Defaults().root)

# clone yolov5 repo and install requirements
# ensure GPU is enabled
Defaults.prepare_YOLOv5()
```

    Setup complete. Using torch 1.8.0+cu101 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)


# Image Sets

Many options

## Vanilla image sets

Recursively search a folder (`repo`) that contains images and labels.

```python
repo = "./Image Repo/labeled/Final Roboflow Export (841)"
name = "nospaces"
```

```python
from ModelAssistedLabel.train import AutoWeights
```

```python
wm = AutoWeights(repo, name, MAX_SIZE=None)
```

```python
wm.name="fromIndex"
```

```python
%%time
wm.generate_weights(2000)
```

```python
wm.last_results_path
```




    './fromIndex-876096'



```python
!ls "test/images"
```

```python
runs = {
    './fromIndex-876096': {
        'description': "841 (?) "
    }
}
```
