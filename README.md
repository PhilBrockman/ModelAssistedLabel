# Model-asisted Labeling with YOLOv5



```python
!git add *
```

```python
!git commit -m "wrong filename"
```

    [master 94635e6] wrong filename
     3 files changed, 371 insertions(+), 1 deletion(-)
     create mode 100644 00_config.ipynb
     create mode 100644 docs/config.html


```python
!git push
```

    Counting objects: 6, done.
    Delta compression using up to 2 threads.
    Compressing objects: 100% (6/6), done.
    Writing objects: 100% (6/6), 4.15 KiB | 850.00 KiB/s, done.
    Total 6 (delta 3), reused 0 (delta 0)
    remote: Resolving deltas: 100% (3/3), completed with 3 local objects.[K
    To https://github.com/PhilBrockman/ModelAssistedLabel.git
       fee5d8a..94635e6  master -> master


## Background

Object Detection is great! ... if your labeled dataset already exists. I wanted to use machine learning to turn my regular rowing machine into a "smart" rowing machine (specifically: I want to track my workout stats).

I was unable to find a suitable existing set of labeled LCD digits.

After working through [a Roboflow tutorial]( https://models.roboflow.com/object-detection/yolov5), I started to use Roboflow to annotate  and store my images. Quickly, I resolved to use the model's outputs and labels for incoming images.

---

### Expected Inputs:
* ***Labels***: Assuming use of the [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60).
* ***Images***: Assuming jpgs

Note about file names: Pairs are based on sharing a base filename. For example `image.jpg`/`image.txt` will be paired and `other_image5.jpg'/`other_image5.txt`.

### Expected Use:

Produce the predicted annotations for a new set of images.

(I ended up building a [key-driven image labeler](https://github.com/PhilBrockman/autobbox) to modify my model's predictions, but that codebase is no longer being maintained. I personally used Roboflow to both store my images and subsequently annotate as I got started wit this project.)

# Preparing Repository

Start by cloning https://github.com/ultralytics/yolov5.

```python
from ModelAssistedLabel.core import Defaults, prepare_YOLOv5
root = Defaults().root
os.chdir(f"{root}")
prepare_YOLOv5()
```

    Setup complete. Using torch 1.8.0+cu101 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', major=7, minor=0, total_memory=16160MB, multi_processor_count=80)


# Image Sets

### Vanilla image sets

Recursively search a folder (`repo`) that contains images and labels.

```python
%%time 
repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)"
name = "nospaces"
wm = AutoWeights(repo, name)
wm.generate_weights(10)
```

### Augmenting an image set
