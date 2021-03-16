# YOLOv5 Model Assisted Labeling
> Work based on https://models.roboflow.com/object-detection/yolov5.


Labeling a new dataset from scratch is tiresome. To label images faster, I train a YOLOv5 model and apply its predictions to new images.

## Expected Inputs:
* ***Labels***: Assuming use of the [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60).
* ***Images***: Assuming jpgs

Note about file names: Pairs are based on sharing a base filename. For example `image.jpg`/`image.txt` will be paired and `other_image5.jpg'/`other_image5.txt`.

## Expected Outputs:

A set of labeled predicted annotations for a new set of images.

(I ended up building a [key-driven image labeler](https://github.com/PhilBrockman/autobbox) to modify my model's predictions, but that codebase is no longer being maintained. I personally used Roboflow to both store my images and subsequently annotate as I got started wit this project.

## Preparing Repository

Start by cloning https://github.com/ultralytics/yolov5

```python
ModelAssistedLabel.core.prepare_YOLOv5()
```

# Image Sets

### Vanilla image sets

```python
from ModelAssistedLabel.core import AutoWeights
```

Recursively search a folder (`repo`) that contains images and labels.

```python
%%time 
repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)"
name = "nospaces"
wm = AutoWeights(repo, name)
wm.generate_weights(10)
```

### Augmenting an image set
