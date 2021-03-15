# YOLOv5 Model Assisted Labeling
> Work based on https://models.roboflow.com/object-detection/yolov5.


## How to use

### Vanilla image sets

```python
%load_ext autoreload 
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


Recursively search a folder (`repo`) that contains images and labels.

```python
from ModelAssistedLabel.core import Generation, Defaults
g = Generation(repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)", 
              out_dir = "/content/drive/MyDrive/Coding/01_train",
              data_yaml=Defaults().data_yaml)

def tostr(split):
  return [{k: len(v)} for k,v in split.items()]

def sumg(split):
  return sum([list(x.values())[0] for x in tostr(split)])

for MAX_SIZE in [10, None]:
  g.set_split(MAX_SIZE=MAX_SIZE)
  print("summary: ", tostr(g.split))
  print("checksum:", sumg(g.split))
```

    summary:  [{'train': 7}, {'valid': 2}, {'test': 1}]
    checksum: 10
    summary:  [{'train': 589}, {'valid': 169}, {'test': 83}]
    checksum: 841

