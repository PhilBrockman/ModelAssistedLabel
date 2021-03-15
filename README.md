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
              out_dir = ".",
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


Writing these files to disk for when we need them later.

```python
zipped = g.process_split(" vanilla")
```

```python
import os

local = os.path.basename(zipped)
!unzip "{local}" #grab data
!rm -f -r '{local}' #remove zip file
mv '{local[:-4]}/data.yaml' .
```

```python
ls
```

     00_ultralytics.ipynb   Makefile
     01_split.ipynb         MANIFEST.in
     02_augment.ipynb       [0m[01;34mModelAssistedLabel[0m/
     CONTRIBUTING.md        README.md
     data.yaml             [01;34m'Roboflow Export (841) vanilla 21-03-15 20-18-11'[0m/
     docker-compose.yml     settings.ini
     [01;34mdocs[0m/                  setup.py
     index.ipynb            _Synch.ipynb
     LICENSE                [01;34myolov5[0m/


```python
from ModelAssistedLabel.core import Trainer
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-69-90ceaab1bc7f> in <module>()
    ----> 1 from ModelAssistedLabel.core import Trainer
    

    ImportError: cannot import name 'Trainer' from 'ModelAssistedLabel.core' (/content/drive/My Drive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py)

    

    ---------------------------------------------------------------------------
    NOTE: If your import is failing due to a missing package, you can
    manually install dependencies using either !pip or !apt.
    
    To view examples of installing some common dependencies, click the
    "Open Examples" button below.
    ---------------------------------------------------------------------------


