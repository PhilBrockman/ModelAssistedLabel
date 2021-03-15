# YOLOv5 Model Assisted Labeling
> Work based on https://models.roboflow.com/object-detection/yolov5.


## How to use

### Vanilla image sets

```python
from ModelAssistedLabel.core import Generation, Defaults
os.chdir(Defaults().root)
```

```python
os.getcwd()
os.getcwd()
1+1
os.getcwd()
```




    '/content/drive/My Drive/Coding/ModelAssistedLabel'



Recursively search a folder (`repo`) that contains images and labels.

```python
g = Generation(repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)",
              out_dir = ".",
              data_yaml=Defaults().data_yaml)

def tostr(split):
  return [{k: len(v)} for k,v in split.items()]

def sumg(split):
  return sum([list(x.values())[0] for x in tostr(split)])

for MAX_SIZE in [10, None, 5]:
  g.set_split(MAX_SIZE=MAX_SIZE)
  print("summary: ", tostr(g.split))
  print("checksum:", sumg(g.split))
```

    summary:  [{'train': 7}, {'valid': 2}, {'test': 1}]
    checksum: 10
    summary:  [{'train': 589}, {'valid': 169}, {'test': 83}]
    checksum: 841
    summary:  [{'train': 4}, {'valid': 1}, {'test': 0}]
    checksum: 5


```python
test_eq(".", g.out_dir) #double check writing to current directory
pwd1 = os.getcwd()
os.chdir(Defaults().root)
```

```python
pwd2 = os.getcwd()
test_eq(pwd1, pwd2) #make sure we're still in the root directory
```

```python
zipped = g.process_split(" vanilla") #create a zip file in the ROOT directory
```

```python
import os

local = os.path.basename(zipped)
!unzip "{local}" #grab data
!rm -f -r '{local}' #remove zip file
```

```python
#move the contents of the zip file into postion within the ROOT directory
for splt in ["test/", "train/", "valid/", "data.yaml"]:
  os.system(f"mv '{local[:-4]}/{splt}' .")

#removed the folder that was taken out of the zip
os.system(f"rm -f -r '{local[:-4]}'")
```




    0



```python
from ModelAssistedLabel.core import Trainer
```

```python
t = Trainer("abc")
```

```python
train_path = "yolov5/runs/train"
ldir = lambda path: set(os.listdir(path))

before = ldir(train_path)
t.train(1)
after = ldir(train_path)

current = list(after - before)[0]
```

```python
weights_path = f'{training_path}/{current}/weights'
weights = os.listdir(weights_path)
test_eq(len(weights), 2)
os.path.join(weights_path, "best.pt")
```




    'yolov5/runs/train/abc8/weights/best.pt'



```python
a
```
