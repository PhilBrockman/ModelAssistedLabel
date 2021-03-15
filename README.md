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


Writing these files to disk for when we need them later.

```python
zipped = g.process_split(" vanilla")
```

    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.jpg | ./train/images/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.txt | ./train/labels/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.jpg | ./train/images/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.txt | ./train/labels/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.jpg | ./train/images/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.txt | ./train/labels/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.jpg | ./train/images/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.txt | ./train/labels/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.jpg | ./valid/images/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.txt | ./valid/labels/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.txt


```python
import os

local = os.path.basename(zipped)
!unzip "{local}" #grab data
!rm -f -r '{local}' #remove zip file
```

    Archive:  Roboflow Export (841) vanilla 21-03-15 21-36-20.zip
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/images/
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/images/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.jpg  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/images/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.jpg  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/images/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.jpg  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/images/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.jpg  
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/labels/
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/labels/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.txt  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/labels/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.txt  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/labels/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.txt  
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/train/labels/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.txt  
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/valid/
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/valid/images/
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/valid/images/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.jpg  
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/valid/labels/
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/valid/labels/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.txt  
       creating: Roboflow Export (841) vanilla 21-03-15 21-36-20/test/
      inflating: Roboflow Export (841) vanilla 21-03-15 21-36-20/data.yaml  


```python
for splt in ["test/", "train/", "valid/", "data.yaml"]:
  os.system(f"mv '{local[:-4]}/{splt}' .")
os.system(f"rm -f -r '{local[:-4]}'")
```

```python
!nbdev_build_lib
```

    Converted 00_ultralytics.ipynb.
    Converted 01_split.ipynb.
    Converted 02_train.ipynb.
    Converted index.ipynb.


```python
from ModelAssistedLabel.core import Trainer
```

```python
t= Trainer(name="testfromindex")
```

```python
%%time
t.train(2)
```

    CPU times: user 1.69 ms, sys: 3.35 ms, total: 5.05 ms
    Wall time: 10.5 s


```python
ls yolov5/runs/train
```

    [0m[01;34mbingo[0m/  [01;34mcustom[0m/  [01;34mcustom2[0m/  [01;34mcustom3[0m/

