# YOLOv5 Model Assisted Labeling
> Work based on https://models.roboflow.com/object-detection/yolov5.


## How to use

### Vanilla image sets

```python
from ModelAssistedLabel.core import Generation, Defaults
```

Recursively search a folder (`repo`) that contains images and labels.

```python
repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)"
name = "nospaces"
tidy_weights(repo, name)
```

    summary:  [{'train': 4}, {'valid': 1}, {'test': 0}]
    checksum: 5
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.jpg | ./train/images/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.txt | ./train/labels/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.jpg | ./train/images/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.txt | ./train/labels/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.jpg | ./train/images/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.txt | ./train/labels/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.jpg | ./train/images/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.txt | ./train/labels/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.jpg | ./valid/images/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.txt | ./valid/labels/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.txt





    '/content/drive/MyDrive/Coding/Roboflow Export (841)/nospaces'



```python
!rm ModelAssistedLabel/setup.py
```

```python

!nbdev_build_lib
```

```python
%run "_Synch.ipynb"
```

    /content/drive/MyDrive/Coding/ModelAssistedLabel
    install nbdev: 
    build resources: you
    Converted 00_ultralytics.ipynb.
    Converted 01_split.ipynb.
    Converted 02_train.ipynb.
    Converted index.ipynb.
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb
    converting /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb to README.md
    git commit -m: successfully wrapped train.py
    Executing: git config --local include.path ../.gitconfig
    Success: hooks are installed and repo's .gitconfig is now trusted
    [master 27453bd] <function <lambda> at 0x7f87b2d83830>
     33 files changed, 878 insertions(+), 703 deletions(-)
     create mode 100644 ModelAssistedLabel/exmaples.py
     rewrite README.md (81%)
     rewrite docs/index.html (72%)
     delete mode 100644 train/images/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.jpg
     create mode 100644 train/images/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.jpg
     delete mode 100644 train/images/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.jpg
     delete mode 100644 train/images/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.jpg
     delete mode 100644 train/images/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.jpg
     create mode 100644 train/images/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.jpg
     create mode 100644 train/images/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.jpg
     create mode 100644 train/images/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.jpg
     rewrite train/labels.cache (89%)
     delete mode 100644 train/labels/digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.txt
     create mode 100644 train/labels/digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.txt
     delete mode 100644 train/labels/digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.txt
     delete mode 100644 train/labels/digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.txt
     delete mode 100644 train/labels/digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.txt
     create mode 100644 train/labels/digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.txt
     create mode 100644 train/labels/save_dirrsave_dirrdd31a247c313689d77bbf8fbf2bd7dbac8d44333-jpg-jpg_jpg.rf.ce0b7e55771cab3ea9a210b96e80541f.txt
     create mode 100644 train/labels/screenytake-35-jpg-cropped-jpg_jpg.rf.8a59d5d74ecd425f9bfdffba63277cc6.txt
     create mode 100644 valid/images/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.jpg
     delete mode 100644 valid/images/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.jpg
     delete mode 100644 valid/images/save_dirrtake-85-jpg-cropped-jpg-jpg_jpg.rf.a46c93393994428d61a0432560984233.jpg
     rewrite valid/labels.cache (88%)
     create mode 100644 valid/labels/digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.txt
     delete mode 100644 valid/labels/digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.txt
     delete mode 100644 valid/labels/save_dirrtake-85-jpg-cropped-jpg-jpg_jpg.rf.a46c93393994428d61a0432560984233.txt
    Counting objects: 21, done.
    Delta compression using up to 2 threads.
    Compressing objects: 100% (21/21), done.
    Writing objects: 100% (21/21), 13.62 KiB | 1.13 MiB/s, done.
    Total 21 (delta 9), reused 0 (delta 0)
    remote: Resolving deltas: 100% (9/9), completed with 8 local objects.[K
    To https://github.com/PhilBrockman/ModelAssistedLabel.git
       c2bac2e..27453bd  master -> master


```python
ls
```

    00_ultralytics.ipynb  docker-compose.yml  MANIFEST.in          _Synch.ipynb
    01_split.ipynb        [0m[01;34mdocs[0m/               [01;34mModelAssistedLabel[0m/  [01;34mtest[0m/
    02_train.ipynb        index.ipynb         README.md            [01;34mtrain[0m/
    CONTRIBUTING.md       LICENSE             settings.ini         [01;34mvalid[0m/
    data.yaml             Makefile            setup.py             [01;34myolov5[0m/

