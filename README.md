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
%run "_Synch.ipynb"
```

    /content/drive/MyDrive/Coding/ModelAssistedLabel
    install nbdev: sure
    build resources: yes
    Converted 00_ultralytics.ipynb.
    Converted 01_split.ipynb.
    Converted 02_train.ipynb.
    Converted index.ipynb.
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/02_train.ipynb
    An error occurred while executing the following cell:
    ------------------
    from nbdev.showdoc import show_doc
    from ModelAssistedLabel.exmaples import *
    ------------------
    
    [0;31m[0m
    [0;31mNameError[0mTraceback (most recent call last)
    [0;32m<ipython-input-1-d22d9f1e17fa>[0m in [0;36m<module>[0;34m()[0m
    [1;32m      1[0m [0;32mfrom[0m [0mnbdev[0m[0;34m.[0m[0mshowdoc[0m [0;32mimport[0m [0mshow_doc[0m[0;34m[0m[0;34m[0m[0m
    [0;32m----> 2[0;31m [0;32mfrom[0m [0mModelAssistedLabel[0m[0;34m.[0m[0mexmaples[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
    [0m
    [0;32m/content/drive/My Drive/Coding/ModelAssistedLabel/ModelAssistedLabel/exmaples.py[0m in [0;36m<module>[0;34m()[0m
    [1;32m      4[0m [0;34m[0m[0m
    [1;32m      5[0m [0;31m# Cell[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
    [0;32m----> 6[0;31m [0;32mdef[0m [0mprepare_split[0m[0;34m([0m[0mresource_dir[0m[0;34m,[0m [0mout_dir[0m [0;34m=[0m [0;34m"."[0m[0;34m,[0m [0mdata_yaml[0m[0;34m=[0m[0mDefaults[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdata_yaml[0m[0;34m,[0m [0mMAX_SIZE[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
    [0m[1;32m      7[0m   [0mg[0m [0;34m=[0m [0mGeneration[0m[0;34m([0m[0mrepo[0m[0;34m=[0m[0mresource_dir[0m[0;34m,[0m [0mout_dir[0m[0;34m=[0m[0mout_dir[0m[0;34m,[0m [0mdata_yaml[0m[0;34m=[0m[0mdata_yaml[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [1;32m      8[0m   [0mg[0m[0;34m.[0m[0mset_split[0m[0;34m([0m[0mMAX_SIZE[0m[0;34m=[0m[0mMAX_SIZE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    
    [0;31mNameError[0m: name 'Defaults' is not defined
    NameError: name 'Defaults' is not defined
    
    Conversion failed on the following:
    index.ipynb
    converting /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb to README.md



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /usr/local/lib/python3.7/dist-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
        728             try:
    --> 729                 ident, reply = self.session.recv(self.stdin_socket, 0)
        730             except Exception:


    /usr/local/lib/python3.7/dist-packages/jupyter_client/session.py in recv(self, socket, mode, content, copy)
        802         try:
    --> 803             msg_list = socket.recv_multipart(mode, copy=copy)
        804         except zmq.ZMQError as e:


    /usr/local/lib/python3.7/dist-packages/zmq/sugar/socket.py in recv_multipart(self, flags, copy, track)
        582         """
    --> 583         parts = [self.recv(flags, copy=copy, track=track)]
        584         # have first part already, only loop while more to receive


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket.Socket.recv()


    zmq/backend/cython/socket.pyx in zmq.backend.cython.socket._recv_copy()


    /usr/local/lib/python3.7/dist-packages/zmq/backend/cython/checkrc.pxd in zmq.backend.cython.checkrc._check_rc()


    KeyboardInterrupt: 

    
    During handling of the above exception, another exception occurred:


    KeyboardInterrupt                         Traceback (most recent call last)

    /content/drive/My Drive/Coding/ModelAssistedLabel/_Synch.ipynb in <module>()
    ----> 1 if len(message("git commit -m: ")) > 2:
          2   get_ipython().system('nbdev_install_git_hooks')
          3   get_ipython().system('git config --global user.email "phil.brockman@gmail.com"')
          4   get_ipython().system('git config --global user.name "Phil Brockman"')
          5   get_ipython().system('git add *')


    /content/drive/My Drive/Coding/ModelAssistedLabel/_Synch.ipynb in <lambda>(x)
    ----> 1 message = lambda x: input(x)
    

    /usr/local/lib/python3.7/dist-packages/ipykernel/kernelbase.py in raw_input(self, prompt)
        702             self._parent_ident,
        703             self._parent_header,
    --> 704             password=False,
        705         )
        706 


    /usr/local/lib/python3.7/dist-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
        732             except KeyboardInterrupt:
        733                 # re-raise KeyboardInterrupt, to truncate traceback
    --> 734                 raise KeyboardInterrupt
        735             else:
        736                 break


    KeyboardInterrupt: 


```python
ls
```

    00_ultralytics.ipynb  docker-compose.yml  MANIFEST.in          _Synch.ipynb
    01_split.ipynb        [0m[01;34mdocs[0m/               [01;34mModelAssistedLabel[0m/  [01;34mtest[0m/
    02_train.ipynb        index.ipynb         README.md            [01;34mtrain[0m/
    CONTRIBUTING.md       LICENSE             settings.ini         [01;34mvalid[0m/
    data.yaml             Makefile            setup.py             [01;34myolov5[0m/

