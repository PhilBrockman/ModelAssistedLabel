# YOLOv5 Model Assisted Labeling
> Work based on https://models.roboflow.com/object-detection/yolov5.


Labeling a new dataset from scratch is tiresome. To label images faster, I train a YOLOv5 model and apply its predictions to new images.

## How to use

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


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-5-7f51d2053f8b> in <module>()
    ----> 1 get_ipython().run_cell_magic('time', '', 'repo = "/content/drive/MyDrive/Coding/Roboflow Export (841)"\nname = "nospaces"\nwm = AutoWeights(repo, name)\nwm.generate_weights(10)')
    

    /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py in run_cell_magic(self, magic_name, line, cell)
       2115             magic_arg_s = self.var_expand(line, stack_depth)
       2116             with self.builtin_trap:
    -> 2117                 result = fn(magic_arg_s, cell)
       2118             return result
       2119 


    <decorator-gen-53> in time(self, line, cell, local_ns)


    /usr/local/lib/python3.7/dist-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        186     # but it's overkill for just that one bit of state.
        187     def magic_deco(arg):
    --> 188         call = lambda f, *a, **k: f(*a, **k)
        189 
        190         if callable(arg):


    /usr/local/lib/python3.7/dist-packages/IPython/core/magics/execution.py in time(self, line, cell, local_ns)
       1191         else:
       1192             st = clock2()
    -> 1193             exec(code, glob, local_ns)
       1194             end = clock2()
       1195             out = None


    <timed exec> in <module>()


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in __init__(self, resource_dir, name, out_dir, MAX_SIZE, custom_data_yaml, verbose, train_path)
        383     self.resource_paths = ["test/", "train/", "valid/", "data.yaml"]
        384     for r in self.resource_paths: #make sure none of these paths already exist
    --> 385       assert os.path.exists(r) is False, f"{r} exists... You may be in the middle of an active project"
        386     self.out_dir = out_dir
        387     self.train_path = train_path


    AssertionError: train/ exists... You may be in the middle of an active project


```python
%run "_Synch.ipynb"
```

    /content/drive/MyDrive/Coding/ModelAssistedLabel
    install nbdev: 
    build resources: UPDATE
    Converted 00_ultralytics.ipynb.
    Converted 01_split.ipynb.
    Converted 02_train.ipynb.
    Converted index.ipynb.
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/01_split.ipynb
    converting: /content/drive/My Drive/Coding/ModelAssistedLabel/00_ultralytics.ipynb
    converting /content/drive/My Drive/Coding/ModelAssistedLabel/index.ipynb to README.md
    git commit -m: collapsing index into functions
    Executing: git config --local include.path ../.gitconfig
    Success: hooks are installed and repo's .gitconfig is now trusted
    [master 9ba08e1] <function <lambda> at 0x7f87b2db4680>
     8 files changed, 174 insertions(+), 261 deletions(-)
    Counting objects: 12, done.
    Delta compression using up to 2 threads.
    Compressing objects: 100% (12/12), done.
    Writing objects: 100% (12/12), 3.47 KiB | 592.00 KiB/s, done.
    Total 12 (delta 10), reused 0 (delta 0)
    remote: Resolving deltas: 100% (10/10), completed with 10 local objects.[K
    To https://github.com/PhilBrockman/ModelAssistedLabel.git
       27453bd..9ba08e1  master -> master

