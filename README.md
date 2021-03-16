# Model-asisted Labeling with YOLOv5



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
from ModelAssistedLabel.core import Defaults

os.chdir(Defaults().root)
Defaults.prepare_YOLOv5()
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

    summary:  [{'train': 4}, {'valid': 1}, {'test': 0}]
    checksum: 5
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-130-jpg_jpg.rf.ec2c59c5f95de888149f881f4f3732c8.jpg | ./train/images/digittake-130-jpg_jpg.rf.ec2c59c5f95de888149f881f4f3732c8.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-130-jpg_jpg.rf.ec2c59c5f95de888149f881f4f3732c8.txt | ./train/labels/digittake-130-jpg_jpg.rf.ec2c59c5f95de888149f881f4f3732c8.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-116-jpg_jpg.rf.879baef3a241e202524ad57bfb8fca4c.jpg | ./train/images/digittake-116-jpg_jpg.rf.879baef3a241e202524ad57bfb8fca4c.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-116-jpg_jpg.rf.879baef3a241e202524ad57bfb8fca4c.txt | ./train/labels/digittake-116-jpg_jpg.rf.879baef3a241e202524ad57bfb8fca4c.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/save_dirrcropped-jpg_jpg.rf.340eb8ea3c457536f9ca160912bf907b.jpg | ./train/images/save_dirrcropped-jpg_jpg.rf.340eb8ea3c457536f9ca160912bf907b.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/save_dirrcropped-jpg_jpg.rf.340eb8ea3c457536f9ca160912bf907b.txt | ./train/labels/save_dirrcropped-jpg_jpg.rf.340eb8ea3c457536f9ca160912bf907b.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-93-jpg_jpg.rf.8f21c0ccb915fc3d6f29f8cb28ce7d51.jpg | ./train/images/digittake-93-jpg_jpg.rf.8f21c0ccb915fc3d6f29f8cb28ce7d51.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-93-jpg_jpg.rf.8f21c0ccb915fc3d6f29f8cb28ce7d51.txt | ./train/labels/digittake-93-jpg_jpg.rf.8f21c0ccb915fc3d6f29f8cb28ce7d51.txt
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/images/digittake-155-jpg_jpg.rf.990bb1b3db89338db4b4b08a8c17b151.jpg | ./valid/images/digittake-155-jpg_jpg.rf.990bb1b3db89338db4b4b08a8c17b151.jpg
    target/dest /content/drive/MyDrive/Coding/Roboflow Export (841)/labels/digittake-155-jpg_jpg.rf.990bb1b3db89338db4b4b08a8c17b151.txt | ./valid/labels/digittake-155-jpg_jpg.rf.990bb1b3db89338db4b4b08a8c17b151.txt



    ---------------------------------------------------------------------------

    IsADirectoryError                         Traceback (most recent call last)

    <ipython-input-43-7f51d2053f8b> in <module>()
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
        425     self.train_path = train_path
        426     #automatically build the resource paths and prepare for traniing
    --> 427     self.__prepare_split__(MAX_SIZE=MAX_SIZE, data_yaml=custom_data_yaml, verbose=verbose)
        428     assert self.g is not None
        429 


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in __prepare_split__(self, MAX_SIZE, data_yaml, verbose)
        471     self.g = Generation(repo=self.resource_dir, out_dir=self.out_dir, data_yaml=data_yaml)
        472     self.g.set_split(MAX_SIZE=MAX_SIZE)
    --> 473     self.__split_and_organize_folders__(verbose=verbose)
        474 
        475   def __split_and_organize_folders__(self, verbose):


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in __split_and_organize_folders__(self, verbose)
        497       print("checksum:", sumg(self.g.split))
        498 
    --> 499     zipped = self.g.write_split_to_disk(self.name) #create a zip file in the ROOT directory
        500     local = os.path.basename(zipped)
        501     os.system(f'unzip "{local}"') #grab data


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in write_split_to_disk(self, descriptor, autoname_output)
        270 
        271     dirs = self.__write_images__() #write images
    --> 272     zipped = self.__zip_dirs__(out_folder, dirs) #zip folders
        273     os.system(f"mv '{zipped}' '{self.out_dir}'") #move the output
        274     return f"{self.out_dir}/{zipped}"


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in __zip_dirs__(self, folder, dirs)
        288     """
        289     FileUtilities.mkdir(folder)
    --> 290     self.__write_data_yaml__(folder)
        291     for subdir in self.split:
        292       os.system(f"mv './{subdir}' '{folder}/'")


    /content/drive/MyDrive/Coding/ModelAssistedLabel/ModelAssistedLabel/core.py in __write_data_yaml__(self, filename)
        345       filename: Location to write the yaml data.
        346     """
    --> 347     f = open(filename,"w+")
        348     f.writelines(self.data_yaml)
        349     f.close()


    IsADirectoryError: [Errno 21] Is a directory: 'Roboflow Export (841)nospaces 21-03-16 21-34-54'


### Augmenting an image set
