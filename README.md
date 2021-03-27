# Model-asisted Labeling with YOLOv5
> custom image set annotation with a model's help


![base64 splash](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/modelassistedlabel%20splash.jpg?raw=true)

## Background

My exercise equipment, despite even being electronic, doesn’t connect to a network.

But if I instead point my webcam at the equipment’s LCD output, I can make a machine learn to identify and interpret useful information. Perfect! I’ll just utilize object detection to determine the location and identity of the machine’s analog readout. 

First question, just a tiny one, how do you do that?  

After wading through several guides, I found [Roboflow's YOLOv5 tutorial]( https://models.roboflow.com/object-detection/yolov5). They helped provide a hands-on and accessible experience in machine learning.

Unfortunately, I didn't have much luck with existing models being able to readily parse digits. Instead, I decided to start building my own dataset.

I shouldn't have been caught off-guard by the tedium of manually annotating images. As my mind starts to drift, I wonder if I’m a reCAPTCHA interface that’s gained sentience, and I break through. If I label enough digits, I can train a YOLO model to tell me what it sees. I can then take that information and pre-label my images with those predictions. 

The pieces come together.  I can focus on writing code while I use Roboflow to sort, generate, and deliver my images. I sleuth through [Ultralytic's](https://github.com/ultralytics/yolov5) original project and build wrappers around the essential functions in `detect.py` and `train.py`.

This repository contains the tools that let me "pre-label" my images before sending them off for human inspection and correction.

I use the `Viewer` class to 
{% include note.html content='In `./Image Repo` I provide access to 841 labeled images (lumped in one folder) and 600 unlabeled images (seperated into three sets of 200 images - lighting condition is the same within each run, but differs between runs). ' %}




## Getting Started

{% include tip.html content='[Open In Colab](https://colab.research.google.com/github/PhilBrockman/ModelAssistedLabel/blob/master/index.ipynb)' %}

```
# clone this repository
!git clone -b future_forward https://github.com/PhilBrockman/ModelAssistedLabel.git
%cd "ModelAssistedLabel"
```

    Cloning into 'ModelAssistedLabel'...
    remote: Enumerating objects: 424, done.[K
    remote: Counting objects: 100% (424/424), done.[K
    remote: Compressing objects: 100% (120/120), done.[K
    remote: Total 4401 (delta 305), reused 414 (delta 299), pack-reused 3977[K
    Receiving objects: 100% (4401/4401), 207.35 MiB | 14.02 MiB/s, done.
    Resolving deltas: 100% (1328/1328), done.
    Checking out files: 100% (2375/2375), done.
    /content/drive/My Drive/Coding/vision.philbrockman.com/ModelAssistedLabel


### Expected Inputs:

-  **labeled images**
    + All of the images and labels must be in a common folder (subfolders allowed).
    + Labels must be in [YOLOv5 format](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885).
{% include note.html content='Image/label pairs are based on their base filename. For example `image.jpg/image.txt` would be paired as would `other_image5.jpg/other_image5.txt`.' %}





```
# these images have already been labeled
labeled_images   = "Image Repo/labeled/Final Roboflow Export (841)"
```

  - **unlabeled images**

```
# these images need to be labeled
unlabeled_images = "Image Repo/unlabeled/21-3-22 rowing (200) 1:53-7:00"
```

### Expected Output:

* **Folder** that contains: 
    - `images/`
      + a copy of every image in **Unlabeled Data**
    - `labels/`
      + result of running object detection on each image
    - a results folder produced by Ultralytic's `train.py` on the **Labeled Data** (if not using pre-trained weights)
    - `class labels.txt` to preserve the identity of the classes


```
from ModelAssistedLabel.config import Defaults
import os

project_name = "seven segment digits - "
export_folder = Defaults._itername(project_name)

print(export_folder)
```

    seven segment digits - 1


```
# make the export folder
os.mkdir(export_folder)

# make the images and labels subfolders
for resource_folder in ["images", "labels"]:
  os.mkdir(os.path.join(export_folder, resource_folder))
```

## Configure defaults

Several values are stored by the `Defaults` class. Any value can be overridden (and new values can be added. Make sure to `save()` any changes!

```
d = Defaults()
print(" -- Defined Keys: --")
print("\n".join([x for x in d.__dict__.keys()]))
```

     -- Defined Keys: --
    config_file
    root
    split_ratio
    data_yaml
    resource_map
    trainer_template


Speciy the absolute path of the root directory.

```
!pwd
```

    /content/drive/My Drive/Coding/vision.philbrockman.com/ModelAssistedLabel


```
d.root = "/content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/"
```

Save any changes and enter root directory

```
d.save()
d.to_root()
```

    moving to /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/


I borrow the instructions to set up the Ultralytics repo from [the Roboflow tutorial]( https://models.roboflow.com/object-detection/yolov5). (If I'd be allowed one undo on this project, I wish I would have intially forked this project from that tutorial.)

```
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo

%cd yolov5
# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets
%cd ..
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-44-e9b01eb05e5a> in <module>()
    ----> 1 als# clone YOLOv5 repository
          2 get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
          3 
          4 get_ipython().magic('cd yolov5')
          5 # install dependencies as necessary


    NameError: name 'als' is not defined


Make sure GPU is enabled.

```
if torch.cuda.is_available():
  print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) ))
  d.to_root() #step up a level
else:
   raise Exception("enable GPU")
```

    Setup complete. Using torch 1.8.0+cu101 _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)
    moving to /content/drive/MyDrive/vision.philbrockman.com/ModelAssistedLabel/


    [autoreload of utils failed: Traceback (most recent call last):
      File "/usr/local/lib/python3.7/dist-packages/IPython/extensions/autoreload.py", line 247, in check
        superreload(m, reload, self.old_objects)
    ModuleNotFoundError: spec not found for the module 'utils'
    ]


The names of my classes are digits. Under the hood, the YOLOv5 model is working of the index of the class, rather than the human-readable name. Consequently, the identities of each class index must be supplied.

```
class_idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
```

## Processing input

```
ls "{labeled_images}/label"
```

    digittake-100-jpg_jpg.rf.53b56204bb1420470a789f1ce414d896.jpg
    digittake-100-jpg_jpg.rf.ebcd3c7eead884c717d2597547fbce6a.jpg
    digittake-101-jpg_jpg.rf.179e7197074cf09e28cf1636e15a6d85.jpg
    digittake-101-jpg_jpg.rf.c8218e8e7e93277819beae37596631c1.jpg
    digittake-102-jpg_jpg.rf.0f584f62f0610b2edf24852bf27a8af2.jpg
    digittake-102-jpg_jpg.rf.47683283f95916aca1cdebd4451cd12d.jpg
    digittake-103-jpg_jpg.rf.03983a3ff4cf9bc3da9e40367e1b1ec7.jpg
    digittake-103-jpg_jpg.rf.b5fb3692b01dfb12385674f5182befe9.jpg
    digittake-104-jpg_jpg.rf.5d9b139ea3d110ca0897c50cd8da8936.jpg
    digittake-104-jpg_jpg.rf.8593b2a1806be33bc921fbc5a23dd37d.jpg
    digittake-105-jpg_jpg.rf.2c17c0dffd27396048fc4f9a9934eaef.jpg
    digittake-105-jpg_jpg.rf.cac4ee0d1a8cf4b781ef1aade6171944.jpg
    digittake-106-jpg_jpg.rf.780b7e954c1a7786ba65757732ba9bf6.jpg
    digittake-106-jpg_jpg.rf.c3aed7904d1d86d87851a4380d862e91.jpg
    digittake-107-jpg_jpg.rf.8bae964e97069305519d1d790c6c1926.jpg
    digittake-107-jpg_jpg.rf.d6be4ac4fdc759917b6044db145c8810.jpg
    digittake-108-jpg_jpg.rf.93eed691a59a5753b7ed82e14b4753e9.jpg
    digittake-108-jpg_jpg.rf.cc957914017f978dea63305c9cc1a35d.jpg
    digittake-109-jpg_jpg.rf.077b1bb5d3d49b064ba037f203f19937.jpg
    digittake-109-jpg_jpg.rf.4767b0da71db0f78fd243c3d922c26fd.jpg
    digittake-110-jpg_jpg.rf.400f95eb52d28840c47258c2d73b3cee.jpg
    digittake-110-jpg_jpg.rf.835145b2ce4b85cc750da3ff93060072.jpg
    digittake-111-jpg_jpg.rf.272b78f83b61cb1b10ac350ba038ccf1.jpg
    digittake-112-jpg_jpg.rf.b9c35fc2c09532529c24d1a14a57333e.jpg
    digittake-113-jpg_jpg.rf.23d093be40cc2c1ad6b2289c9b93cb5c.jpg
    digittake-113-jpg_jpg.rf.c1ced02c7675f7143186ed3c7bf19ebb.jpg
    digittake-114-jpg_jpg.rf.06474298e3b8752cd8e773c56bb0aeec.jpg
    digittake-114-jpg_jpg.rf.0c4ece020756644ec3dee4d7b0ec7570.jpg
    digittake-115-jpg_jpg.rf.7532776bfb6b6ae9f0bbdb4526e1107c.jpg
    digittake-116-jpg_jpg.rf.879baef3a241e202524ad57bfb8fca4c.jpg
    digittake-116-jpg_jpg.rf.9b937d4ba2d4f4a5464ac996334af9e8.jpg
    digittake-117-jpg_jpg.rf.7f02ac7a3a4ef67709bc98cf91607f11.jpg
    digittake-117-jpg_jpg.rf.c3cb2b22a41b92335e71cbec20fcddd9.jpg
    digittake-118-jpg_jpg.rf.4e5d3203cdc1304a145e929a26887946.jpg
    digittake-118-jpg_jpg.rf.7d33fad9ebd5d877748e25b138566cd2.jpg
    digittake-119-jpg_jpg.rf.28353cdf72c6cc7998d96c7820a7cb1c.jpg
    digittake-119-jpg_jpg.rf.9c71dade54d8c3e932e8ebe7cafde9fa.jpg
    digittake-11-jpg_jpg.rf.35b398c6cd3bbcf368b2ac33fecf4ba1.jpg
    digittake-120-jpg_jpg.rf.189a46a59d264e87dd70d256d8234d1a.jpg
    digittake-120-jpg_jpg.rf.70b23d389c86243a3d11e777f4134c40.jpg
    digittake-121-jpg_jpg.rf.a29991f0a9539fa96d90c6bbe1c18993.jpg
    digittake-121-jpg_jpg.rf.d5a86c7d9e0b7de175af7c6260b7727c.jpg
    digittake-122-jpg_jpg.rf.597ef7068d1aa65803765c3bd18b0601.jpg
    digittake-122-jpg_jpg.rf.6dbc43596d1194b234ea70f3a370c907.jpg
    digittake-123-jpg_jpg.rf.3af2d45a6a9fdb7f4ee21be8a1e837d0.jpg
    digittake-123-jpg_jpg.rf.981268c249c0797261833afc7c415885.jpg
    digittake-124-jpg_jpg.rf.0c5e1f04b42f1e6fea5bda6ca0e4404b.jpg
    digittake-124-jpg_jpg.rf.7f7c512bf6e8896399fe8d6be661f959.jpg
    digittake-125-jpg_jpg.rf.20ef627e4fd520dec706fec1826ad494.jpg
    digittake-125-jpg_jpg.rf.664c9f79e5ef7be1d9384032f54ba5eb.jpg
    digittake-126-jpg_jpg.rf.3554a38cb0bf7b11c2e15ea8011fa852.jpg
    digittake-126-jpg_jpg.rf.976459f968204b617a2d79b600469523.jpg
    digittake-127-jpg_jpg.rf.b95c5211dc0ad8302dfc6ba83dc449ea.jpg
    digittake-127-jpg_jpg.rf.bc1d0948049ba3b9bc1ce23d0a4ab1c5.jpg
    digittake-128-jpg_jpg.rf.9bacca3053954ea0531b82ac23a13069.jpg
    digittake-128-jpg_jpg.rf.faa43cce13807ff5a0dfda2348116745.jpg
    digittake-129-jpg_jpg.rf.4b8fc3ced2bc71d423bdd517d87f2b54.jpg
    digittake-129-jpg_jpg.rf.e132a452a923004e18041ec4c8853cce.jpg
    digittake-12-jpg_jpg.rf.5a09f90d8bd0bb4a2454f400edadc915.jpg
    digittake-130-jpg_jpg.rf.6af4ec2c116901713b9258870a4773aa.jpg
    digittake-130-jpg_jpg.rf.ec2c59c5f95de888149f881f4f3732c8.jpg
    digittake-131-jpg_jpg.rf.3dab3ea2d017aa47fa6a93f53376ce3b.jpg
    digittake-131-jpg_jpg.rf.d76c163d166277926f2ab8858d693775.jpg
    digittake-132-jpg_jpg.rf.7781b6d302305fe9f5074de05e8b204f.jpg
    digittake-132-jpg_jpg.rf.826cae37fdd0e7691ad592fcb9414151.jpg
    digittake-133-jpg_jpg.rf.57269338004e02a6effd7e70f73c9e3f.jpg
    digittake-133-jpg_jpg.rf.610249ba6ae5a8ea2fae8ade55181e79.jpg
    digittake-134-jpg_jpg.rf.1a22ae8db020fee8c0afeaf4ff1bc05d.jpg
    digittake-134-jpg_jpg.rf.f0b98925a1c6277d9418bad16fc0b5f0.jpg
    digittake-135-jpg_jpg.rf.c0efb8cf7d10b985832629452b44cab2.jpg
    digittake-135-jpg_jpg.rf.f2cc15c201a86a327f8cf7b86b985959.jpg
    digittake-136-jpg_jpg.rf.99336f3053c3237faed2a80e8f686966.jpg
    digittake-136-jpg_jpg.rf.be25feff4b722ee07f904929da83120b.jpg
    digittake-137-jpg_jpg.rf.29999f0b5bce05df98aabfb5c18c8278.jpg
    digittake-137-jpg_jpg.rf.dae7b1976030748e09a923df29c6f73d.jpg
    digittake-138-jpg_jpg.rf.1a2ecd5f04f4b5d03edde77be2077fc1.jpg
    digittake-138-jpg_jpg.rf.e098f4c9a68fc51bdf2573dd1934fd2b.jpg
    digittake-139-jpg_jpg.rf.04cdb0b95cf6c5acf5bc0cb1b6d28a60.jpg
    digittake-139-jpg_jpg.rf.12738711bd5f6c98bf7c5eca91591e67.jpg
    digittake-13-jpg_jpg.rf.ff0f3b19b8e0adc2286919918f8f0606.jpg
    digittake-140-jpg_jpg.rf.ab0767663fa24d67c79e5dc38709543b.jpg
    digittake-140-jpg_jpg.rf.ba25a57667c674e919796199e2b28ade.jpg
    digittake-141-jpg_jpg.rf.2630e207244e91bfdab2e1f0c7dd1ed5.jpg
    digittake-141-jpg_jpg.rf.31f0641996c5b91af51467aa22e425f9.jpg
    digittake-142-jpg_jpg.rf.500085bb11edf2d0a0ea097f5ab624f0.jpg
    digittake-142-jpg_jpg.rf.7e735ff6af25c48c3db55ba456aaf3a3.jpg
    digittake-143-jpg_jpg.rf.b9289ee63ea678e66b7c5e3279296952.jpg
    digittake-143-jpg_jpg.rf.fa9b3deef744bd3ad77b18659bd4cf0a.jpg
    digittake-144-jpg_jpg.rf.0c79c3c6cab80da901f3573c580f9b0e.jpg
    digittake-144-jpg_jpg.rf.d7ec429b12dfb6a7441f90f7a6528068.jpg
    digittake-145-jpg_jpg.rf.6fd7986374f44f80ca2788ad09421ffe.jpg
    digittake-145-jpg_jpg.rf.cb9d205eea8f117623de34b8420b576f.jpg
    digittake-146-jpg_jpg.rf.57089a97b4712403f4c05b5b06475abd.jpg
    digittake-146-jpg_jpg.rf.cc22215f185178b43bd6a47640c24004.jpg
    digittake-147-jpg_jpg.rf.89fac78673dc20147595d70f03baf6e3.jpg
    digittake-147-jpg_jpg.rf.fc51dfa551aa21cf49dfafa63ba55c58.jpg
    digittake-148-jpg_jpg.rf.b732fffcac6a39b81b6adf93195a3d0e.jpg
    digittake-148-jpg_jpg.rf.faa04a36c255ca752c0f0fffd8315feb.jpg
    digittake-149-jpg_jpg.rf.f593b82c3dbff0f31a6e8571cccc0dc9.jpg
    digittake-14-jpg_jpg.rf.1c658211c320c9d60d0ac98a62370f5b.jpg
    digittake-14-jpg_jpg.rf.d4020369553beb1585eea80e9eed7dad.jpg
    digittake-150-jpg_jpg.rf.abaf9d50852e6d9ab016909ca4f79c50.jpg
    digittake-151-jpg_jpg.rf.f6e065d32fe53673953536f16900f4d5.jpg
    digittake-152-jpg_jpg.rf.65961115dc149628b0d9c972fc455af2.jpg
    digittake-153-jpg_jpg.rf.c540375397ac13ca361faf4acc756a30.jpg
    digittake-154-jpg_jpg.rf.9b338b39646e7f28ed770a455bee8b2e.jpg
    digittake-155-jpg_jpg.rf.990bb1b3db89338db4b4b08a8c17b151.jpg
    digittake-156-jpg_jpg.rf.fdb2435f091503787b30ec3054cde0ff.jpg
    digittake-157-jpg_jpg.rf.91b17884600e35869a7f1c07d2135889.jpg
    digittake-158-jpg_jpg.rf.f268a37b55784de3b2575115f1132a8c.jpg
    digittake-159-jpg_jpg.rf.7e85a130bb6d1456ec5a88ba1de406ec.jpg
    digittake-15-jpg_jpg.rf.9d80c398824ba69a65f6a5a870f08468.jpg
    digittake-160-jpg_jpg.rf.81b8ade846a725ae139fa9736951063f.jpg
    digittake-16-jpg_jpg.rf.a1f11d444f83baa8b116dbaabcc860aa.jpg
    digittake-16-jpg_jpg.rf.c03113de13a823f6dd654e3a80e88649.jpg
    digittake-17-jpg_jpg.rf.37d3c83bcb467cde8f3d575c16a7e364.jpg
    digittake-17-jpg_jpg.rf.9e798f37735e92308705f4f4bb1a1f68.jpg
    digittake-18-jpg_jpg.rf.6ce38047020d47ae74aa7487a3a8d7db.jpg
    digittake-18-jpg_jpg.rf.b4bb80a194a6dcc40c9d6f0270f3bd86.jpg
    digittake-19-jpg_jpg.rf.337c031612641985884e1ad0ec592d2b.jpg
    digittake-19-jpg_jpg.rf.bb98590febbb62ac3f78e22dafb71bdb.jpg
    digittake-20-jpg_jpg.rf.20017ea1673254b950024c897c7a7251.jpg
    digittake-20-jpg_jpg.rf.60c822edf790183bde1722b00b9516aa.jpg
    digittake-211-jpg_jpg.rf.351261f9149fbc709eab4c6c7d9d7278.jpg
    digittake-212-jpg_jpg.rf.c2507de987417de37a2302b08d69bb70.jpg
    digittake-213-jpg_jpg.rf.a6dc9af14e69f113a3d4f6e32e3d7075.jpg
    digittake-214-jpg_jpg.rf.415f5356f40b74a141a87a0f495e3b27.jpg
    digittake-215-jpg_jpg.rf.c7118b3729c93ecae4fd6cbf2be9905c.jpg
    digittake-216-jpg_jpg.rf.61eb08ab1864fa3c60f72e0b50b9b07f.jpg
    digittake-217-jpg_jpg.rf.54d793a69e4ba676e78e6c7c56b1e39e.jpg
    digittake-218-jpg_jpg.rf.4964de301981d4e32447c195674a4dc8.jpg
    digittake-219-jpg_jpg.rf.d3caf8df1f37dc4712fa1f9aa925ee7f.jpg
    digittake-21-jpg_jpg.rf.04879fe5c34751153f43f60941a7f578.jpg
    digittake-21-jpg_jpg.rf.21e0619530f14b580f0fede08499e410.jpg
    digittake-220-jpg_jpg.rf.1c3934c829a658e34bac05e629917e6c.jpg
    digittake-221-jpg_jpg.rf.fee9d60fcb78b1858feebc77731d5abb.jpg
    digittake-222-jpg_jpg.rf.ce1d059ced59d6a36c531ab6a1556130.jpg
    digittake-223-jpg_jpg.rf.c2aefbc5c4a89db411a75de946b0286b.jpg
    digittake-224-jpg_jpg.rf.7543b732e4698f913ab0ada80bc2b722.jpg
    digittake-225-jpg_jpg.rf.a1c68bb7bf8a492a953684a9ad8d2fd6.jpg
    digittake-226-jpg_jpg.rf.01621ceb884364344e9d4d169d82b21f.jpg
    digittake-227-jpg_jpg.rf.720b3334d89a9377ad5b128529ac296e.jpg
    digittake-228-jpg_jpg.rf.8e675a68295352b621f954a1c77848f4.jpg
    digittake-229-jpg_jpg.rf.da6bc9eba593d79774d280cbf223e1cd.jpg
    digittake-22-jpg_jpg.rf.8245d676928ef00e272f6cd437326729.jpg
    digittake-22-jpg_jpg.rf.e001f3d2fb099bfdf5099dfee055f3ac.jpg
    digittake-230-jpg_jpg.rf.5c07401f6e4b08a1cff22ec2afdb42b5.jpg
    digittake-231-jpg_jpg.rf.7e91f5ea10b9f7b5f0a1e979c7e2229b.jpg
    digittake-232-jpg_jpg.rf.2c16c3144295b21b24e6a68c18e6db36.jpg
    digittake-233-jpg_jpg.rf.fef6841303ae703c5eb6bb27fd9249ca.jpg
    digittake-234-jpg_jpg.rf.8e371f2e202172326beb9c498ddc6918.jpg
    digittake-235-jpg_jpg.rf.529c8897d86e2472cb1989909aba4444.jpg
    digittake-236-jpg_jpg.rf.766ee8b59de8c928b3bac20f6e53a3d3.jpg
    digittake-237-jpg_jpg.rf.3401f85973317781c1da1301795b8849.jpg
    digittake-238-jpg_jpg.rf.896f777aac57319bca70c2a4140a7c4c.jpg
    digittake-239-jpg_jpg.rf.330a77b5b6eac54b87851b1ae996ff62.jpg
    digittake-23-jpg_jpg.rf.00548b600c00b9a0159307cccf3347f1.jpg
    digittake-23-jpg_jpg.rf.89df62b1a229463cdecb2b6e587adf86.jpg
    digittake-240-jpg_jpg.rf.379b41f040da918d9880e4457fbb00ed.jpg
    digittake-241-jpg_jpg.rf.461389866cb7d4f2aff4138993381c5b.jpg
    digittake-245-jpg_jpg.rf.3565294c657195290d423a79d2a3f07e.jpg
    digittake-246-jpg_jpg.rf.d6716017298d34e076f45667f4b1d43c.jpg
    digittake-247-jpg_jpg.rf.92a1543483920d5fba87c09f4c0c7db8.jpg
    digittake-248-jpg_jpg.rf.f17e6dd07a624205e3f877493e97aad5.jpg
    digittake-249-jpg_jpg.rf.3661e47eab4339985a7fae489bdd71fb.jpg
    digittake-24-jpg_jpg.rf.1724b7b95cea1f0184b0765f3c874a9e.jpg
    digittake-24-jpg_jpg.rf.5467363eaf5106fd1c63fe36a1b3b766.jpg
    digittake-250-jpg_jpg.rf.2233a40bc93696068414cc9541c7e1a7.jpg
    digittake-251-jpg_jpg.rf.8f97d8eb7943eeddb83ad1c2d19d6688.jpg
    digittake-252-jpg_jpg.rf.76a88aaf59fc58c940fdc49800f0b1d0.jpg
    digittake-253-jpg_jpg.rf.d3f3d8216e547c1192a63e567021731f.jpg
    digittake-254-jpg_jpg.rf.c1abc99ccdf7a647da293ce1598479f5.jpg
    digittake-255-jpg_jpg.rf.c0343c080288d79d6eecff1f323d669c.jpg
    digittake-256-jpg_jpg.rf.64fb15c1676b746bf224197c2abdcf31.jpg
    digittake-257-jpg_jpg.rf.afc1896d0dbd7a93986f127c993bd3c5.jpg
    digittake-258-jpg_jpg.rf.a04811832ba366c43fe76330f9922b71.jpg
    digittake-259-jpg_jpg.rf.a78b891656d18e51f08d6f78b328dba0.jpg
    digittake-25-jpg_jpg.rf.181325424e6e32753eac9d507e67e990.jpg
    digittake-25-jpg_jpg.rf.c0a76449fd9a4ae4ae9676bb2ebefe5b.jpg
    digittake-260-jpg_jpg.rf.53bcf1bbf4002e3fe8327c6e7d6b9bdb.jpg
    digittake-261-jpg_jpg.rf.6e0125e129a7e593aab3c8b4ae833f32.jpg
    digittake-262-jpg_jpg.rf.d0d2d3ca9073bf1ae808247d80f0ba56.jpg
    digittake-263-jpg_jpg.rf.04a6a60feed8d630c2b4eef1a6a7371a.jpg
    digittake-264-jpg_jpg.rf.83066f4224651f58f679a16e0623d96f.jpg
    digittake-265-jpg_jpg.rf.ec4cd448b4052f75d12a0a56a3e667e5.jpg
    digittake-266-jpg_jpg.rf.47d5fc1cdb6294c9d6d7572c24d8d4e4.jpg
    digittake-267-jpg_jpg.rf.1ba5a671e734a60bf53c445e55d905f8.jpg
    digittake-268-jpg_jpg.rf.6292b350d0f6d7d711cfda71ea9750d2.jpg
    digittake-269-jpg_jpg.rf.7641bae59736bf370269f25587e1620a.jpg
    digittake-26-jpg_jpg.rf.5b7cb9c81986b882f6e4a0581ad8b96c.jpg
    digittake-270-jpg_jpg.rf.b4c3cc532c12d941d5916efd2e31d18a.jpg
    digittake-271-jpg_jpg.rf.0b64f7e3464ff668fc711ae32e79d4f8.jpg
    digittake-272-jpg_jpg.rf.9d2ed10d9a5591789b73d54bfd8ac1ee.jpg
    digittake-273-jpg_jpg.rf.5ac546f1d8ab043dcf053861d777729a.jpg
    digittake-274-jpg_jpg.rf.4c16f25b08c4d3b51a4853840c30dc42.jpg
    digittake-275-jpg_jpg.rf.64e67d192ad4fa8c3a9789f86bbce37e.jpg
    digittake-276-jpg_jpg.rf.e55b9d99184aebcbfef7a56f60b85093.jpg
    digittake-277-jpg_jpg.rf.83e0cb7b3c5de5f72a0a86e84f8c9b85.jpg
    digittake-278-jpg_jpg.rf.ebd80ed68baa99ce14f72b344b4b5b7b.jpg
    digittake-279-jpg_jpg.rf.1ea057c3dbb75045495976326b1a4046.jpg
    digittake-27-jpg_jpg.rf.161924638cbdf820edec93f9028efc39.jpg
    digittake-280-jpg_jpg.rf.5f6dadfa266833a3f694f37d1a51fa36.jpg
    digittake-281-jpg_jpg.rf.c587a21718dad72b3090daa6268e817a.jpg
    digittake-282-jpg_jpg.rf.8cc4946e829825cc96a965903707447d.jpg
    digittake-283-jpg_jpg.rf.36057cdc04923ee8ac611145c4500e0d.jpg
    digittake-284-jpg_jpg.rf.0c5b4ff9b5ff175d400cbe12ce42dbb7.jpg
    digittake-285-jpg_jpg.rf.3a6b8b5c97a2b3b6033054f9391faa85.jpg
    digittake-286-jpg_jpg.rf.a74a88f1036a6c89a1efde48ead7cae1.jpg
    digittake-287-jpg_jpg.rf.14c792dcc4e41ca5a3afddec2119e0bb.jpg
    digittake-288-jpg_jpg.rf.9a9849dcbd4891eed242bd095d8cb73b.jpg
    digittake-289-jpg_jpg.rf.37d2b2671c0975684b6501fd5db5b5d0.jpg
    digittake-28-jpg_jpg.rf.5da3ae1e628e1714dda582417721f3c6.jpg
    digittake-290-jpg_jpg.rf.8a17740d2d536b8048b14cddfa9ba9f0.jpg
    digittake-291-jpg_jpg.rf.5609649d7b367c112ce5fd4d0b345271.jpg
    digittake-292-jpg_jpg.rf.f28e23ee5ccab1711cd15a43ad82161a.jpg
    digittake-293-jpg_jpg.rf.2ff508557993bd1f32c8564b2ec0645b.jpg
    digittake-294-jpg_jpg.rf.4981a33f478aa6b6482d49a61e8c32b6.jpg
    digittake-295-jpg_jpg.rf.8f47c1dd3d132339af27acd4d1297f85.jpg
    digittake-296-jpg_jpg.rf.3bc64469fc73da94de90fac379ca510a.jpg
    digittake-297-jpg_jpg.rf.a2e20b91dc9e3f1493b132b62476d5ef.jpg
    digittake-298-jpg_jpg.rf.aa1765ea20334421e034081325f11506.jpg
    digittake-299-jpg_jpg.rf.cc5d0a43f38e94ed15ee42993f553c91.jpg
    digittake-29-jpg_jpg.rf.8e1a83c4305cd8746ab80cb2eff102e7.jpg
    digittake-300-jpg_jpg.rf.15973adbcb25c25d0d5f68e1ce397cb6.jpg
    digittake-301-jpg_jpg.rf.c69384cc70ad5ff4166b1f62ed495dd4.jpg
    digittake-302-jpg_jpg.rf.c84c08ce985e4689b20fdcb0b591e456.jpg
    digittake-303-jpg_jpg.rf.dd703aff099c06a43db8d7fa2fbfd45e.jpg
    digittake-304-jpg_jpg.rf.654d43a4673204ad0307114fab558559.jpg
    digittake-305-jpg_jpg.rf.b652e20e9789920d289874fb52383d24.jpg
    digittake-306-jpg_jpg.rf.efdcd12f1480798feaf9d9f8314579b1.jpg
    digittake-307-jpg_jpg.rf.0d3a47f784fae768a0fbbf8a4b1747b5.jpg
    digittake-308-jpg_jpg.rf.134356b8034ccaf574e2d0deaec78568.jpg
    digittake-309-jpg_jpg.rf.73a2099cc137092a48103ba219419f25.jpg
    digittake-30-jpg_jpg.rf.afc6d90ed352bd94bcc259dd59d2fe34.jpg
    digittake-310-jpg_jpg.rf.53b409c14fb445453c4d25c45ed93b8e.jpg
    digittake-311-jpg_jpg.rf.8cc61e70328d75059bc732a65485306f.jpg
    digittake-312-jpg_jpg.rf.8d25e440db4eeedd9eb17f16dcaa3581.jpg
    digittake-313-jpg_jpg.rf.04baefcc95ed30631a7fc9675c534788.jpg
    digittake-314-jpg_jpg.rf.bf9506778739ee2828d12908d60a2345.jpg
    digittake-315-jpg_jpg.rf.464f450bb358a20451b8257c56b8656f.jpg
    digittake-316-jpg_jpg.rf.7a9d8bf042dbe719eaaa2cb8dff68248.jpg
    digittake-317-jpg_jpg.rf.380e7422c18ad666894bafd30c4cee2f.jpg
    digittake-318-jpg_jpg.rf.fc6e5fb16067f8654cc74ded92314e86.jpg
    digittake-319-jpg_jpg.rf.6b8817c98f999acb9486c2f51c207f56.jpg
    digittake-31-jpg_jpg.rf.2d0281fce45dcbf93b51bedecaa2ce1b.jpg
    digittake-320-jpg_jpg.rf.bb502b7d19e557c335a597234212a354.jpg
    digittake-321-jpg_jpg.rf.cf47fa3cd911915b4be270fd6adf9ae5.jpg
    digittake-322-jpg_jpg.rf.5f2250f9e2c2b5b478316d1768f9ce6e.jpg
    digittake-323-jpg_jpg.rf.e7c67a25ed108f594a719603ea28cc70.jpg
    digittake-324-jpg_jpg.rf.9305198f4d7c957afa0e1c0924ec2366.jpg
    digittake-325-jpg_jpg.rf.055ae9e479d0c2060198d0e8d182ae3d.jpg
    digittake-326-jpg_jpg.rf.504a431686382272df3b5c1106b9a279.jpg
    digittake-327-jpg_jpg.rf.51a6a64a1c8528eab30e68ab241cbab0.jpg
    digittake-328-jpg_jpg.rf.9dabc92e1067b5b1db6c9c49271e45c5.jpg
    digittake-329-jpg_jpg.rf.387fb4c11abf98b0c46618ec44a04ede.jpg
    digittake-32-jpg_jpg.rf.288a105b2a8f24157752afc8e81ef150.jpg
    digittake-330-jpg_jpg.rf.633a5d667564c2855ddac6692e456cf1.jpg
    digittake-331-jpg_jpg.rf.2af6a0e0777fc783e979a67725786495.jpg
    digittake-332-jpg_jpg.rf.444387434fa21de0cd0660f1fe0df4e3.jpg
    digittake-333-jpg_jpg.rf.0a69b2e0f92bebd1716540755b72e440.jpg
    digittake-334-jpg_jpg.rf.21aeb8f749c302be0a1caa402a8fc274.jpg
    digittake-335-jpg_jpg.rf.3ec953d134bb308939554a0a29516af8.jpg
    digittake-336-jpg_jpg.rf.870127f51af7e1e539b60dcb92055bde.jpg
    digittake-337-jpg_jpg.rf.50b185e02b8bde5a90f15bf67a457ea5.jpg
    digittake-338-jpg_jpg.rf.402bd8cf52c25711b8f2e7c5974be7ec.jpg
    digittake-339-jpg_jpg.rf.c49451f8c706032df361fe7c046b15e7.jpg
    digittake-33-jpg_jpg.rf.f1d00d9a9f1e64e8da2e528824bbb27f.jpg
    digittake-340-jpg_jpg.rf.b4bfb7a0db492435b840d4fc48492016.jpg
    digittake-341-jpg_jpg.rf.c1b76fda25b85e97d0cf1ddbc007ee15.jpg
    digittake-342-jpg_jpg.rf.4789fd6988b2e760837a195a44028c54.jpg
    digittake-343-jpg_jpg.rf.47e20e28ed6128d3080c7bc3057fa698.jpg
    digittake-344-jpg_jpg.rf.cbe8fd13bf132dbb26a2ba0611d29c8f.jpg
    digittake-345-jpg_jpg.rf.43cce7a6cb97da5a1ad0cefecdce4e0c.jpg
    digittake-346-jpg_jpg.rf.8c656d4294e517d82ced5ddeb3059920.jpg
    digittake-347-jpg_jpg.rf.82a8cb72447a881b5da59ff74829ba21.jpg
    digittake-348-jpg_jpg.rf.421fc5a30b0784b85410648566915062.jpg
    digittake-349-jpg_jpg.rf.e6f91896fc2a113269529c140bd33b66.jpg
    digittake-34-jpg_jpg.rf.30e25c033dba4d660b0d448391876ae1.jpg
    digittake-350-jpg_jpg.rf.d057fbf0cf59ebf86c0fd83263770980.jpg
    digittake-351-jpg_jpg.rf.495ab51c6661fefa164ab75b52a6ff43.jpg
    digittake-352-jpg_jpg.rf.f194b436486f021c08091e8caf6b6b4e.jpg
    digittake-353-jpg_jpg.rf.f2567505fa9810667601045cce09740c.jpg
    digittake-354-jpg_jpg.rf.e1a75f578fb1139b69617c09d9e0a9a1.jpg
    digittake-355-jpg_jpg.rf.4dcff8c053236a3dd1757a6f36b6e4cd.jpg
    digittake-356-jpg_jpg.rf.c523e11923884ae9e10c6e2267a628b5.jpg
    digittake-357-jpg_jpg.rf.6446f16c5c7157a36793a382780bbb31.jpg
    digittake-358-jpg_jpg.rf.4e89c24cf7810fc3a20da7a1cc8a8da4.jpg
    digittake-359-jpg_jpg.rf.5cb8ba165ad4f933be6787b93470c2c8.jpg
    digittake-35-jpg_jpg.rf.8a59d7b9357aabfcee643ec907aa92d4.jpg
    digittake-360-jpg_jpg.rf.072cc4f6b1ec4741b0227ced7cd2b364.jpg
    digittake-361-jpg_jpg.rf.326bff98049044cffc9d33c44e55f5c1.jpg
    digittake-362-jpg_jpg.rf.64e194f8da84b4b0efb1db941d5d88d9.jpg
    digittake-363-jpg_jpg.rf.cf011a9cac332c57e92e76bcdb4cb4f5.jpg
    digittake-364-jpg_jpg.rf.3b10b69adde2c156ea69d3104c052535.jpg
    digittake-365-jpg_jpg.rf.14af0a36229135840abb9f97aa1f43e4.jpg
    digittake-366-jpg_jpg.rf.9b397cbe80dfb6a937e7b0df99d94c3a.jpg
    digittake-367-jpg_jpg.rf.2b6e7a84c813708480dc7917391f2fd2.jpg
    digittake-368-jpg_jpg.rf.df690fb13cff0bfae028fd08057aec39.jpg
    digittake-369-jpg_jpg.rf.bd44da74349a2111aec9118b83616870.jpg
    digittake-36-jpg_jpg.rf.e627314a188d36552b8827ae53c359aa.jpg
    digittake-370-jpg_jpg.rf.7ad58af9ba414471582a817a5339ed5a.jpg
    digittake-371-jpg_jpg.rf.a9fea8fbbd62307b5ced07804a12617e.jpg
    digittake-372-jpg_jpg.rf.d6336aa505e3562f53a0ded536f764fa.jpg
    digittake-377-jpg_jpg.rf.3d46013271b36041c0968522a4371d89.jpg
    digittake-378-jpg_jpg.rf.45e3c243c049489ab0ef7db18ece1f79.jpg
    digittake-379-jpg_jpg.rf.82185306a550910de6418dada2f34176.jpg
    digittake-37-jpg_jpg.rf.25b628132e792a2aa3763f4d1e70f32f.jpg
    digittake-380-jpg_jpg.rf.bc63580baff53206f3a80cc4bd50cb0d.jpg
    digittake-381-jpg_jpg.rf.2dcaafde5a93aaaf4cef9abe4010cfdd.jpg
    digittake-382-jpg_jpg.rf.1239bc945550e19924bd4562e7bd5800.jpg
    digittake-383-jpg_jpg.rf.3ae5428831bde14fba6d1e7107fcb562.jpg
    digittake-384-jpg_jpg.rf.b194c1401c465bc8297003b7a5d9dee5.jpg
    digittake-385-jpg_jpg.rf.f931b87cb082bb54e0d99e9451228e7b.jpg
    digittake-386-jpg_jpg.rf.b5be7c71d4a6feb53aa8b129deae45f0.jpg
    digittake-387-jpg_jpg.rf.5c52599bd15803f4282ab334776ea14f.jpg
    digittake-388-jpg_jpg.rf.bf0620130210e3969d133bf086908c00.jpg
    digittake-389-jpg_jpg.rf.ce7ec86b2e4c80aafcbae193bd76f6ad.jpg
    digittake-38-jpg_jpg.rf.5a243b5bc1a96fb492b2eb56c2940553.jpg
    digittake-390-jpg_jpg.rf.923ac18c85e5713778d2e26193c2c7a0.jpg
    digittake-391-jpg_jpg.rf.e7c2bc18f755dec2789f1331db5242f0.jpg
    digittake-392-jpg_jpg.rf.dd25ee08b9e8261b4454505e909e4a19.jpg
    digittake-393-jpg_jpg.rf.fc5e61d05c885d74a0688d6e0833ea5c.jpg
    digittake-394-jpg_jpg.rf.f0135db64acb55e3f29112072cba9757.jpg
    digittake-395-jpg_jpg.rf.8727c5dec183a3b6f9758a7df1f63ba8.jpg
    digittake-396-jpg_jpg.rf.12023437c7c758ca7b49558d4220a8f2.jpg
    digittake-397-jpg_jpg.rf.fa02fd515a21725df09efc7517888896.jpg
    digittake-398-jpg_jpg.rf.75f14222ecf2e776a5e35bdfba68e859.jpg
    digittake-399-jpg_jpg.rf.8a0f377dafe0d1077f321971b103c6e6.jpg
    digittake-39-jpg_jpg.rf.47d481268965977a7a923cd29cc25f34.jpg
    digittake-400-jpg_jpg.rf.5516560b77b4932d5d99f19498b75bb4.jpg
    digittake-401-jpg_jpg.rf.6331f71bb2094787dc04bb9a16051f35.jpg
    digittake-402-jpg_jpg.rf.2d564d7f3bcaeba735400741be178cb5.jpg
    digittake-403-jpg_jpg.rf.9a40dabdab99df362a760b3e6bd446e5.jpg
    digittake-404-jpg_jpg.rf.d915263fe83d74083b02744dcad9bcdb.jpg
    digittake-405-jpg_jpg.rf.1ac14f19c0adc8a848f046d8b7792c43.jpg
    digittake-406-jpg_jpg.rf.3a30bcdb80ac90e75729a73a96aa0cba.jpg
    digittake-407-jpg_jpg.rf.80518ff9d4ac689b6064e0a2bbf1cd56.jpg
    digittake-408-jpg_jpg.rf.650b382da33b9647231a76fa3f1651a0.jpg
    digittake-409-jpg_jpg.rf.4e67eb07d7a281765e74402cf78f6471.jpg
    digittake-40-jpg_jpg.rf.133210c2f6beff3fd874e814dc12450a.jpg
    digittake-410-jpg_jpg.rf.eaca9fb7e734d3d0edb83bb87e4d83f9.jpg
    digittake-41-jpg_jpg.rf.5684e0d8d5c4d1df4010e64470a33497.jpg
    digittake-42-jpg_jpg.rf.b46536905af67a19f26f56a75a58661d.jpg
    digittake-43-jpg_jpg.rf.94d76abffc3c7d18fd104739fa44b2a3.jpg
    digittake-44-jpg_jpg.rf.b48687b13467e8961863c69f14cce047.jpg
    digittake-45-jpg_jpg.rf.bc0eb6182fea3e33dab02711454e71a1.jpg
    digittake-46-jpg_jpg.rf.b797c2c2bc50addef1c6b0736d546f7a.jpg
    digittake-47-jpg_jpg.rf.66d164b89d63b59c03e4912b3b83a294.jpg
    digittake-48-jpg_jpg.rf.99f008f1160b6966fd91301a8367ea8e.jpg
    digittake-49-jpg_jpg.rf.3536083ff8546672adbc8573deaa4852.jpg
    digittake-50-jpg_jpg.rf.a577b1f9b44d37671a786eaea6cbc063.jpg
    digittake-51-jpg_jpg.rf.798ed0fe5d44f8cda21bd02763063787.jpg
    digittake-52-jpg_jpg.rf.798cf1dcc7a60cbff6c43f5587082f4f.jpg
    digittake-53-jpg_jpg.rf.bd351b3764dcadb0084cb81d608b50fe.jpg
    digittake-54-jpg_jpg.rf.d807ccc48dc75b5ab6047784b06f5af7.jpg
    digittake-55-jpg_jpg.rf.90d631addbcd9884ad0c21e70ab29d38.jpg
    digittake-56-jpg_jpg.rf.8016736fb3675c6018e76975b6feb1d6.jpg
    digittake-57-jpg_jpg.rf.8b38ac88bbb253baffb04aa50ac8df2a.jpg
    digittake-58-jpg_jpg.rf.37b232cc220b1db86febddf7ef8fae0c.jpg
    digittake-59-jpg_jpg.rf.e832d94719e74516485c773e97d80b5e.jpg
    digittake-60-jpg_jpg.rf.70d0542c567ed02c44767bf7e4d9dd32.jpg
    digittake-61-jpg_jpg.rf.dd8b6499b15494dc6fb53b32a6b14e30.jpg
    digittake-62-jpg_jpg.rf.5c32f179631804835eb1d4f34cba3463.jpg
    digittake-63-jpg_jpg.rf.1d35283b521f38ad923070c9a842220b.jpg
    digittake-64-jpg_jpg.rf.26b80e1bf3c6827db15c35d3045df0bd.jpg
    digittake-65-jpg_jpg.rf.4dc54c494cee088390ca2cb770b51119.jpg
    digittake-66-jpg_jpg.rf.95253009459f7f68fc3d6c70ee0e598f.jpg
    digittake-67-jpg_jpg.rf.4a2a778ae5f800b49667119453df967e.jpg
    digittake-68-jpg_jpg.rf.4f2e9f8744d1352bb213cd966b5789e2.jpg
    digittake-69-jpg_jpg.rf.1f766478954f5d7b1ca5ebcf6f1c231b.jpg
    digittake-70-jpg_jpg.rf.e0c13619f60edf4d2dd999a9a5084048.jpg
    digittake-71-jpg_jpg.rf.6dfd2847b5b3a9e921741a836d8417b9.jpg
    digittake-72-jpg_jpg.rf.afdbdc444e6afdd4e68bdfa1356cdf92.jpg
    digittake-73-jpg_jpg.rf.8c072334274cbbbf182bedab3cac20e1.jpg
    digittake-74-jpg_jpg.rf.4658b2614af581ca11e86c082d761e91.jpg
    digittake-75-jpg_jpg.rf.c8894a23732f5e30130edeb82c7c0838.jpg
    digittake-76-jpg_jpg.rf.4d5c6aebb6f1c7c484985a199441fbf6.jpg
    digittake-77-jpg_jpg.rf.204298917a6355d3b0872303f9853067.jpg
    digittake-78-jpg_jpg.rf.d7b7a239383951702875cd5f0a4d019d.jpg
    digittake-79-jpg_jpg.rf.f2fb90ae5e5cc66f111ce504ec7a8d2b.jpg
    digittake-80-jpg_jpg.rf.37e8a01df4463ce8c06a8d5aecc1333d.jpg
    digittake-81-jpg_jpg.rf.94eed3cf9672c4656144aaaef92348bc.jpg
    digittake-82-jpg_jpg.rf.a20ef1f4eb7665a94d67d2b75245a4c1.jpg
    digittake-83-jpg_jpg.rf.f8a9191a070e177d3749134f2e8177aa.jpg
    digittake-84-jpg_jpg.rf.665670204a9d2a538dbbb37c990abdcd.jpg
    digittake-85-jpg_jpg.rf.83c2933ee9df2884cc977bddffdb8324.jpg
    digittake-86-jpg_jpg.rf.23d448e0424b9fadc2eeaf48faae1c98.jpg
    digittake-87-jpg_jpg.rf.b6c239e3a75d633b0afbb255bba95919.jpg
    digittake-88-jpg_jpg.rf.8f580048c8f55eb4fa7b7624cc29c9d2.jpg
    digittake-89-jpg_jpg.rf.bad7cfedd24a486d4c548f2183d33387.jpg
    digittake-90-jpg_jpg.rf.a611e634ae3914d5fe0144f0cd513a67.jpg
    digittake-91-jpg_jpg.rf.b3aa3e56915744acc6cb0d6996689cf4.jpg
    digittake-92-jpg_jpg.rf.4ca2f46e520a2740c1ee9e4be33c4750.jpg
    digittake-93-jpg_jpg.rf.8f21c0ccb915fc3d6f29f8cb28ce7d51.jpg
    digittake-94-jpg_jpg.rf.4d345f184e72d51cf58e81bac2bc1657.jpg
    digittake-95-jpg_jpg.rf.0685c5d1ce258c718b0392f52e9825e6.jpg
    digittake-96-jpg_jpg.rf.deee8f43c2fccce4e1d8965c9b3defb0.jpg
    digittake-97-jpg_jpg.rf.dfbc060e1865faf636c1e2a23583b32e.jpg
    digittake-98-jpg_jpg.rf.c37cf21c9b060cf49c363f6744367757.jpg
    digittake-99-jpg_jpg.rf.d2b5ea619286f7cd5cad06fc0de2864e.jpg
    save_dirrattempt2-save_dirattempt2260693f35709a7fe26304978d77ab9ccc45fa5ad-jpg-jpg_jpg.rf.2e7fefe6a5129d051383093c9966b9c8.jpg
    save_dirrattempt2-save_dirattempt2373864709100c8c0b8f3e6c3722ff9453747b889-jpg-jpg_jpg.rf.6328ec020eb081bc98df0c020299ddf6.jpg
    save_dirrattempt2-save_dirattempt2647517ae17cb333fe4187bdaec6db1e919c52896-jpg-jpg_jpg.rf.c2bc4f162a43034a95bbef52df0b6860.jpg
    save_dirrattempt2-save_dirattempt281200578963bee698c6c450dfb448255f0b991af-jpg-jpg_jpg.rf.e208c0167816de1da417b78706047f85.jpg
    save_dirrattempt2-save_dirattempt29879e8fcac36d3a82bea304b0af56bde5533f5f1-jpg-jpg_jpg.rf.068cd47f231923eb483e913cb024edbf.jpg
    save_dirrcropped-jpg_jpg.rf.00d0bc296f3a6561d1d088c0b65c901d.jpg
    save_dirrcropped-jpg_jpg.rf.036528472f0aaf7b3c46dd54647149f1.jpg
    save_dirrcropped-jpg_jpg.rf.06cc0ffa5171bf327d37cb7a9bba098c.jpg
    save_dirrcropped-jpg_jpg.rf.093f08ffcf8f66e8cf22a263157e2693.jpg
    save_dirrcropped-jpg_jpg.rf.0a15208aae3af62061845fd39bcda794.jpg
    save_dirrcropped-jpg_jpg.rf.0d3c2fb3e34660d81a8b21142dfe3501.jpg
    save_dirrcropped-jpg_jpg.rf.10c170123d876ceef3e79679f1d948ba.jpg
    save_dirrcropped-jpg_jpg.rf.127f02f69b5f5146c1376b9b592d58c7.jpg
    save_dirrcropped-jpg_jpg.rf.18532b3fef8ff09611fcf98c5208b33c.jpg
    save_dirrcropped-jpg_jpg.rf.1d5d52e5464ac4d4c5831fe2f1e3a132.jpg
    save_dirrcropped-jpg_jpg.rf.1e9ec96784de76926635253e68ac58b5.jpg
    save_dirrcropped-jpg_jpg.rf.25997bfa90bee0f2e2a02a77edb16a57.jpg
    save_dirrcropped-jpg_jpg.rf.265d11389f8a6e0334b0f0bbbbb1b584.jpg
    save_dirrcropped-jpg_jpg.rf.29864ad14491fd39fa28639dcc57aba9.jpg
    save_dirrcropped-jpg_jpg.rf.340eb8ea3c457536f9ca160912bf907b.jpg
    save_dirrcropped-jpg_jpg.rf.344a46a569f5bd9a39b0e7bb5379570e.jpg
    save_dirrcropped-jpg_jpg.rf.354d1845656be3c316728ec0f68ff67d.jpg
    save_dirrcropped-jpg_jpg.rf.3969b3430d647b08f64d647fe28cbd30.jpg
    save_dirrcropped-jpg_jpg.rf.3aa6397d1581eaa8bd3de49909b70285.jpg
    save_dirrcropped-jpg_jpg.rf.3d3166a98e8b5308e96b07470cc2b123.jpg
    save_dirrcropped-jpg_jpg.rf.40d9d752978239da12f2f82a0aafd25e.jpg
    save_dirrcropped-jpg_jpg.rf.42d80e408660040c5f4abb139917d757.jpg
    save_dirrcropped-jpg_jpg.rf.446aca2e2d6f8282d319bfb88cb805df.jpg
    save_dirrcropped-jpg_jpg.rf.45f85d3540bc991d5a17eaa81bb5108c.jpg
    save_dirrcropped-jpg_jpg.rf.4c8671f8dbb897a9042bad61da14f2e6.jpg
    save_dirrcropped-jpg_jpg.rf.4ca1fe5e2123ada4de8e773f08fcc0b1.jpg
    save_dirrcropped-jpg_jpg.rf.4eec64317693d49b544717fc09d23c8e.jpg
    save_dirrcropped-jpg_jpg.rf.5026f61658e6eee54a51667af5a36aa1.jpg
    save_dirrcropped-jpg_jpg.rf.55a13907e1a375d798dd979bc64f6a9b.jpg
    save_dirrcropped-jpg_jpg.rf.5c0a07519edc67ca2a2dca1829f2454a.jpg
    save_dirrcropped-jpg_jpg.rf.5d1491fe59273231921bf5e05bda5c19.jpg
    save_dirrcropped-jpg_jpg.rf.652e98ef445cf95befeecba8537671bd.jpg
    save_dirrcropped-jpg_jpg.rf.6a6536389e7798fe3983c6b581380882.jpg
    save_dirrcropped-jpg_jpg.rf.7177e33650ef80bee8afc9b7f6173c49.jpg
    save_dirrcropped-jpg_jpg.rf.719342a0e530ccf1c925250eaaedc9f0.jpg
    save_dirrcropped-jpg_jpg.rf.729a51122a990f331bc7a55f50f5bcae.jpg
    save_dirrcropped-jpg_jpg.rf.73be0ffa3cb2523080657db90f75c124.jpg
    save_dirrcropped-jpg_jpg.rf.783b6535221eb9f6af474295b6b1cb0f.jpg
    save_dirrcropped-jpg_jpg.rf.79fb539812f1d69cd7ea9fa9eddee48d.jpg
    save_dirrcropped-jpg_jpg.rf.89ee29085418151c90e69ad390407bc4.jpg
    save_dirrcropped-jpg_jpg.rf.8e5c214282b88d1e275357286636fa36.jpg
    save_dirrcropped-jpg_jpg.rf.8f0ffcc78b78852f80a202705a88da6a.jpg
    save_dirrcropped-jpg_jpg.rf.99ecd9f32bc3332a17b9900174127e1f.jpg
    save_dirrcropped-jpg_jpg.rf.a23d09f9f5b536f80900d507ee61225c.jpg
    save_dirrcropped-jpg_jpg.rf.ad76e86cf2c9b9de942bfaeca9693116.jpg
    save_dirrcropped-jpg_jpg.rf.add34357e3e0ba78cb848eb8a3ee29d1.jpg
    save_dirrcropped-jpg_jpg.rf.ae306a04b6fcb3d76b74738d643233fc.jpg
    save_dirrcropped-jpg_jpg.rf.aeb6d4aa8b0ae48f1dbb38a696304340.jpg
    save_dirrcropped-jpg_jpg.rf.b304dcf15dee8660c66bca1526ca090e.jpg
    save_dirrcropped-jpg_jpg.rf.b91e275de28a38f6197460ccb7e475a6.jpg
    save_dirrcropped-jpg_jpg.rf.c61178ecb6429b18ff74d411b25e6f58.jpg
    save_dirrcropped-jpg_jpg.rf.c6f7dc25be5868cfd6cd6cbd3dc00bd2.jpg
    save_dirrcropped-jpg_jpg.rf.c8492377d4bbf62dc6e7f73d15472b59.jpg
    save_dirrcropped-jpg_jpg.rf.cbcb828b62f06725ede5454e7b23eb4f.jpg
    save_dirrcropped-jpg_jpg.rf.cfe6802d8b33f44abff83caf535d3747.jpg
    save_dirrcropped-jpg_jpg.rf.d0118d946dd7ce0824f9a691b4a83a6b.jpg
    save_dirrcropped-jpg_jpg.rf.d0b41dd2a43df201342f97756fc23906.jpg
    save_dirrcropped-jpg_jpg.rf.d2b67f192d5698ae7645cf611c821c53.jpg
    save_dirrcropped-jpg_jpg.rf.d3c862ccfba6b3ea49947f0c768f847c.jpg
    save_dirrcropped-jpg_jpg.rf.d677c4c1b76598ce5e00374b199bbdb1.jpg
    save_dirrcropped-jpg_jpg.rf.dc3d5d83e2e88ccb30d0db95438d8afd.jpg
    save_dirrcropped-jpg_jpg.rf.dd583a2b0dd0ee135d0f3428079609be.jpg
    save_dirrcropped-jpg_jpg.rf.e357fe05edce4931d5b8fbb7b4d73e70.jpg
    save_dirrcropped-jpg_jpg.rf.f0d478fae784affc416576aa16fd59e7.jpg
    save_dirrcropped-jpg_jpg.rf.f26b4cd12193a002c67aafdd9c7b13bd.jpg
    save_dirrcropped-jpg_jpg.rf.fb02b25f56c6623124259b13279d24f4.jpg
    save_dirrcropped-jpg_jpg.rf.fbd3e47077e5d35146486382bf6caf6d.jpg
    save_dirrcropped-jpg_jpg.rf.fef1a98812ec2f93a757317e1e17c929.jpg
    save_dirrdigittake-102-jpg-cropped-jpg-jpg_jpg.rf.ea6c1f32023f6d46561f9d455cb34696.jpg
    save_dirrdigittake-106-jpg-cropped-jpg-jpg_jpg.rf.f236e6eb20c5c337a4e9f494038fddff.jpg
    save_dirrdigittake-116-jpg-cropped-jpg-jpg_jpg.rf.b358f4b5d44bb17558aeb77c9572369c.jpg
    save_dirrdigittake-118-jpg-cropped-jpg-jpg_jpg.rf.25ad15f4a6a69c22b01c04d59bfaa48a.jpg
    save_dirrdigittake-119-jpg-cropped-jpg-jpg_jpg.rf.770bbf12a0f2198215852283e5e8f902.jpg
    save_dirrdigittake-120-jpg-cropped-jpg-jpg_jpg.rf.f36a35d6b68b491d0b72af582e2533f4.jpg
    save_dirrdigittake-123-jpg-cropped-jpg-jpg_jpg.rf.c94c9320d0a650cfb254f8c3b2f27ea2.jpg
    save_dirrdigittake-124-jpg-cropped-jpg-jpg_jpg.rf.f94aa8d7d2c486ed2328f347a33df139.jpg
    save_dirrdigittake-126-jpg-cropped-jpg-jpg_jpg.rf.1bf2f8d5284b119a3c15537a1e86efc1.jpg
    save_dirrdigittake-127-jpg-cropped-jpg-jpg_jpg.rf.6e95de28a77ea3b744919efcf916044d.jpg
    save_dirrdigittake-128-jpg-cropped-jpg-jpg_jpg.rf.2b9e8c99f03c5d50afba140610460281.jpg
    save_dirrdigittake-129-jpg-cropped-jpg-jpg_jpg.rf.a5e14802b6b1e44d0c7dc307c989b52d.jpg
    save_dirrdigittake-131-jpg-cropped-jpg-jpg_jpg.rf.ecfd1ae493cbdfbb580d5a19d03510d9.jpg
    save_dirrdigittake-34-jpg-cropped-jpg-jpg_jpg.rf.8b694840d3b2c12f9d24cfdd56b114a1.jpg
    save_dirrdigittake-35-jpg-cropped-jpg-jpg_jpg.rf.c01d5e021faa3f22f46c2ec94baaaaaf.jpg
    save_dirrdigittake-69-jpg-cropped-jpg-jpg_jpg.rf.20ca13a1b5d9df473f695ae012e3f4d2.jpg
    save_dirrdigittake-70-jpg-cropped-jpg-jpg_jpg.rf.a4473cc96fd2536e83763fb83c56e452.jpg
    save_dirrdigittake-72-jpg-cropped-jpg-jpg_jpg.rf.3dc4adc006bdc8a0d08f307368af013f.jpg


```
from ModelAssistedLabel.fileManagement import Generation

backup_dir = "archive/Generation/zips"

g = Generation(repo=labeled_images, 
               out_dir=backup_dir,
               verbose=True)

g.set_split()
g.get_split()
```




    [{'train': 0}, {'valid': 0}, {'test': 0}]



```
zipped = g.write_split_to_disk(descriptor=export_folder)
```

    
    dirs ['./train', './valid', './test']
    yaml archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01/data.yaml
    subdir train
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01
    subdir valid
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01
    subdir test
    	outdir archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01
    os.listdir ['train', 'valid', 'test', 'data.yaml']


Next, the images need to be written in a way so that the Ultralytics repository can understand their content. The `Autoweights` class both organizes data and create weights. Running an "initialize" command makes changes to the disk.

```
from ModelAssistedLabel.train import AutoWeights
#configure a basic AutoWeights class instance
aw = AutoWeights(name=export_folder, out_dir=backup_dir)

# create train/valid/test split from a bag of labeled images (recusively seek out images/labels)
aw.initialize_images_from_zip(zipped)
```

    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01/train' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01/valid' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01/test' .
    mv 'unzipped/archive/Generation/zips/Final Roboflow Export (841)seven segment digits - 1 21-03-27 05-32-01/data.yaml' .


Peep on the sizes of the train/valid/test groups.

```
aw.traverse_resources()
```

    train/images
    	 > 0 files
    train/labels
    	 > 0 files
    valid/images
    	 > 0 files
    valid/labels
    	 > 0 files
    test/images
    	 > 0 files
    test/labels
    	 > 0 files
    File:  data.yaml


## Generate Weights

With the images written to disk, we can run the Ultralytics training algorithm. I loved watching the progress fly by in real time on the original `train.py`. Fortunately, the Ultralytics folk write the results file to disk so the model's training data is still accessible!

```
from ModelAssistedLabel.train import Trainer
```

```
%%time

aw.generate_weights(epochs=2, yaml_data=Defaults().trainer_template)
```

    CPU times: user 38.5 ms, sys: 10.7 ms, total: 49.3 ms
    Wall time: 8.4 s





    'yolov5/runs/train/seven segment digits - 15/'



The results folder is stored as an attribute as well, and it has a lot of data stored therein.

```
import os
aw.last_results_path, len(os.listdir(aw.last_results_path))
```




    ('yolov5/runs/train/seven segment digits - 15/', 4)



However, the weights are stored in a subfolder called (aptly) "weights".

```
os.listdir(aw.last_results_path + "/weights")
```




    []



View the last couple lines 

```
ls
```

     00_config.ipynb         [0m[01;34mModelAssistedLabel[0m/
     01_split.ipynb         'ModelAssistedLabel config.json'
     02_train.ipynb         'modelassistedlabel splash.jpg'
     03_detect.ipynb         README.md
     [01;34marchive[0m/                settings.ini
    '_capture input.ipynb'   setup.py
    'colab img.png'         [01;34m'seven segment digits - 1'[0m/
     CONTRIBUTING.md         _Synch.ipynb
     data.yaml               [01;34mtest[0m/
     docker-compose.yml     [01;34m'test (1)'[0m/
     [01;34mdocs[0m/                   [01;34mtrain[0m/
    [01;34m'Image Repo'[0m/           [01;34m'train (1)'[0m/
     index.ipynb             [01;34mvalid[0m/
     LICENSE                [01;34m'valid (1)'[0m/
     Makefile                [01;34myolov5[0m/
     MANIFEST.in


```
with open(aw.last_results_path + "results.txt") as results_file:
  results = results_file.readlines()
print("Epoch   gpu_mem       box       obj       cls     total    labels  img_size")
results[-5:]
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-29-623b1110f1f6> in <module>()
    ----> 1 with open(aw.last_results_path + "results.txt") as results_file:
          2   results = results_file.readlines()
          3 print("Epoch   gpu_mem       box       obj       cls     total    labels  img_size")
          4 results[-5:]


    FileNotFoundError: [Errno 2] No such file or directory: 'yolov5/runs/train/seven segment digits - 13/results.txt'


## Labeling a new set of images

And the `Viewer` class doesn't care how recently your weights were generated so you can plug in existing weights.

```
from ModelAssistedLabel.detect import Viewer

# access the folder of results from the AutoWeights instance
results_folder = aw.last_results_path

# I'm choosing to use the best weight.
weight_path = results_folder + "/weights/best.pt"

# Viewer needs a set of weights and an array of labels for the detected object types
v = Viewer(weight_path, class_idx)
```

Selects all images in the unlabeled folder and let's us look through the computer's eyes at the images.

```
%matplotlib inline 
import random, glob

images = glob.glob(f"./{unlabeled_images}/*.jpg")

for image in random.sample(images,5):
  v.plot_for(image)
```

```
results = []
for image in images:
  results.append(v.predict_for(image))
```

## Exporting annotated images

Store the class labels with index 0 on line 1, index 1 on line 2, and so on.

```
with open(os.path.join(export_folder, "label_map.txt"), "w") as label_map:
  label_map.writelines("\n".join(class_idx))
```

Ensure that image/label pairs have a common root filename

```
import random, PIL, shutil
salt = lambda: str(random.random())[2:]

for result in results:
  #generate a likely-to-be-unique filename
  shared_root = Defaults._itername(f"{project_name}-{salt()}")

  #save the image to the outfile
  image = PIL.Image.open(result["image path"])
  image.save(os.path.join(export_folder, "images", f"{shared_root}.jpg"))

  #save the predictions to the outfile
  predictions = result["predictions"]
  with open(os.path.join(export_folder, "labels", f"{shared_root}.txt"), "w") as prediction_file:
    prediction_file.writelines("\n".join([x["yolov5 format"] for x in predictions]))

#check if weights were generated
if aw is not None and os.path.exists(aw.last_results_path):
  print(f"Moving yolov5 results folder: {aw.last_results_path}")
  shutil.move(aw.last_results_path, export_folder)
else:
  print("No weights to save")
```

At this point I would have uploaded this set of image/label pairs to Roboflow for correction and annotation. As the model grows more accurate, I would alter camera position or lighting until the model started stumbling again. I want to be keeping the model on its toes!

To be transparent, I developed a [custom React annotator](https://github.com/PhilBrockman/autobbox) that better suited my needs.

I labeled dozens upon dozens and dozens of images with Roboflow and would recommend their free annotation service! 

## Wrap up

My original goal of "smartifying" my rowing machine is closer than ever. 

It is possible to parse workout information (thought currently, I only have access to a maximum of 4 digits). I wonder if the model could keep up if there were 20+ digits to capture.

I know that lighting and camera position have an effect on accuracy. Here's how I'm holding my computer steady as I modify the lighting: [standing](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-laptop-mount.jpg), [floor 1](https://raw.githubusercontent.com/PhilBrockman/ModelAssistedLabel/master/DIY-computer-capture.jpg), [floor 2](https://github.com/PhilBrockman/ModelAssistedLabel/blob/master/DIY-capture.jpeg?raw=true).

Here are 3 runs captured under different lighting conditions:
* `21-3-22 rowing (200) 7:50-12:50` (direct lighting from one light source)
* `21-3-22 rowing (200) 1:53-7:00` (direct lighting from one light source with glare)
* `21-3-18 rowing 8-12 ` (direct light and ambient lamps turned on)
{% include note.html content='All unlabeled images were taken inside a blacked-out room. The are stored in `Image Repo/unlabeled/`' %}





### Lingering Questions

My labeled images are disorderly. There's data from other rowing machines and from [a kind *stranger*'s github repo](https://github.com/SachaIZADI/Seven-Segment-OCR). Some images have been cropped to only include the display. Did having varied data slow me down overall? Or did it make the models more robust? 

