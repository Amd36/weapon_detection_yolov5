# Weapon_detection_using_yolov5
This repo uses yolov5 custom object detection model that is trained specifically for weapon detection along with simillar handheld objects. The steps to train and test the model is highlighted below:

## Prerequisites
If you want to train this model using your own machine then create a virtual environment using anaconda or python venv. Acitvate your environment and run the following commands:

    cd yolov5
    pip install -r requirements.txt

These commands will install the necessary dependencies into your environment.

## Train the model
Now that we have our environment prepared, we can use **train.py** to train the object detection model. I have already prepared the dataset and organized them into proper directories for the training process. The dataset I am using is provided in the link:

https://github.com/ari-dasci/OD-WeaponDetection

I am using only the **Weapon and similar handheld object** dataset. If you want to add your own dataset, refer to the **preprocess_dataset2.py** where I have prepared a detailed script that unzips the dataset and sorts them into proper directories. The annotations that come bundled with the zip file are also not compatible with yolov5 but the script handles all these and creates proper annotations too. The script is enough detailed so refer to that for further use.

I also have prepared a **weapon_data.yaml** file that is necessary for training yolov5 models. 

With all that being said, we will begin to train the model. Use the following command(or change the arguments as necessary for your dataset):

    python train.py --data weapon_data.yaml --weights yolov5s.pt --img 640 --epochs 20 --cache ram

The trining should start. The time required for this step might take some time depending on what machine you have so be patient and let the computer do its thing. The results will be saved in the following directory:

    yolov5\runs

