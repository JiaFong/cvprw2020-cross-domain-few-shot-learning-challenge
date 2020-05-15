# cvprw2020 cross-domain few-shot learning challenge

The source code of LMM-PQS for cvprw 2020 cross-domain few-shot learning challenge.

### Datasets
Please check the README.md in filelists.

### Pretrained Model
* ResNet10 Baseline/ProtoNet/RelationNet are provided in logs/checkpoints/miniImageNet.

* ResNet18 Baseline : https://drive.google.com/drive/folders/1qzUxKOGwl5OwAmhOCT149TZFuofnbs2A?usp=sharing
* ResNet18 Protonet : https://drive.google.com/drive/folders/17JxacjaPDx8o9L0Fy8KGTTMlyoSAMYVS?usp=sharing

## General information

* **No meta-learning in-domain**
* Only ImageNet based models or meta-learning allowed.
* 5-way classification
* n-shot, for varying n per dataset
* 600 randomly selected few-shot 5-way trials up to 50-shot (scripts provided to generate the trials)
* Average accuracy across all trials reported for evaluation.


### Specific Tasks:

**EuroSAT**

  • Shots: n = {5, 20, 50}

**ISIC2018**

  • Shots: n = {5, 20, 50}

**Plant Disease**

  • Shots: n = {5, 20, 50}

**ChestX-Ray8**

  • Shots: n = {5, 20, 50}

### Environment
Python 3.7
Pytorch 1.3.1
h5py 2.9.0

### Steps
1. Download all the needed datasets (see README.md in filelists).
2. Change configuration in config.py to the correct paths in your own computer.
3. Train baseline model on miniImageNet
- **Standard supervised learning on miniImageNet**

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug
    ```
- **Train meta-learning method (protonet) on miniImageNet**
The available method list: [protonet/protonet_ptl/relationnet/relationnet_softmax]

    ```bash
        python ./train.py --dataset miniImageNet --model ResNet10  --method protonet --n_shot 5 --train_aug
    ```
4. Save features for evaluation (optional, if there is no need to adapt the features during testing) 

- **Save features for testing**

    ```bash
        python save_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

5. Test with saved features (optional, if there is no need to adapt the features during testing) 

    ```bash
        python test_with_saved_features.py --model ResNet10 --method baseline --dataset CropDisease --n_shot 5 --train_aug
    ```

6. Test
There are three fine-tune method you can choose:

* **finetune_backbone_linear.py**:
The original finetune.py in the challenge code.

The available method list: [protonet/protonet_ptl/baseline]
The available model list: [ResNet10/ResNet18]

- You can choose to fine-tune the backbone or not.
    ```bash
        python finetune_backbone_linear.py --model ResNet10 --method baseline  --train_aug --n_shot 5 --freeze_backbone
    ```

    ```bash
        python finetune_backbone_linear.py --model ResNet10 --method baseline  --train_aug --n_shot 5 
    ```

- You can also train a new linear layer with the backbone from few-shot models.
    ```bash
        python finetune_backbone_linear.py --model ResNet10 --method protonet  --train_aug --n_shot 5 
    ```

* **finetune_few_shot_models_PQS.py**:
This method will apply the pseudo query set to the few-shot model you want to fine-tune with. 
The few-shot models will execute the same as in the meta-training phase, using support set and pseudo query set to fine-tune the backbone.
So, there is no option for --freeze_backbone in this file.

The available method list: [protonet/protonet_ptl/relationnet/relationnet_softmax]
The available model list: [ResNet10/ResNet18]

    ```bash
        python finetune_few_shot_models_PQS.py --model ResNet10 --method protonet  --train_aug --n_shot 5
    ```

* **finetune_backbone_LMM-PQS.py**:
This method fine-tunes the backbone in the few-shot style, using the backbone from Baseline or ProtoNet and applying a cosine mean-centroid classifier.
The PTLoss and LMM(CosFace) are applied during fine-tuning.

The available method list: [protonet/protonet_ptl/baseline]
The available model list: [ResNet10/ResNet18]

    ```bash
        python finetune_backbone_LMM-PQS.py --model ResNet10 --method baseline  --train_aug --n_shot 5
    ```


No matter which finetune method you chosse, a dataset contains 600 tasks.
After evaluating 600 times, you will see the result like this: 600 Test Acc = 49.91% +- 0.44%.

### Challenge Website and Repository
You can visit these website for more information.

website: https://www.learning-with-limited-labels.com/
repository: https://github.com/IBM/cdfsl-benchmark

