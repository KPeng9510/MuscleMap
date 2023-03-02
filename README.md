
## MuscleMap: Towards Video-based Activated Muscle Group Estimation

### We would like to address something novel, helpful and yet overlooked by the community
We tackle the new task of video-based Activated Muscle Group Estimation (AMGE) aiming at identifying currently activated muscular regions of humans performing a specific activity. Video-based AMGE is an important yet overlooked problem. To this intent, we provide the MuscleMap136 featuring >15K video clips with 136 different activities and 20 labeled muscle groups. This dataset opens the vistas to multiple video-based applications in sports and rehabilitation medicine. We further complement the main MuscleMap136 dataset, which specifically targets physical exercise, with Muscle-UCF90 and Muscle-HMDB41, which are new variants of the well-known activity recognition benchmarks extended with AMGE annotations. 
<div align="center">
  <img src="https://github.com/KPeng9510/MuscleMap/blob/master/demo/TS.png" width="500px"/>
</div>

### We want to protect people from injury and show the best wishes to AI
Our model can be used to predict abnormal activity execution when a person is doing a weird action, which can be leveraged to detect abnormal exercise behavior and dangerous muscle usage for health care and sports.

## Acquisition of the Dataset

Acquisition of MuscleMap136: The user aggreement document is preparing now. Once you are interested in our dataset, please send an email to kunyu.peng@kit.edu, the processed dataset will be forwarded to you after you aggree the user agreement and sign it. Once the user agreement document is ready, the corresponding information will be released as soon as possible, thank you for your understanding.

Acquisition of Muscle-UCF90 and Muscle-HMDB41: These two datasets also need the user agreement. Please also forwad the official aggreement for using both HMDB and UCF101 to the above mentioned email.
### Highlights of our dataset:

We open the door for video-based activated muscle group estimation task and contributes a specific designed AMGE datsest named as MuscleMap136 together with UCF-90 and HMDB41 which could deliver a good benchmark for this new task. AMGE is an important but yet overlooked task both in computer vision field and the health care field. We hope our contribution could be interesting.

## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md). The code is built based one PySlowFast (https://github.com/facebookresearch/SlowFast), we appreciate the great work from PySlowFast Team!

## Our model

### Highlights of our model:

We for the first time make used of MCTs to distill the knowledge from the RGB difference modality to the RGB modality to enhance the generalization ability of the end-to-end model on the activated muscle group estimation task. During inference time we do not use another modality to preserve the efficiency of our proposed approach. We proposed a Sender-Receiver schema for the knowledge distillation based on MCTs. We further proposed the MCTF module to fuse the original MCTs and the receiver MCTs of main RGB model to achieve a better generalization ability.

<div align="center">
  <img src="https://github.com/KPeng9510/MuscleMap/blob/master/demo/main_Model.png" width="900px"/>
</div>


## Evaluation Protocol

We used test_unseen and val_unseen to test the model on actions excluded from the train set for all three datasets. Test_unseen and val_unseen contain actions with unseen muscle activation combinations to test generalization. Mean averaged precision (mAP) is used as the evaluation metric. The sample list for train, val and test will be sent to you together with the dataset.


