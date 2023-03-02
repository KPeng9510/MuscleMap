
## MuscleMap: Towards Video-based Activated Muscle Group Estimation
We tackle the new task of video-based Activated Muscle Group Estimation (AMGE) aiming at identifying currently activated muscular regions of humans performing a specific activity. Video-based AMGE is an important yet overlooked problem. To this intent, we provide the MuscleMap136 featuring >15K video clips with 136 different activities and 20 labeled muscle groups. This dataset opens the vistas to multiple video-based applications in sports and rehabilitation medicine. We further complement the main MuscleMap136 dataset, which specifically targets physical exercise, with Muscle-UCF90 and Muscle-HMDB41, which are new variants of the well-known activity recognition benchmarks extended with AMGE annotations. 
<div align="center">
  <img src="https://github.com/KPeng9510/MuscleMap/blob/master/demo/TS.png" width="500px"/>
</div>


## Acquisition of the Dataset

Acquisition of MuscleMap136: The user aggreement document is preparing now. Once you are interested in our dataset, please send an email to kunyu.peng@kit.edu, the processed dataset will be forwarded to you after you aggree the user agreement and sign it. Once the user agreement document is ready, the corresponding information will be released as soon as possible, thank you for your understanding.

Acquisition of Muscle-UCF90 and Muscle-HMDB41: These two datasets also need the user agreement. Please also forwad the official aggreement for using both HMDB and UCF101 to the above mentioned email.


## Installation

Please find installation instructions in [INSTALL.md](INSTALL.md). The code is built based one PySlowFast (https://github.com/facebookresearch/SlowFast), we appreciate the great work from PySlowFast Team!

## Our model
<div align="center">
  <img src="https://github.com/KPeng9510/MuscleMap/blob/master/demo/main_Model.png" width="900px"/>
</div>


## Evaluation Protocol

We used test_unseen and val_unseen to test the model on actions excluded from the train set for all three datasets. Test_unseen and val_unseen contain actions with unseen muscle activation combinations to test generalization. Mean averaged precision (mAP) is used as the evaluation metric.


