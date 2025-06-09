# UAV-MOT with Camera Motion Compensation using Optical Flow

This project is a Multi-Object Tracking (MOT) method for UAV-captured videos, designed to handle the challenges posed by camera movement. It is based on the [FairMOT](https://github.com/ifzhang/FairMOT) model, with an added position correction algorithm using Optical Flow.


## 1. Overview

Traditional Multi-Object Tracking (MOT) methods perform well on videos from static cameras. However, in videos captured by Unmanned Aerial Vehicles (UAVs), the camera moves freely in 3D space. This causes significant changes in object appearance and complicates their motion paths, leading to a notable decline in tracking performance for conventional MOT methods.



## 2. Baseline Model
This work is built upon the excellent research of **FairMOT**. We use it as our baseline for both object detection and tracking. For details on the original model, please refer to their paper and repository.

* **Paper:** [FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888) 
* **Official GitHub:** [https://github.com/ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)

## 3. Proposed Method: Position Correction with Optical Flow

## Acknowledgements
- This project is heavily based on FairMOT. We thank the authors for their great work.
- We thank the creators of the [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)  and [UAVDT](https://arxiv.org/abs/1804.00438)  datasets for making their valuable data publicly available.
