#### Introduction

This code base contains utilities for processing segment kinematics obtained via biplane fluoroscopy and skin marker trajectories obtained from motion capture. The code base is broadly organized into two sections: `biplane_kine` and `biplane_tasks`. `Biplane_tasks` contains scripts for data analysis tasks, such as smoothing, determining the effects of smoothing, computing segment kinematics from skin markers, etc. `Biplane_kine` is a library that contains the algorithmic logic that underpins the tasks in `biplane_tasks`.

#### Installation

```
git clone https://github.com/klevis-a/process-vicon-biplane
cd process-vicon-biplane
conda env create -f environment.yml
```

#### Background

Although extensively utilized to estimate body segment kinematics, skin-marker motion capture is plagued by errors arising from soft-tissue artefact (STA). STA arises from the decoupled movement of skin-markers relative to the underlying bone as caused by skin gliding, muscle contraction, and inertial effects. In the upper extremity, the kinematic error induced by STA ranges from 2-15% for the primary axis of motion but can be >100% for secondary axes of motion. In contrast, model-based markerless tracking of biplane fluoroscopy recordings has sub-millimeter and sub-degree accuracy.

Although it is possible to design STA-attenuation algorithms (an active area of research), the design and validation of STA attenuation algorithms is currently impeded by the lack of artefact-free data for quantifying STA. Prior studies have recorded human motion using both biplane fluoroscopy (serving as ground truth) and skin markers to quantify STA. However, biplane fluoroscopy studies require specialized equipment/software, expose patients to radiation, and necessitate significant manual post-processing. As such, the largest STA quantification study in the upper extremity is limited to 4 subjects.

Our group has collected a dataset of humeral and scapular kinematics from 20 health subjects simultaneously imaged using a biplane fluoroscopy system and a 10-camera motion capture system (Vicon) at 100 Hz. This repository contains Python code for pre-processing and analyzing this dataset with the aim of quantifying STA.
