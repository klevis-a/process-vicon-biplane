#### Introduction

This code base contains utilities for processing segment kinematics obtained via biplane fluoroscopy and skin marker trajectories obtained from motion capture. The code base is broadly organized into two sections: `biplane_kine` and `biplane_tasks`. `Biplane_tasks` contains scripts for data analysis tasks, such as smoothing, determining the effects of smoothing, computing segment kinematics from skin markers, etc. `Biplane_kine` is a library that contains the algorithmic logic that underpins the tasks in `biplane_tasks`.

#### Installation

```
git clone https://github.com/klevis-a/process-vicon-biplane
cd process-vicon-biplane
conda env create -f environment.yml
```

#### Configuration

All code within the repository relies on two configuration files (`logging.ini` and `parameters.json`) to locate the associated data repository, configure analysis parameters, and instantiate logging. The location of the `config` directory is specified as a command line parameter (so it is feasible for this folder to reside anywhere in the filesystem). Analysis tasks must be executed as module scripts:

`python -m biplane_tasks.export.export_threejs_biplane_markers config`

Template `logging - template.ini` and `parameters - template.json` files are located in the `config` directory and should be copied and renamed to `logging.ini` and `parameters.json`. Each analysis task contains Python documentation describing its utility and the parameters that it expects from `parameters.json`.

#### Background

Although extensively utilized to estimate body segment kinematics, skin-marker motion capture is plagued by errors arising from soft-tissue artefact (STA). STA arises from the decoupled movement of skin-markers relative to the underlying bone as caused by skin gliding, muscle contraction, and inertial effects. In the upper extremity, the kinematic error induced by STA ranges from 2-15% for the primary axis of motion but can be >100% for secondary axes of motion.

Although it is possible to develop STA-attenuation algorithms (an active area of research), their design and validation is currently impeded by the lack of artefact-free data for quantifying STA. For example, the largest STA quantification study in the upper extremity is limited to 4 subjects.

Our group has collected a dataset of humeral and scapular kinematics from 20 health subjects simultaneously imaged using a biplane fluoroscopy system and a 10-camera motion capture system (Vicon) at 100 Hz. This repository contains Python code for pre-processing and analyzing this dataset with the aim of quantifying STA.

