# <p align="center">FLEX: A Large-Scale Multi-Modal Multi-Action Dataset for Fitness Action Quality Assessment</p>

### <p align="center">*Hao Yin, Lijun Gu, Paritosh Parmar, Lin Xu, Tianxiao Guo, Weiwei Fu, Yang Zhang, Tianyou Zheng*</p>

<p align="center">
  <a href="https://haoyin116.github.io/FLEX_Dataset/"><img src="https://img.shields.io/badge/Project-Page-8A2BE2" alt="Project Page"></a>
  <a href="https://badges.toozhao.com/stats/01JVNNN837B0VMFVDGT55N9NR6" title="Get your own page views count badge on badges.toozhao.com"><img src="https://badges.toozhao.com/badges/01JVNNN837B0VMFVDGT55N9NR6/blue.svg" alt="Page Views Count"></a>
</p>

### Abstract
With the increasing awareness of health and the growing desire for aesthetic physique, fitness has become a prevailing trend. However, the potential risks associated with fitness training, especially with weight-loaded fitness actions, cannot be overlooked. Action Quality Assessment (AQA), a technology that quantifies the quality of human action and provides feedback, holds the potential to assist fitness enthusiasts of varying skill levels in achieving better training outcomes. Nevertheless, current AQA methodologies and datasets are limited to single-view competitive sports scenarios and RGB modality and lack professional assessment and guidance of fitness actions. To address this gap, we propose the FLEX dataset, the first multi-modal, multi-action, large-scale dataset that incorporates surface electromyography (sEMG) signals into AQA. FLEX utilizes high-precision MoCap to collect 20 different weight-loaded actions performed by 38 subjects across 3 different skill levels, containing 5 different views of the RGB video, 3D pose, sEMG, and physiological information. Additionally, FLEX incorporates knowledge graphs into AQA, constructing annotation rules in the form of penalty functions that map weight-loaded actions, action keysteps, error types, and feedback. We conducted various baseline methodologies on FLEX, demonstrating that multimodal data, multiview data, and fine-grained annotations significantly enhance model performance. FLEX not only advances AQA methodologies and datasets towards multi-modal and multi-action scenarios but also fosters the integration of artificial intelligence within the fitness domain.

### Key Words
action quality assessment, fitness, action understanding, video understanding

### Experiment Code

#### Requirements

To install requirements:

```setup
conda env create -f environment.yml
```
#### Training

To train the model(s) in the paper, run this command:

```train
bash ./scripts/train.sh 0 Seven try --Seven_cls 1
```

>📋 In FLEX, the Seven_cls can be in the range of 1 to 20. 
#### Evaluation

To evaluate my model on ImageNet, run:

```eval
bash ./scripts/test.sh 0 Seven try --Seven_cls 1
```
#### Contributing

Our code is based on [CoRe](https://github.com/yuxumin/CoRe). Thanks for their great work!
