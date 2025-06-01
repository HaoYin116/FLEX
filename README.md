# FLEX AQA Experiments Code
[![Project Page](https://img.shields.io/badge/Project-Page-8A2BE2)](https://haoyin116.github.io/FLEX_Dataset/)
[![Page Views Count](https://badges.toozhao.com/badges/01JVNNN837B0VMFVDGT55N9NR6/blue.svg)](https://badges.toozhao.com/stats/01JVNNN837B0VMFVDGT55N9NR6 "Get your own page views count badge on badges.toozhao.com")

This repository is the official implementation of FLEX Dataset Baselines.

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Training

To train the model(s) in the paper, run this command:

```train
bash ./scripts/train.sh 0 Seven try --Seven_cls 1
```

>📋 In FLEX, the Seven_cls can be in the range of 1 to 20. 

## Evaluation

To evaluate my model on ImageNet, run:

```eval
bash ./scripts/test.sh 0 Seven try --Seven_cls 1
```

## Contributing

>📋  Our code is based on [CoRe](https://github.com/yuxumin/CoRe). Thanks for their great work!
