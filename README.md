# FLEX AQA Experments Code

This repository is the official implementation of FLEX Dataset Baseline.

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Training

To train the model(s) in the paper, run this command:

```train
bash ./scripts/trian.sh 0 Seven try --Seven_cls 1
```

>📋 In FLEX, the Seven_cls can be in the range of 1 to 20. 

## Evaluation

To evaluate my model on ImageNet, run:

```eval
bash ./scripts/test.sh 0 Seven try --Seven_cls 1
```

## Contributing

>📋  Our code is based on [CoRe](https://github.com/yuxumin/CoRe). Thanks for their great work!
