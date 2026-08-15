# <p align="center">FLEX: A Largescale Multimodal, Multiview Dataset for Learning Structured Representations for Fitness Action Quality Assessment</p>

### <p align="center">*Hao Yin, Lijun Gu, Paritosh Parmar, Lin Xu, Tianxiao Guo, Weiwei Fu, Yang Zhang, Tianyou Zheng*</p>

<p align="center">
  <a href="https://haoyin116.github.io/FLEX_Dataset/"><img src="https://img.shields.io/badge/Project-Page-8A2BE2" alt="Project Page"></a>
  <!-- <a href="https://badges.toozhao.com/stats/01JVNNN837B0VMFVDGT55N9NR6" title="Get your own page views count badge on badges.toozhao.com"><img src="https://badges.toozhao.com/badges/01JVNNN837B0VMFVDGT55N9NR6/blue.svg" alt="Page Views Count"></a> -->
</p>

### Abstract
Action Quality Assessment (AQA)—the task of quantifying how well an action is performed—has great potential for detecting errors in gym weight training, where accurate feedback is critical to prevent injuries and maximize gains. Existing AQA datasets, however, are limited to single-view competitive sports and RGB video, lacking multimodal signals and professional assessment of fitness actions. We introduce FLEX, the first large-scale, multimodal, multiview dataset for fitness AQA that incorporates surface electromyography (sEMG). FLEX contains over 7,500 multi-view recordings of 20 weight-loaded exercises performed by 38 subjects of diverse skill levels, with synchronized RGB video, 3D pose, sEMG, and physiological signals. Expert annotations are organized into a Fitness Knowledge Graph (FKG) linking actions, key steps, error types, and feedback, supporting a compositional scoring function for interpretable quality assessment. FLEX enables multimodal fusion, cross-modal prediction—including the novel Video→EMG task—and biomechanically oriented representation learning. Building on the FKG, we further introduce FLEX-VideoQA, a structured question–answering benchmark with hierarchical queries that drive cross-modal reasoning in vision–language models. Baseline experiments demonstrate that multimodal inputs, multi-view video, and fine-grained annotations significantly enhance AQA performance. FLEX thus advances AQA toward richer multimodal settings and provides a foundation for AI-powered fitness assessment and coaching.

### Key Words
action quality assessment, fitness, action understanding, video understanding

### Dataset Access Procedure
```
This dataset is not publicly downloadable. To request access:
1. Complete the dataset request form linked below.
2. Confirm that you agree to the terms of use (academic purposes only, no commercial exploitation).
3. After your request is reviewed and approved, you will receive instructions to access the dataset.
```
[➡️ Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdVWgFO3XSlkvKnaLIEaCkedJ-QDb4wclVgE2LhLQ4nGPBmRQ/viewform?usp=dialog)


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

To evaluate the models, run:

```eval
bash ./scripts/test.sh 0 Seven try --Seven_cls 1
```

Please feel free to reach out to me if you have any questions or face any problems.

#### Citation

If you find our work useful, please consider citing:
```
@article{yin2025flex,
      title={FLEX: A Largescale Multimodal, Multiview Dataset for Learning Structured Representations for Fitness Action Quality Assessment}, 
      author={Hao Yin, Lijun Gu, Paritosh Parmar, Lin Xu, Tianxiao Guo, Weiwei Fu, Yang Zhang, Tianyou Zheng},
      journal={arXiv preprint arXiv:2506.03198},
      year={2025},
      }
```

#### Contributing

Our code is based on [CoRe](https://github.com/yuxumin/CoRe). Thanks for their great work!
