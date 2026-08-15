# FLEX: A Large-scale Multimodal, Multiview Dataset for Fitness AQA

FLEX is a multimodal, multiview dataset and benchmark for fitness Action Quality Assessment (AQA). It contains more than 7,500 multi-view recordings covering 20 weight-loaded exercises, with synchronized RGB video, 3D pose, surface electromyography, physiological signals, and expert annotations organized through a Fitness Knowledge Graph.

## Repository structure

- [`code/`](code/) contains the training and evaluation code, experiment configurations, and detailed usage instructions.
- [`web/`](web/) contains the source for the [FLEX project website](https://haoyin116.github.io/FLEX_AQA_Dataset/).

## Quick start

```bash
cd code
conda env create -f environment.yml
bash ./scripts/train.sh 0 Seven try --Seven_cls 1
```

See [`code/README.md`](code/README.md) for dataset access, training, and evaluation details.

## Citation

```bibtex
@article{yin2025flex,
  title   = {FLEX: A Largescale Multimodal, Multiview Dataset for Learning Structured Representations for Fitness Action Quality Assessment},
  author  = {Hao Yin and Lijun Gu and Paritosh Parmar and Lin Xu and Tianxiao Guo and Weiwei Fu and Yang Zhang and Tianyou Zheng},
  journal = {arXiv preprint arXiv:2506.03198},
  year    = {2025}
}
```
