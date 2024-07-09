# ADSNet: Adaptation of Distinct Semantic for Uncertain Areas in Polyp Segmentation, BMVC 2023 

Official Pytorch implementation of [ADSNet: Adaptation of Distinct Semantic for Uncertain Areas in Polyp Segmentation](https://arxiv.org/pdf/2405.07523). 

In The 34th British Machine Vision Conference, 20th - 24th November 2023, Aberdeen, UK.

## Tensorflow
Official Tensorflow implementation is available at [ADSNet-BMVC2023-Tensorflow](https://github.com/vinhhust2806/ADSNet-BMVC2023-Tensorflow)

## 1. Architecture

<p align="center">
<img src="architecture.png" width=75% height=40% 
class="center">
</p>

## 2. Abstract
Colonoscopy is a common and practical method for detecting and treating polyps.
Segmenting polyps from colonoscopy image is useful for diagnosis and surgery progress.
Nevertheless, achieving excellent segmentation performance is still difficult because of
polyp characteristics like shape, color, condition, and obvious non-distinction from the
surrounding context. This work presents a new novel architecture namely Adaptation of
Distinct Semantics for Uncertain Areas in Polyp Segmentation (ADSNet), which modifies misclassified details and recovers weak features having the ability to vanish and
not be detected at the final stage. The architecture consists of a complementary trilateral decoder to produce an early global map. A continuous attention module modifies
semantics of high-level features to analyze two separate semantics of the early global
map. The suggested method is experienced on polyp benchmarks in learning ability and
generalization ability, experimental results demonstrate the great correction and recovery
ability leading to better segmentation performance compared to the other state of the art
in the polyp image segmentation task. Especially, the proposed architecture could be
experimented flexibly for other CNN-based encoders, Transformer-based encoders, and
decoder backbones.

## 3. Usage
### Recommended environment:
Please use ```pip install -r requirements.txt``` to install the libraries.

### Data preparation:
Download the training and testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them into 'polyp/' folder.

### Training:
Run ```python main.py```

## 4. Polyp Segmentation Compared Results
We also provide some result of baseline methods, You could download from [Google Drive](https://drive.google.com/file/d/1xvjRl70pZbOO6wI5p94CSpZK2RAUnUnx/view?usp=sharing), including results of compared models.

## 5. Citation
If you have found our work useful, please use the following reference to cite this project:
```
@article{nguyen2024adaptation,
  title={Adaptation of Distinct Semantics for Uncertain Areas in Polyp Segmentation},
  author={Nguyen, Quang Vinh and Huynh, Van Thong and Kim, Soo-Hyung},
  journal={arXiv preprint arXiv:2405.07523},
  year={2024}
}

@inproceedings{Nguyen_2023_BMVC,
author    = {Quang Vinh Nguyen and Van Thong Huynh and Soo-Hyung Kim},
title     = {Adaptation of Distinct Semantics for Uncertain Areas in Polyp Segmentation},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0806.pdf}
}
```
