# Shadow Detection and Removal

BDRAR can detect shadows, while DHNet can remove (and detect) shadows. Here are two colabs chowcasing each work

BDRAR:
[[colab]](https://colab.research.google.com/drive/1wrIWT3gG7pAWwbBbdI2MBPvZvWoQPT4G#scrollTo=7zSUxMHio8By)
[[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhu_Bi-directional_Feature_Pyramid_ECCV_2018_paper.pdf)
[[original github]](https://github.com/zijundeng/BDRAR)

Dual Hierarchical Aggregation Network:
[[DHNet colab]](https://colab.research.google.com/drive/1cJ_dsBUXFaFtjoZB9gDYeahjmysnvnTq#scrollTo=vKTxRib8UOSN)
[[paper]](https://arxiv.org/abs/1911.08718)
[[original github]](https://github.com/vinthony/ghost-free-shadow-removal)

## Citations
```
@inproceedings{zhu18b,   
    author = {Zhu, Lei and Deng, Zijun and Hu, Xiaowei and Fu, Chi-Wing and Xu, Xuemiao and Qin, Jing and Heng, Pheng-Ann},    
    title = {Bidirectional Feature Pyramid Network with Recurrent Attention Residual Modules for Shadow Detection},    
    booktitle = {ECCV},    
    year  = {2018}    
}
```
```
@misc{cun2019ghostfree,
    title={Towards Ghost-free Shadow Removal via Dual Hierarchical Aggregation Network and Shadow Matting GAN},
    author={Xiaodong Cun and Chi-Man Pun and Cheng Shi},
    year={2019},
    eprint={1911.08718},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Original BDRAR Readme

## Shadow Maps
The results of shadow detection on SBU and UCF can be found at [Google Drive](https://drive.google.com/open?id=1Fhg5iuB2MSBtzklliXU65EwNbakfCkC1).

## Trained Model
You can download the trained model which is reported in our paper at [Google Drive](https://drive.google.com/open?id=1Cw3nUmWEmnTnAVXPn3xZYhQ_uzmWmsr7).

## Requirement
* Python 2.7
* PyTorch 0.4.0
* torchvision
* numpy
* Cython
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Preparation
1. Set the path of pretrained ResNeXt model in resnext/config.py
2. Set the path of [SBU](https://www3.cs.stonybrook.edu/~cvl/dataset.html) dataset in config.py

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

## Usage

### Training
1. Run by ```python train.py```

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently 
change it as you need.

Training a model on a single GTX 1080Ti GPU takes about 40 minutes.

### Testing
1. Put the trained model in ckpt/BDRAR
2. Run by ```python infer.py```
