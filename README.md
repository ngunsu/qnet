# Cross-Spectral Local Descriptors via Quadruplet Network

[PDF](http://www.mdpi.com:8080/1424-8220/17/4/873) 

Bibtex
```latex
@Article{s17040873,
AUTHOR = {Aguilera, Cristhian A. and Sappa, Angel D. and Aguilera, Cristhian and Toledo, Ricardo},
TITLE = {Cross-Spectral Local Descriptors via Quadruplet Network},
JOURNAL = {Sensors},
VOLUME = {17},
YEAR = {2017},
NUMBER = {4},
ARTICLE NUMBER = {873},
URL = {http://www.mdpi.com/1424-8220/17/4/873},
ISSN = {1424-8220},
DOI = {10.3390/s17040873}
}
```

## Instructions

First install the torch framework and cudnn

1. [Install torch](http://torch.ch/docs/getting-started.html#_)
2. [Cudnn torch](https://github.com/soumith/cudnn.torch)

### Datasets

#### Nirscenes patches

Two options: Download the generated t7 dataset from

- [Google Drive](https://drive.google.com/open?id=1ilDBegjTW_DH02FvuOyWo47ecjjNwP0T)

**Training and evaluation t7 files are different**

#### VIS-LWIR ICIP2015

1. Download the dataset from

- [Google Drive](https://drive.google.com/open?id=1lQ3pFy483J00PxXMohEwoc05bqQHltAh)


### Eval

#### Nirscenes eval (requires cuda, cudnn)

Evaluation code can be found in the *eval* folder. To eval one sequence:

1. You have to generate or download  the nirscenes patch dataset
2. Install xlua

```bash
luarocks install xlua
luarocks install moses
```

3. Run

```bash
cd eval
th nirscenes_eval.lua -dataset_path [path] -net [trained network]
```

For example, to eval the field sequence using the Q-Net article trained network. 

```bash
th nirscenes_eval.lua -dataset_path ../datasets/nirscenes/test -net ../trained_networks/qnet.t7
```

For more options, run 

```bash
th nirscenes_eval -h
```

#### VIS-LWIR eval (ICIP2015) (just cuda support)

1. You have to download the dataset first
2. Run

```bash
cd eval
th icip2015_eval.lua -dataset_path ../datasets/icip2015/ -net [trained network] 
```

For example. To eval Q-Net

```bash
cd eval
th icip2015_eval.lua -dataset_path ../datasets/icip2015/ -net ../trained_networks/qnet.t7 
```

### Training

1. Install penlight, torchx and json

```bash
luarocks install penlight
luarocks install torchx
luarocks install json
```

2. Train a network

 ```bash
 cd train
 th nirscenes_quadruplets_train.lua
 ```

Run

```bash
th nirscenes_quadruplets_train.lua -h
```

to see the options

*Note* The training code is different from the one used in the article. This new version is smaller. Additionally, the dataset was generated from zero. So, small differences in FPR95 may happen.


