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

- [Google Drive](https://drive.google.com/drive/folders/0B_tuqO61RC9hUVo1RHRHUGdGQU0?usp=sharing)

Follow the following steps to generate the dataset

1. Download the original nirscenes dataset [link](http://ivrl.epfl.ch/supplementary_material/cvpr11/)
2. Decompress the dataset on /datasets/nirscenes
3. Convert the images in each folder to ppm  (Torch doesn't support tiff). Use your favorite software
4. Install csvigo

    ```bash
    luarocks install csvigo
    ```

5. Use our script to generate the dataset

    ```bash
    cd utils
    th nirscenes_to_t7.lua
    ```


#### VIS-LWIR ICIP2015

1. Download the dataset from

- [Google Drive](https://drive.google.com/drive/folders/0B_tuqO61RC9hUDI1bmNqU1dKWGc?usp=sharing)


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
    th nirscenes_eval.lua -dataset_path [path] -net [trained network] -net_type qnet
    ```

For example, to eval the field sequence using the Q-Net article trained network. It is assumed that the dataset is stored in ../datasets  

    ```bash
    th nirscenes_eval.lua -dataset_path ../datasets -net ../trained_networks/qnet.t7 -net_type qnet
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
    th icip2015_eval.lua -dataset_path ../nirscenes/icip2015/ -net [trained network] -net_type qnet
    ```

For example. To eval Q-Net

    ```bash
    cd eval
    th icip2015_eval.lua -dataset_path ../datasets/icip2015/ -net ../trained_networks/qnet.t7 -net_type qnet
    ```

### Training

1. Install penlight

    ```bash
    luarocks install penlight
    ```

2. Train a network

    ```bash
    cd train
    th nirscenes_doall.lua -training_sequences [country|field|...] -net [2ch|siam|psiam]
    ```

For example, train a 2ch network using the country sequence

 ```bash
 cd train
 th nirscenes_doall.lua -training_sequences country -net 2ch
 ```

Results will be stored in the results folder.For more options, run

```bash
th nirscenes_doall.lua -h
```

*Note* The training code is different from the one used in the article. This new version is much faster. 


