# forocoches-language-generation
## Description
This is a PyTorch implementation of a decoder only transformer inspired on GPT-2. 

The model was trained from scratch on a custom dataset of over 1 million threads from the Spanish forum [ForoCoches](https://www.forocoches.com/). **At the moment, this dataset is not publicly available.** An article will come with it.

***WARNING***: The dataset used to train the model contains bad language and offensive content. Therefore, it is likely that the model will generate offensive predictions. **Use it at your own risk.**


## Prerequisites
* PyTorch
* Numpy
* Transformers
* tqdm (for fancy training)


## Installation
### 1. Clone the repo

```
git clone https://github.com/Atenrev/forocoches-language-generation.git
cd forocoches-language-generation
```

### 2. Data
The dataset used to train the provided model consists of raw threads scraped from the whole website. It has been processed, formatted and lowercased. It is not publicly available for now.

However, you can use your own dataset by modifying the ```dataset.py``` script.

### 3. Download the model
* Download the pre-trained model from here: https://drive.google.com/file/d/1a9_5BhS-vP4SX4Wj2LxXcqb2YR2MKKGg/view?usp=sharing. Place it in the ```models``` folder.
* Download the pre-trained tokenizer from here: https://drive.google.com/file/d/1rorJMCcP72FsgR5w3fl6wNunhsKG4IQY/view?usp=sharing. Extract it into the ```models``` folder.

Here is an example of what the root should look like:
```
.
├── dataset.py
├── model.py
├── inference.py
├── trainer.py
├── dataset\
└── models\
    ├── fctokenizer-small
    └── fcgen_epoch_3_v.pth
```

## Train
Run ```trainer.py```:

``` sh
python trainer.py 

optional arguments:
  -h, --help                            show this help message and exit
  --dataset DATASET                     location of the dataset
  --tokenizer_config TOKENIZER_CONFIG   location of the vocabulary
  --batch_size BATCH_SIZE               training batch size
  --epochs EPOCHS                       number of training epochs
  --lr LR                               learning rate
  --beta1 BETA1                         adam beta
  --save_dir SAVE_DIR                   save model directory
  --load_model LOAD_MODEL               model to load and resume trainin
```

## Inference
Run ```inference.py --prompt "<|THREAD|>"```:

``` sh
python .\inference.py

optional arguments:
  -h, --help                            show this help message and exit
  --load_model LOAD_MODEL               model to load and do inference
  --prompt PROMPT                       prompt for the model
  --tokenizer_config TOKENIZER_CONFIG   location of the tokenizer config file
```

The script will print the results in the console. It will also save them in a ```prediction.txt``` file as well.

Note that there exist several tags that you can use when prompting the model. Usually, the model will work best when you input the *<|THREAD|>* tag at the beginning and follow the thread structure.

### Example
Output prediction from executing ```python inference.py --prompt "<|THREAD|>"```:

```
donde se puede ver el éxito de pablo casado?

se supone que en la sexta han entrevistado al líder de vox pero no ha estado en el debate. lo echan en el canal para ver la hipocresía.

--- COMMENT ---
https://static2.elcomercio.es/www/multimedia/201904/27/media/cortadas/casado-iglesias-khgf-u701106542958jfd-624x385@el%20comercio.jpg

--- COMMENT ---
cita: cita de .onán.
showthread.php?p=331273493#post331273493
https://static2.elcomercio.es/www/multimedia/201904/27/media/cortadas/casado-iglesias-khgf-u701106542958jfd-624x385@el%20comercio.jpg
lo que ha hecho el resto, es el resultado.
```
