# Manual


This repository contains code for the paper:

```
@misc{zhou2020defense,
    title={Defense against Adversarial Attacks in NLP via Dirichlet Neighborhood Ensemble},
    author={Yi Zhou, Xiaoqing Zheng, Cho-Jui Hsieh, Kai-wei Chang and Xuanjing Huang},
    year={2020},
    eprint={2006.11627},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Dependencies

```
pip install -r requirements.txt
pip install -r luna/requirements.txt
```

You can download the pre-processed data set from `https://drive.google.com/file/d/1xMqj3cHN7B6bgfnM6rZg6N6eZPPJ_TGy/view?usp=sharing`, and unzip it into `datasets/`.

## Usage

Here are some args for the program:

- `mode` specifies a running mode of the program:  `train` / `attack` / `peval`

- `task_id` specifies which data set is used: `IMDB` / `AGNEWS` / `SNLI`

- `arch` specifies the model architecture: `boe` / `cnn` / `lstm` / `bert`

- `weighted_embed` specifies whether to use the Dirchlet Neighbour Ensemble (DNE).

- `adv_iter` specifies the maximum number of iterations of adversarial training.

- `adv_policy` specifies which method is used to adversarial examples: `hot` / `rdm` / `diy`. `hot`: to generate the adversarial examples by HotFlip, `rdm`: to generate adversarial examples by randomly replacing one or more word in an input sentence with their synonyms. Both of `hot` and `rdm` use the discrete word substitution-based perturbations. `diy` to generate virtual adversarial examples as DNE. 

- `attack_method` specifies which attack algorithm is used: `PWWS` / `GA-LM` / `GA` .

## Training & Evaluate

### Normal Training

If you want to train a CNN model on the IMDB dataset, you can run the following command:

```
python play.py --mode=train --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=0
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play.py --mode=attack --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=0 --attack_method=PWWS --data_split=dev --data_downsample=500 --data_random=True
```

### Adversarial Training

If you want to train a CNN model on the IMDB dataset using the adversarial loss, you can run the command below:

```
python play.py --mode=train --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=3 --adv_policy=hot
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play.py --mode=attack --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=3 --adv_policy=hot --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True
```

### Training by our proposed baseline RAN

The models trained by RAN will take as inputs the corrupted copy of each input sentence, in which every word of the sentence is randomly replaced with one of its synonyms. In the inference time, the same random replacement is used and the prediction scores are ensembled to get an output.

If you want to train a CNN model on the IMDB dataset by RAN, you can run the following command:

```
python play.py --mode=train --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play.py --mode=attack --task_id=IMDB --arch=cnn --weighted_embed=False --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99 --pred_transform=embed_aug --pred_transform_args=0.99 --attack_method=PWWS --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16
```

### Training with DNE

If you want to train a CNN model on the IMDB dataset with DNE, you can run the following command:

```
python play.py --mode=train --task_id=IMDB --arch=cnn --weighted_embed=True --adv_iter=3 --adv_policy=diy --dir_alpha=1.0 --dir_decay=0.5
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play.py --mode=attack --task_id=IMDB --arch=cnn --weighted_embed=True --adv_iter=3 --adv_policy=diy --attack_method=PWWS --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16 --dir_alpha=1.0 --dir_decay=0.5
```

