# Manual

## Dependencies

```
pip install -r requirements
pip install -r luna/requirements
pip install git+https://github.com/allenai/allennlp.git@bdb29a831ed68cb948b18b42fa61646b9ec11bf8
```

## Usage

Here are some important args for the program:

- `mode` specifies the running mode of the program:  `train` / `attack` / `peval`

- `task_id` specifies the data set to use: `IMDB` / `AGNEWS` / `SNLI`

- `arch` specifies the model architecture: `boe` / `cnn` / `lstm` / `bert`

- `embed` specifies whether to use dirchlet neighbour ensemble. By default the value is null, if set to `w` , the DNE will be used.

- `adv_iter` specifies the number of iterations of adversarial training.

- `adv_policy` specifies which kind of policy to use for adversarial training: `hot` / `rdm` / `diy`. `hot` means generating adversarial examples with HotFlip, `rdm` means generating examples by randomly replacing a word with its synonyms. Both of them denote a word-level subsititution, which does adversarial training in a discrete manner. `diy` means we do not change the tokens of a sentence, the model will generate adversarial examples in a contigious manner, e.g., the DNE. Examples may be found in the later text.

- `attack_method` specifies which adversary to use: `pwws` / `genetic` / `genetic_nolm` .

## Training & Evaluate

### Normal Training

Suppose you want to train a CNN model on the IMDB dataset, you shall run the command below:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=0
```

Then you can run an attacker (e.g., PWWS) on 500 samples on the develop set by the command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=0 --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True
```

### Adversarial Training

Suppose you want to train a CNN model on the IMDB dataset with the adversarial loss, you shall run the command below:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=hot
```

Then you can run an attacker (e.g., PWWS) on 500 samples on the develop set by the command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=hot --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True
```

### Training by Random Replacement

Suppose you want to train a CNN model on the IMDB dataset with a word randomly replaced by its synonyms, you shall run the command below:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99
```

Then you can run an attacker (e.g., PWWS) on 500 samples on the develop set by the command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99 --pred_transform=embed_aug --pred_transform_args=0.99 --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16
```

### Training with DNE

Suppose you want to train a CNN model on the IMDB dataset with DNE, you shall run the command below:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed=w --adv_iter=3 --adv_policy=diy
```

Then you can run an attacker (e.g., PWWS) on 500 samples on the develop set by the command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed=w --adv_iter=3 --adv_policy=diy --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16
```