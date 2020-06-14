# Manual

## Dependencies

```
pip install -r requirements
pip install -r luna/requirements
pip install git+https://github.com/allenai/allennlp.git@bdb29a831ed68cb948b18b42fa61646b9ec11bf8
```

You can download the pre-processed data set from `https://drive.google.com/file/d/1xMqj3cHN7B6bgfnM6rZg6N6eZPPJ_TGy/view?usp=sharing`, and unzip it into `datasets/`.

## Usage

Here are some args for the program:

- `mode` specifies a running mode of the program:  `train` / `attack` / `peval`

- `task_id` specifies which data set is used: `IMDB` / `AGNEWS` / `SNLI`

- `arch` specifies the model architecture: `boe` / `cnn` / `lstm` / `bert`

- `embed` specifies whether to use the Dirchlet Neighbour Ensemble (DNE). This value is null by default. If it is set to `w`, the DNE will be used.

- `adv_iter` specifies the maximum number of iterations of adversarial training.

- `adv_policy` specifies which method is used to adversarial examples: `hot` / `rdm` / `diy`. `hot`: to generate the adversarial examples by HotFlip, `rdm`: to generate adversarial examples by randomly replacing one or more word in an input sentence with their synonyms. Both of `hot` and `rdm` use the discrete word substitution-based perturbations. `diy` to generate virtual adversarial examples as DNE. 

- `attack_method` specifies which attack algorithm is used: `pwws` / `genetic` / `genetic_nolm` .

## Training & Evaluate

### Normal Training

If you want to train a CNN model on the IMDB dataset, you can run the following command:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=0
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=0 --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True
```

### Adversarial Training

If you want to train a CNN model on the IMDB dataset using the adversarial loss, you can run the command below:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=hot
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=hot --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True
```

### Training by our proposed baseline RAN

The models trained by RAN will take as inputs the corrupted copy of each input sentence, in which every word of the sentence is randomly replaced with one of its synonyms. In the inference time, the same random replacement is used and the prediction scores are ensembled to get an output.

If you want to train a CNN model on the IMDB dataset by RAN, you can run the following command:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed='' --adv_iter=3 --adv_policy=rdm --adv_replace_num=0.99 --pred_transform=embed_aug --pred_transform_args=0.99 --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16
```

### Training with DNE

If you want to train a CNN model on the IMDB dataset with DNE, you can run the following command:

```
python play_with_glue.py --mode=train --task_id=IMDB --arch=cnn --embed=w --adv_iter=3 --adv_policy=diy
```

Then, you can run an attacking algorithm (e.g., PWWS) on 500 samples randomly sampled from the develop set by the following command:

```
python play_with_glue.py --mode=attack --task_id=IMDB --arch=cnn --embed=w --adv_iter=3 --adv_policy=diy --attack_method=pwws --data_split=dev --data_downsample=500 --data_random=True --pred_ensemble=16
```

