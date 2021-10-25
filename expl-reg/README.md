# Learning with Explanation-Generalized Data

This directory contains the code to train the source model or train with refinement advice (execute `run_model.py`), and also the code we used to generate post-hoc explanation heat-maps (execute `constituency_tree.py`).

## Requirements

```sh
conda create -n expl-reg python==3.7.4
conda activate expl-reg
# modify CUDA version as yours
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
# requirements from pytorch-transformers
pip install tokenizers==0.0.11 boto3 filelock requests tqdm regex sentencepiece sacremoses
```


## Running experiments

### Training source models

```sh
python run_model.py --do_train --do_lower_case \
--data_dir data/hateval --task_name hateval \
--base_model bert --bert_model bert-base-uncased \
--max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --early_stop 5 \
--output_dir runs/hateval_vanilla_seed_0 --seed 0 --negative_weight 0.1
```

### Extracting rules from human explanations

Please refer to `matcher/README.md`.

### Refining neural network models

Without noisy labels:

```sh
python run_model.py --do_train --do_lower_case --no_label \
--train_advice  advice_path \
--dev_advice dev_advice_path \
--data_dir data/gab --task_name gab \
--base_model bert --bert_model runs/hateval_vanilla_seed_0 \
--max_seq_length 128 --train_batch_size 32 \
--learning_rate 5e-6 --num_train_epochs 20 \
--early_stop 5 --early_stop_iter 5 --output_dir runs/rs0.01_lr5e-6_seed0 \
--seed 0 --reg_explanations --nb_range 0 --sample_n 1 --negative_weight 0.1 \
--reg_strength 0.01
```

With noisy labels:

```sh
python run_model.py --do_train --do_lower_case \
--train_advice  advice_path \
--dev_advice dev_advice_path \
--data_dir data/gab --task_name gab \
--base_model bert --bert_model runs/hateval_vanilla_seed_0 \
--max_seq_length 128 --train_batch_size 32 \
--learning_rate 5e-6 --num_train_epochs 20 \
--early_stop 5 --early_stop_iter 20 --output_dir runs/rs0.01_lr5e-6_seed0 \
--seed 0 --reg_explanations --nb_range 0 --sample_n 1 --negative_weight 0.1 \
--reg_strength 0.01 --mix_label_ver2
```

### Evaluation

- Target domain dataset

  ```sh
  python run_model.py --do_eval --do_lower_case \
  --data_dir data/gab --task_name gab \
  --base_model bert --bert_model runs/rs0.01_lr5e-6_seed0  --max_seq_length 128 \
  --output_dir runs/rs0.01_lr5e-6_seed0 \
  --seed 0 --test
  ```

- IPTTS

  ```sh
  python run_model.py --do_eval --do_lower_case \
  --data_dir data/bias_madlibs_77k --task_name iptt \
  --base_model bert --bert_model runs/rs0.01_lr5e-6_seed0  --max_seq_length 128 \
  --output_dir runs/rs0.01_lr5e-6_seed0 \
  --seed 0 --test
  ```

