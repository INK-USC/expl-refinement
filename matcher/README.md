# Matcher

After collecting compositional explanations from human annotators, the *matcher* in our framework generalizes them in the target domain. This directory contains the matcher code (execute `main.py`), along with the lexicons for explanation parsing (see `data/`) and the compositional explanations we collected for 3 dataset pairs (see `annotation/`).

## Requirements

For chunking, NER, etc:

1. install stanfordcorenlp by pip install stanfordcorenlp
2. download CoreNLP 3.9.2: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip and unzip and put into qa-nle/stanfordnlp_resources/stanford-corenlp-full-2018-10-05

For string matching:

1. pip install transformers
2. Download bert base uncased into Model folder and unzip: https://github.com/google-research/bert#pre-trained-models

please access information about hateword and put the file in data/hatewords.txt

## Generate regularization results

strict-match:

```sh
python3 main.py --regularization \
--rule_path annotation_file_path --data_path data_path \
--label 0 --soft 0 \
--nproc 30 --ngpu 1 \
--advice_path advice_path
```

soft-match:

```sh
python3 main.py --regularization \
--rule_path annotation_file_path --data_path data_path \
--label 0 --soft 1 --thres 0.8 \
--nproc 10 --ngpu 3 \
--advice_path advice_path
```

`regularization`: flag to generate regularization results

`rule_path`: annotation file path

`data_path`: dataset path, directory containing `.pkl` file

`label`: if use ground truth label to filter matching results, currently should use `0`

`advice_path`: file path to save all matching results

`advice_without_inter_path`: file path to save matching results without interaction

`thres`: threshold to filter the matching results

`nproc`: number of processors

`ngpu`: number of GPUs

## Token transformation

To transform token from `allennlp` to `bert`:

```sh
python transform_advice.py --path advice_path --labeled
```

To transform token from `allennlp` to `roberta`:

```sh
python transform_advice.py --path advice_path --labeled roberta
```

