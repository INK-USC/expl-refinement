# Refining Language Models with Compositional Explanations

Code for [this arxiv article](https://arxiv.org/pdf/2103.10415.pdf).

We introduce **Re**fining Language **M**odel with Compositional **E**xplanation (REMOTE), a framework that refines a trained model by collecting compositional explanations from human and refining the model with broadened coverage during regularization.

The repository contains two components:

- `matcher`: parses the collected natural-language explanations into executable logic rules and matches to other instances in the dataset
- `expl-reg`: refines the trained model with the noisy labels and refinement advice produced by the matcher

Each directory contains README for instructions to run the code.
