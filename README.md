# Refining Language Models with Compositional Explanations

This repo contains the code for our paper, [Refining Language Models with Compositional Explanations](https://arxiv.org/pdf/2103.10415.pdf).

**Re**fining Language **M**odel with Compositional **E**xplanation (REMOTE) is a framework that collects compositional explanations from human and refines a trained model based on the explanations.

![overview-image](https://i.imgur.com/Rg8Ppwn.png)

We aim to adapt a text classification model trained on some *source* data to a new *target* domain.

As shown in the above illustration, we first show the heat-maps of a trained model to human annotators, and collect compositional explanations from them. The explanations are given in natural language, in the form of "why the model is doing wrong" and "advice on how the model should be adjusted."

We *match* the given explanations to a broader set of training examples in the target domain, and adjust the model to align with the human-specified advice via *regularization*.

## Repository Structure

The repository contains two components:

- `matcher`: parses the collected natural-language explanations into executable logic rules and matches to other instances in the dataset
- `expl-reg`: refines the trained model with the noisy labels and refinement advice produced by the matcher

Each directory contains README for instructions to run the code.
