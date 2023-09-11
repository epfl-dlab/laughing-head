# Laughing Heads

This repository contains code related to the paper: 

Laughing Heads: Can Transformers Detect What Makes a Sentence Funny?
Maxime Peyrard, Beatriz Borges, Kristina Gligoric ́ and Robert West

## Abstract

The automatic detection of humor poses a grand challenge for natural language processing. Transformer-based systems have recently achieved remarkable results on this task, but they usually (1) were evaluated in setups where serious vs. humorous texts came from entirely different sources, and (2) focused on benchmarking performance without providing insights into how the models work. We make progress in both respects by training and analyzing transformer-based humor recognition models on a recently introduced dataset consisting of minimal pairs of aligned sentences, one serious, the other humorous. We find that, although our aligned dataset is much harder than previous datasets, transformer-based models recognize the humorous sentence in an aligned pair with high accuracy (78%). In a careful error analysis, we characterize easy vs. hard instances. Finally, by analyzing attention weights, we obtain important insights into the mechanisms by which transformers recognize humor. Most remarkably, we find clear evidence that one single attention head learns to recognize the words that make a test sentence humorous, even without access to this information at training time.

## Data

It is possible to access both the raw unfun data and the pre-processed data used in the paper: 

- Raw Unfun (keeps on growing as players continue to contribute new data points): https://github.com/epfl-dlab/unfun 
- Processed data for this paper available [here](https://drive.google.com/drive/folders/1vXYa6al1N-413ih2S9QYFyEG7MVjl6x7?usp=sharing)
