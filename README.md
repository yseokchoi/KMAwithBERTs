# Korean Morphological Analyzer using two BERTs



Korean Morphological Analyzer(KMA) using two Korean BERT models.
Our code was written based on huggingface and OpenNMT-py codes.
huggingface homepage: https://github.com/huggingface/transformers
OpenNMT-py homepage: https://github.com/OpenNMT/OpenNMT-py

## BERT models

- To get the BERT models from the site(https://aiopen.etri.re.kr/service_dataset.php).



## KMA Overview

- KMA is designed to identify functional morphemes from a word, which can specify the syntactic role of words in a sentence.

![model2-1](./figures/Example1-1.jpg)

**Figure 1**. The output of the KMA for the example input sentence "***(I) lost a black galaxy note***"

![model2-1](./figures/model2-1.jpg)

**Figure 2**. the basis architecture of a KMA which is initialized by *w*BERT and *m*BERT



## Train

```
python train.py
```



## Evaluation

```
python translate.py
```

