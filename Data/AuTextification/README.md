# AuTextification Dataset

### Links
- [GitHub](https://github.com/autextification/AuTexTification-Overview/tree/main)
- [Website](https://sites.google.com/view/autextification/home)
- [Dataset](https://huggingface.co/datasets/symanto/autextification2023)
- [Arxiv](https://arxiv.org/abs/2309.11285)

### Description

The AuTextification dataset involves two tasks: Machine-Generated LLM Detection and Model Attribution
in both english and spanish, respectively called sub-task 1 and sub-task 2.
Model attribution involves determining which LLM the text arose from.
We only consider the english version in this project. Sub-task 1 is a binary classification
problem and Sub-task 2 is a multiclass classification problem with the following LLM's used: LOOM-1B1; BLOOM-3B; BLOOM-7B1;
Babbage; Curie; and text-davinci-003. 

Column ids for MGT Detection (subtask one) are `id`, `prompt`, `text`, `label`, `model` and `domain`.
Column ids for Model Attribution (subtask two) are `id`, `prompt`, `text`, `label`, and `domain`.



