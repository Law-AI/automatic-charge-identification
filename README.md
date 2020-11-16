# Automatic Charge Identification in Indian Legal Documents
Identifying charges from the Indian Penal Code given the textual description of the charges and facts of a criminal case.

## Introduction
This is the repository for the paper titled "Automatic Crime Identification from Facts: A Few Sentence-Level Crime Annotations is All You Need" which is to be presented at <a href="https://coling2020.org/">The 28th International Conference on Computational Linguistics, 2020</a>.

Identifying the relevant charges given the fact descriptions of a legal scenario and the statutory laws defining charges is one of the most important tasks in the judicial process of countries following Civil Law System. This task is challenging, since the statutory laws are usually written in formal and abstract language to encapsulate wide-ranging scenarios. Meanwhile, the fact descriptions can be informal, and can contain a lot of text (like background information) that do not indicate any crime, but are included for the sake of informativeness and completion. Additionally, more than one charge may be relevant, and the frequency distribution of charges is usually highly skewed (long-tail distribution). 

We annotate a small set of fact descriptions with sentence-level charges, i.e., for every sentence in the fact description, we annotate the charges which may be relevant given that sentence alone. We use a model that treats text (fact and charge descriptions alike) as a hierarchy of sentences and words, and constructs intermediate sentence embeddings for each sentence as well as a document embedding for the entire text. We use multi-task learning to optimize both sentence and document-level losses simultaneously.

We make available:

(1) A dataset containing: (a) Charge descriptions of 20 charges (topics in the Indian Penal Code, 1860); (b) A training set consisting of 120 fact descriptions with relevant sentence and document-level charge labels; (c) A test set consisting of 70 fact descriptions with relevant document-level charge labels only.

(2) The implementation of our proposed approach

## Citation

## Requirements

## Codes

## Dataset
### Charge Descriptions
The file "Labels.jsonl" contains the charge descriptions. Each line should contain a JSON string which represents the charge description. Each charge description is a Python Dict with the following keys:
```
  chargeid: str -> Charge ID
  text: List[str] -> List of sentences
```

### Fact Descriptions
The file "Train-Sent.jsonl" and "Test-Doc.jsonl" are fact description datasets. Each line should contain a JSON string which represents the fact description. Each fact description is a Python Dict with the following keys:
```
  factid: str -> Fact ID
  text: List[str] -> List of sentences
  sent_labels: List[List[str]] -> List of List of chargeid, each sublist is the sent-level charge; Optional; not needed for inference or vanilla single-task training
  doc_labels: List[str] -> List of chargeid, entire document-level charges; Optional, not needed for inference
```
## Training
### Input Data
Setup the Charge and Fact Description files as mentioned above. 'sent_labels' are compulsory for multi-task learning, not required for single-task learning.

### Usage

### Output Data

## Inference
### Input Data
Setup the Charge and Fact Description files as mentioned above. 'sent_labels' and 'doc_labels' are not compulsory.

### Usage

### Output Data
