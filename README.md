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

