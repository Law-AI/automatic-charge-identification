import json
import spacy
from itertools import groupby
from tqdm import tqdm

import numpy as np

parser = spacy.load("en_core_web_sm")

def tokenize_text(text):
	parsed_text = parser(text)
	cleaned = [tok.lemma_.lower() if tok.ent_type == 0 else '[' + tok.ent_type_ + ']'\
		for tok in parsed_text if not any([tok.is_punct, tok.is_stop, tok.is_digit, len(tok) == 0])]
	cleaned = [group[0] for group in groupby(cleaned)]
	return cleaned

def build_dataset_from_jsonl(data_file):
	data = []
	with open(data_file) as fr:
		for line in tqdm(fr):
			doc = json.loads(line)
			
			doc['text'] = list(map(lambda x: tokenize_text(x), doc['text']))
			data.append(doc)
	return data

def calc_frequencies(data, word_freq, sent_label_freq=None, doc_label_freq=None):
	for doc in data:
		for i in range(len(doc['text'])):
			for word in doc['text'][i]: word_freq[word] += 1
			if 'sent_labels' in doc:
				for label in doc['sent_labels'][i]: sent_label_freq[label] += 1
		if 'doc_labels' in doc:
			for label in doc['doc_labels']: doc_label_freq[label] += 1

def create_vocab(word_freq, threshold=2, pretrained_vocab=None):
	words = set(w for w, f in word_freq.items() if pretrained_vocab and w in pretrained_vocab or f > threshold)
	word_vocab = {w: i + 2 for i, w in enumerate(words)}
	word_vocab['[PAD]'] = 0
	word_vocab['[UNK]'] = 1
	return word_vocab

def create_label_vocab(label_data):
	label_vocab = {}
	for lab in label_data:
		label_vocab[lab['chargeid']] = len(label_vocab)
	return label_vocab

def create_ptemb_matrix(word_vocab, pretrained, embedding_dim=128):
	ptemb_matrix = np.zeros((len(word_vocab), embedding_dim))
	for word, idx in word_vocab.items():
		if word == '[PAD]': continue
		ptemb_matrix[idx] = pretrained[word] if word in pretrained.key_to_index else np.random.normal(scale=0.6, size=(embedding_dim,))
	return ptemb_matrix

def numericalize_dataset(data, word_vocab, label_vocab):
	def numericalize_text(tokens):
		return [word_vocab[tok] if tok in word_vocab else word_vocab['[UNK]'] for tok in tokens]
	def vectorize_labels(labels):
		vector = [0] * len(label_vocab)
		for lab in labels:
			if lab in label_vocab: vector[label_vocab[lab]] = 1
		return vector

	for doc in data:
		doc['text'] = list(map(lambda sent: numericalize_text(sent), doc['text']))
		if 'sent_labels' in doc:
			doc['sent_labels'] = list(map(lambda slabs: vectorize_labels(slabs), doc['sent_labels']))
		if 'doc_labels' in doc:
			doc['doc_labels'] = vectorize_labels(doc['doc_labels'])

def calc_label_weights(label_vocab, label_freq, num_instances):
	label_wts = np.zeros((len(label_vocab),))
	for lab, freq in label_freq.items():
		if lab in label_vocab:
			label_wts[label_vocab[lab]] = num_instances / freq
	return label_wts