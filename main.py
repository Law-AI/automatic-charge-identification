from collections import defaultdict
from argparse import ArgumentParser

from gensim.models import KeyedVectors

from prepare_data import *
from model.model import *
from training import *

def main():
	parser = ArgumentParser()

	parser.add_argument("--data_path", default="data/", type=str, help="Folder to store dataset")
	parser.add_argument("--train_file", default="Train-Sent.jsonl", type=str, help="Train dataset")
	parser.add_argument("--test_file", default="Test-Doc.jsonl", type=str, help="Test dataset")
	parser.add_argument("--label_file", default="Labels.jsonl", type=str, help="Charge descriptions")
	parser.add_argument("--save_path", default="saved/", type=str, help="Folder to store trained model and metrics")
	
	parser.add_argument("--pretrained", default="ptembs/word2vec.kv", type=str, help="Pretrained word2vec embeddings file [embedding dimensions must match!], use 'None' for no pretrained initialization")
	parser.add_argument("--label_wts", default=True, type=bool, help="Use weighted loss function")

	parser.add_argument("--vocab_thresh", default=2, type=int, help="Min frequency for a word to be included in vocabulary")
	parser.add_argument("--embed_dim", default=128, type=int, help="Embedding dimension")
	parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs")
	parser.add_argument("--batch_size", default=5, type=int, help="Batch size")

	parser.add_argument("--device", default='cuda', type=str, help="Device (cuda/cpu)")

	parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
	parser.add_argument("--l2reg", default=5e-4, type=float, help="L2 Regularization penalty")

	parser.add_argument("--lr_patience", default=5, type=int, help="Number of epochs of non-increasing performance to wait before reducing learning rate, use -1 for fixed learning rate")
	parser.add_argument("--lr_factor", default=0.5, type=float, help="Factor to reduce learning rate by")

	parser.add_argument("--print_every", default=1, type=int, help="Epoch interval after which metrics will be printed")

	args = parser.parse_args()

	print("Loading and tokenizing fact descriptions...")
	traindev_data = build_dataset_from_jsonl(args.data_path + args.train_file)
	test_data = build_dataset_from_jsonl(args.data_path + args.test_file)
	print("Loading and tokenizing charge descriptions...")
	label_data = build_dataset_from_jsonl(args.data_path + args.label_file)

	num_docs = len(traindev_data)
	num_sents = len(sum([doc['text'] for doc in traindev_data], []))

	print("Creating vocab...")
	word_freq = defaultdict(int)
	sent_label_freq = defaultdict(int)
	doc_label_freq = defaultdict(int)

	calc_frequencies(traindev_data, word_freq, sent_label_freq, doc_label_freq)
	calc_frequencies(label_data, word_freq)

	label_vocab = create_label_vocab(label_data)

	if args.pretrained != 'None':
		pretrained = KeyedVectors.load(args.pretrained, mmap = 'r')
		vocab = create_vocab(word_freq, pretrained_vocab=pretrained.key_to_index)
		ptemb_matrix = create_ptemb_matrix(vocab, pretrained)
	else:
		vocab = create_vocab(word_freq)
		ptemb_matrix = None

	print("Numericalizing all data...")
	numericalize_dataset(traindev_data, vocab, label_vocab)
	numericalize_dataset(test_data, vocab, label_vocab)
	numericalize_dataset(label_data, vocab, label_vocab)

	if args.label_wts:
		sent_label_wts = torch.from_numpy(calc_label_weights(label_vocab, sent_label_freq, num_sents)).cuda()
		doc_label_wts = torch.from_numpy(calc_label_weights(label_vocab, doc_label_freq, num_docs)).cuda()
	else:
		sent_label_wts = None
		doc_label_wts = None

	print("Preparing label data and model...")
	charges = prepare_charges(label_data)

	model = Proposed(len(vocab), args.embed_dim, len(label_vocab), 
		charges['charge_text'], charges['sent_lens'], charges['doc_lens'], 
		args.device, sent_label_wts, doc_label_wts, ptemb_matrix).to(args.device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
	
	if args.lr_patience != -1:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.lr_patience, factor=args.lr_factor, verbose=True)
	else:
		scheduler = None

	metrics, model = train(model, traindev_data, test_data, optimizer, 
		lr_scheduler=scheduler, num_epochs=args.epochs, batch_size=args.batch_size, device=args.device)

	with open(args.save_path + "metrics.json", 'w') as fw:
		json.dump(metrics, fw)
	torch.save(model, args.save_path + "model.pt")

if __name__ == '__main__':
	main()