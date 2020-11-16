from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.utils.rnn as U


def prepare_charges(data, device='cuda'):
    batch = defaultdict(list)
    
    for idx in range(len(data)):
        num_sents = len(data[idx]['text'])
        skipped_sents = 0
        for sidx in range(num_sents):
            sent = data[idx]['text'][sidx]

            if len(sent) == 0:
                skipped_sents += 1 
                continue
            
            batch['charge_text'].append(torch.tensor(sent, dtype=torch.long, device=device))
            batch['sent_lens'].append(len(sent))
        
        batch['doc_lens'].append(num_sents - skipped_sents)

    batch['charge_text'] = U.pad_sequence(batch['charge_text'], batch_first=True)
    batch['sent_lens'] = torch.tensor(batch['sent_lens'], dtype=torch.long, device=device)
    batch['doc_lens'] = torch.tensor(batch['doc_lens'], dtype=torch.long, device=device)
    
    return batch      
        

def prepare_minibatch(data, batch_size=5, device='cuda', shuffle=False):
    perm_index = torch.randperm(len(data)) if shuffle else torch.arange(len(data))

    batch_slabs = True if 'sent_labels' in data[0] else False
    batch_dlabs = True if 'doc_labels' in data[0] else False

    start = 0
    while start < len(data):
        end = min(start + batch_size, len(data))
        batch = defaultdict(list)

        for idx in perm_index[start : end]:
            num_sents = len(data[idx]['text'])
            skipped_sents = 0
            for sidx in range(num_sents):
                sent = data[idx]['text'][sidx]
                if len(sent) == 0:
                    skipped_sents += 1
                    continue
                batch['fact_text'].append(torch.tensor(sent, dtype=torch.long, device=device))
                batch['sent_lens'].append(len(sent))

                if batch_slabs:
                    batch['sent_labels'].append(torch.tensor(data[idx]['sent_labels'][sidx], dtype=torch.float, device=device))

            if batch_dlabs:
                batch['doc_labels'].append(torch.tensor(data[idx]['doc_labels'], dtype=torch.float, device=device))
            batch['doc_lens'].append(num_sents - skipped_sents)

        batch['fact_text'] = U.pad_sequence(batch['fact_text'], batch_first=True)
        batch['sent_lens'] = torch.tensor(batch['sent_lens'], dtype=torch.long, device=device)
        batch['doc_lens'] = torch.tensor(batch['doc_lens'], dtype=torch.long, device=device)

        if batch_slabs:
            batch['sent_labels'] = torch.stack(batch['sent_labels'])

        if batch_dlabs:
            batch['doc_labels'] = torch.stack(batch['doc_labels'])

        yield batch
        start = end

def train_eval_pass(model, data, train=False, optimizer=None, batch_size=5, device='cuda'):
    if train:
        model.train()
    else:
        model.eval()

    metrics = {}
    skipped = 0
    loss = 0
    num_batches = 0
    
    metrics_tracker = defaultdict(lambda: torch.zeros((model.num_labels,), device=device))
    
    def update_metrics_tracker(preds, labels):
        match = preds * labels
        metrics_tracker['preds'] += torch.sum(preds, dim=0)
        metrics_tracker['labels'] += torch.sum(labels, dim=0)
        metrics_tracker['match'] += torch.sum(match, dim=0)

    for batch in prepare_minibatch(data, batch_size, device, train):
        if 'cuda' in device:
            torch.cuda.empty_cache()
        # try:
        model_out = model(batch)
        if train:
            optimizer.zero_grad()
            model_out['loss'].backward()
            optimizer.step()
        
        update_metrics_tracker(model_out['doc_preds'], batch['doc_labels'])
        loss += model_out['loss'].item()

        # except RuntimeError:
        #     skipped += 1
        #     continue
            
        # finally:
        num_batches += 1
            
    metrics['loss'] = loss / num_batches
    metrics.update(calc_metrics(metrics_tracker))
    
    return metrics

def calc_metrics(tracker):
    precision = tracker['match'] / tracker['preds']
    recall = tracker['match'] / tracker['labels']
    f1 = 2 * precision * recall / (precision + recall)
    
    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1[torch.isnan(f1)] = 0
    
    metrics = {}
    metrics['label-P'] = precision.tolist()
    metrics['label-R'] = recall.tolist()
    metrics['label-F1'] = f1.tolist()
    metrics['macro-P'] = precision.mean().item()
    metrics['macro-R'] = recall.mean().item()
    metrics['macro-F1'] = f1.mean().item()
    
    return metrics

def train(model, train_data, dev_data, optimizer, lr_scheduler=None, num_epochs=100, batch_size=5, device='cuda'):
    best_metrics = {'macro-F1': 0}
    best_model = model.state_dict()

    print("%5s || %8s | %8s || %8s | %8s %8s %8s" % ('EPOCH', 'Tr-LOSS', 'Tr-F1', 'Dv-LOSS', 'Dv-P', 'Dv-R', 'Dv-F1'))
    
    for epoch in range(num_epochs):
        tr_mets = train_eval_pass(model, train_data, train=True, optimizer=optimizer, batch_size=batch_size, device=device)
        dv_mets = train_eval_pass(model, dev_data, batch_size=batch_size, device=device)
        
        if lr_scheduler is not None:
            lr_scheduler.step(dv_mets['macro-F1'])
        
        print("%5d || %8.4f | %8.4f || %8.4f | %8.4f %8.4f %8.4f" % (epoch, tr_mets['loss'], tr_mets['macro-F1'], dv_mets['loss'], dv_mets['macro-P'], dv_mets['macro-R'], dv_mets['macro-F1']))
        
        if dv_mets['macro-F1'] > best_metrics['macro-F1']: 
            best_metrics = dv_mets
            best_model = model.state_dict()
            
    print("%5s || %8s | %8s || %8.4f | %8.4f %8.4f %8.4f" % ('BEST', '-', '-', best_metrics['loss'], best_metrics['macro-P'], best_metrics['macro-R'], best_metrics['macro-F1']))
    
    return best_metrics, best_model

def infer(model, data, label_vocab, batch_size=5, device='cuda'):
    model.eval()
    predictions = []

    inv_label_vocab = {i: l for l, i in label_vocab.items()}

    for batch in prepare_minibatch(data, batch_size, device, False):
        if 'cuda' in device:
            torch.cuda.empty_cache()
        
        model_out = model(batch)

        for doc in model_out['doc_preds']:
            pred = [inv_label_vocab[idx.item()] for idx in doc.nonzero(as_tuple=False)]
            predictions.append(pred)

    return predictions

        
    
    
            
    
        
        
            
        
