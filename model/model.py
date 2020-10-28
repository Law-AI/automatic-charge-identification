class BiRNN(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.rnn = nn.LSTM(embed_dim, embed_dim // 2, batch_first = True, bidirectional = True)
        
        self.embed_dim = embed_dim
        self.device = device
        
    def forward(self, sequence_in, lens): # [B,L,E], [B]
        batch_size = sequence_in.shape[0]
        hidden = (torch.randn(2, batch_size, self.embed_dim // 2).to(self.device), torch.randn(2, batch_size, self.embed_dim // 2).to(self.device))
        
        packed_sequence_in = U.pack_padded_sequence(sequence_in, lens, batch_first = True, enforce_sorted = False)
        
        # [B,L,E] --> [B,L,E], [2,B,E/2]
        packed_sequence_out, (sequence_rep, _) = self.rnn(packed_sequence_in, hidden)
        
        sequence_out = U.pad_packed_sequence(packed_sequence_out, batch_first = True)[0]
        # [2,B,E/2] --> [B,E]
        sequence_rep = sequence_rep.permute(1,0,2).contiguous().view(batch_size, -1)
        
        return sequence_out, sequence_rep
    
class SeqAttn(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.context = nn.Parameter(torch.rand((embed_dim, 1)))
        
        self.device = device
    
    def forward(self, sequence, lens, dcontext=None): # [B,L,E], [B], [B,L,1], [B,E]
        batch_size = sequence.shape[0]
        max_len = sequence.shape[1]
        # [B,L]
        mask = (torch.arange(max_len).expand(len(lens), max_len).to(self.device) < lens.unsqueeze(1)).float()
        
        # [B,L,E] --> [B*L,E]
        sequence = torch.tanh(self.fc(sequence.view(-1, self.embed_dim)))
        # [B*L,E] --> [B,L,E]
        sequence = sequence.view(batch_size, -1, self.embed_dim)
        sequence = sequence * mask.unsqueeze(2)
        # [E,1] --> [B,E,1]
        context = self.context.expand(batch_size, self.embed_dim, 1) if dcontext is None else dcontext.unsqueeze(2)
        # [B,L,E], [B,E,1] --> [B,L]
        scores = torch.bmm(sequence, context).squeeze(2)
        probs = F.softmax(scores, dim = 1).unsqueeze(2)
        
        # [B,L,E], [B,L,1] --> [B,E]
        sequence_wt = torch.sum(sequence * probs, dim = 1)
        return sequence_wt
    
class Encoder(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.rnn_layer = BiRNN(embed_dim, device)
        self.attn_layer = SeqAttn(embed_dim, device)
        
    def forward(self, sequence, lens, context=None): # [B,L,E], [B]
        encoded, _ = self.rnn_layer(sequence, lens) # [B,L,E] --> [B,L,E]
        weighted = self.attn_layer(encoded, lens, context) # [B,L,E] --> [B,E]
        return weighted
    
class HAN(nn.Module):
    def __init__(self, embed_dim, device):
        super().__init__()
        
        self.sent_encoder = Encoder(embed_dim, device)
        self.doc_encoder = Encoder(embed_dim, device)
        
    def forward(self, sequence, sent_lens, doc_lens, sent_context=None, doc_context=None):
        encoded_sents = self.sent_encoder(sequence, sent_lens, sent_context)
        
        docs = U.pad_sequence(torch.split(encoded_sents, doc_lens.tolist()), batch_first=True)
        encoded_docs = self.doc_encoder(docs, doc_lens, doc_context)
        
        return encoded_sents, encoded_docs
    
class Proposed(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, batch):
        facts = batch['fact_text'] # [FS, Fw]
        charges = self.charge_text # [CS, Cw]
        
        num_fsents = batch['sent_lens'].size(0)
        num_fdocs = batch['doc_lens'].size(0)
        num_csents = self.charge_sent_lens.size(0)
        num_cdocs = self.charge_doc_lens.size(0)
        
        facts = self.embedder(facts) # [FS, Fw, E]
        charges = self.embedder(charges) # [CS, Cw, E]
        
        # [FS, Fw], [FS], [FD] --> [FS, E], [FD, E]
        fact_sents, facts = self.fact_encoder(facts, batch['sent_lens'], batch['doc_lens'])
        
        sent_context = self.sent_context(fact_sents) # [FS, E]
        doc_context = self.doc_context(fact_sents) # [FS, E]
        charge_context = self.charge_context(fact_sents) # [FS, E]
        
        sent_charges = charges.repeat(num_fsents, 1, 1) # [FS*CS, Cw, E]
        sent_context = sent_context.repeat(1, num_csents).view(-1, self.embed_dim) # [FS*CS, E]
        doc_context = doc_context.repeat(1, num_cdocs).view(-1, self.embed_dim) # [FS*CD, E]
        
        sent_charges = self.charge_encoder(sent_charges, self.charge_sent_lens.repeat(num_fsents), self.charge_doc_lens.repeat(num_fsents), sent_context, doc_context) # [FS*CD, E]