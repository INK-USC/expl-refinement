import torch, torch.nn as nn
from transformers import BertModel, BertTokenizer

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


class BERT(nn.Module):
    def __init__(self, cuda_device, fine_tune=False):
        super(BERT, self).__init__()
        self.cuda_device = cuda_device
        self.fine_tune = fine_tune
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_segments_ids(self, tokenized_text):
        segments_ids = []
        curr_id = 0
        for t in tokenized_text:
            segments_ids.append(curr_id)
            if t == '[SEP]':
                curr_id += 1
        
        return segments_ids
    
    def forward(self, tokenized_text):
        indexed_tokens, segments_ids = [], []
        for t in tokenized_text:
            # tokenized_text_ = [r.lower() for r in t if not r in ['[CLS]', '[SEP]', '[PAD]']]
            tokenized_text_ = t
            # Convert token to vocabulary indices
            indexed_tokens.append(self.tokenizer.convert_tokens_to_ids(tokenized_text_))
            # Define indices associated to sentences
            segments_ids.append(self.get_segments_ids(tokenized_text_))
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)
        
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to(self.cuda_device)
        segments_tensors = segments_tensors.to(self.cuda_device)
        
        # Predict hidden states features for each layer
        if self.fine_tune:
            # outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            outputs = self.model(tokens_tensor)
            encoded_layers = outputs[0]
        else:
            with torch.no_grad():
                # See the models docstrings for the detail of the inputs
                outputs = self.model(tokens_tensor)
                # Transformers models always output tuples.
                # See the models docstrings for the detail of all the outputs
                # In our case, the first element is the hidden state of the last layer of the Bert model
                encoded_layers = outputs[0]
        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
        # assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), self.model.config.hidden_size)
        return encoded_layers

