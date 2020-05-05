import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

# Move Data to a Panda DataFrame
df = pd.read_csv('cola/raw/in_domain_train.tsv', delimiter='\t', header=None,
                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

print(df.shape)

# Sentence and Label Lists
sentences = df.sentence.values
sentences = ["[CLS]" + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

# Tokenize Inputs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print("Tokenize version of the the first sentence:")
print(tokenized_texts[0])

# Padding Sentences
# Set the maximum sequence length. The longest sequence in our training set
# is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 128

# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Index Numbers and Padding
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# pad sentences
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                          dtype="long", truncating="post", padding="post")

# Attention masks

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Train and Validation Set

train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids, labels, random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = \
    train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

# transform all data into torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Preparation for Training

# Select a batch size for training. For fine tuning BERT on a
# specific task , BERT authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader
# This helps save on memory during training because, unlike a for loop,
# with iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()
