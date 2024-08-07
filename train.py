import re
import time
import pickle
import string
import argparse
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import wandb

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text): 
    url_pattern  = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)

def clean_text(text): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and len(w) > 2)]) 
    
    return text2.lower()

def get_sentiment(sentiment):
    if sentiment == 'positive':
        return 2
    elif sentiment == 'negative':
        return 1
    else:
        return 0
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        wandb.log({"train_loss": loss.item(), "train_accuracy": total_acc / total_count})

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            wandb.log({"valid_loss": loss.item(), "valid_accuracy": total_acc / total_count})

    return total_acc / total_count

parser = argparse.ArgumentParser(description="Tweet Sentiment Analysis")
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=10, help='learning rate')

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))


wandb.init(project="tweet-sentiment-analysis")


train_data = pd.read_csv("tweet-sentiment-extraction\\train.csv")
train_data.dropna(axis=0, how='any', inplace=True) 
train_data['Num_words_text'] = train_data['text'].apply(lambda x: len(str(x).split())) 
mask = train_data['Num_words_text'] > 2
train_data = train_data[mask]
print('-------Train data--------')
print(train_data['sentiment'].value_counts())
print(len(train_data))
print('-------------------------')
max_train_sentence_length = train_data['Num_words_text'].max()

train_data['text'] = train_data['text'].apply(remove_emoji)
train_data['text'] = train_data['text'].apply(remove_url)
train_data['text'] = train_data['text'].apply(clean_text)

train_data['label'] = train_data['sentiment'].apply(get_sentiment)

test_data = pd.read_csv("tweet-sentiment-extraction\\test.csv")
test_data.dropna(axis=0, how='any', inplace=True) 
test_data['Num_words_text'] = test_data['text'].apply(lambda x: len(str(x).split())) 

max_test_sentence_length = test_data['Num_words_text'].max()

mask = test_data['Num_words_text'] > 2
test_data = test_data[mask]

print('-------Test data--------')
print(test_data['sentiment'].value_counts())
print(len(test_data))
print('-------------------------')

test_data['text'] = test_data['text'].apply(remove_emoji)
test_data['text'] = test_data['text'].apply(remove_url)
test_data['text'] = test_data['text'].apply(clean_text)

test_data['label'] = test_data['sentiment'].apply(get_sentiment)

print('Train Max Sentence Length: '+str(max_train_sentence_length))
print('Test Max Sentence Length: '+str(max_test_sentence_length))

X_train, X_valid, Y_train, Y_valid = train_test_split(
    train_data['text'].tolist(),
    train_data['label'].tolist(),
    test_size=0.2,
    stratify=train_data['label'].tolist(),
    random_state=0
)

print('Train data len: '+str(len(X_train)))
print('Class distribution '+str(Counter(Y_train)))

print('Valid data len: '+str(len(X_valid)))
print('Class distribution '+str(Counter(Y_valid)))

print('Test data len: '+str(len(test_data['text'].tolist())))
print('Class distribution '+str(Counter(test_data['label'].tolist())))

train_dat = list(zip(Y_train, X_train))
valid_dat = list(zip(Y_valid, X_valid))
test_dat = list(zip(test_data['label'].tolist(), test_data['text'].tolist()))

tokenizer = get_tokenizer('basic_english')
train_iter = train_dat
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

train_iter1 = train_dat
num_class = len(set([label for (label, text) in train_iter1]))
vocab_size = len(vocab)
emsize = 128
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


BATCH_SIZE = 16 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_iter2 = train_dat
test_iter2 = test_dat
valid_iter2 = valid_dat

train_dataloader = DataLoader(train_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter2, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

wandb.config.update({
    "epochs": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "embedding_size": emsize,
    "vocab_size": vocab_size,
    "num_classes": num_class
})

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
    
print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
torch.save(model.state_dict(), 'model.pth')
print("Modal checkpoint is saved as ./model.pth")

wandb.log({"test_accuracy": accu_test})

wandb.finish()
