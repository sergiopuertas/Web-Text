import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
import spacy
from gensim.models import KeyedVectors
nlp = spacy.load("en_core_web_sm")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = LabelField(dtype=torch.float)

fields = {'text': ('text', TEXT), 'score': ('score', LABEL)}
data = TabularDataset(path='data/review.csv', format='csv', fields=fields)

train_data, valid_data, test_data = data.split(split_ratio=[0.7, 0.15, 0.15])

# GloVe weights will be loaded directly in TEXT.build_vocab
TEXT.build_vocab(train_data, max_size=25000, vectors=GloVe(name='6B', dim=100))
LABEL.build_vocab(train_data)

# Function to create embedding matrix
def create_embedding_matrix(word_index, model):
    embedding_matrix = torch.zeros((len(word_index), EMBEDDING_DIM))
    for word, idx in word_index.items():
        try:
            embedding_matrix[idx] = torch.FloatTensor(model[word])
        except KeyError:
            embedding_matrix[idx] = torch.randn(EMBEDDING_DIM)
    return embedding_matrix

# We have to download the corresponding word2vec/fasttext models!!!!!!
word2vec_weights = create_embedding_matrix(TEXT.vocab.stoi, word2vec_model)
fasttext_weights = create_embedding_matrix(TEXT.vocab.stoi, fasttext_model.wv)

# Create iterators
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

# GRU Sentiment Model
class GRUSentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, embedding_weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        hidden = hidden[-1,:,:]
        return self.fc(hidden)

# Training Function
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation Function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# Model Parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
DROPOUT = 0.5

model = GRUSentimentModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, TEXT.vocab.vectors).to(device)

# Training and Evaluation
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)
N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')
# Testing
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')


def predict_sentiment(model, sentence, min_len=5):
    model.eval()
    with torch.no_grad():
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))

        indexed = [TEXT.vocab.stoi[t] for t in tokenized]

        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)

        prediction = torch.sigmoid(model(tensor))

        return "Positive review!" if prediction.item() > 0.5 else "Negative review!"

# Ejemplo de uso
sentence = "This movie was absolutely fantastic!"
prediction = predict_sentiment(model, sentence)
print(f'Prediction: {prediction}')
