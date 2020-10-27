import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
from termcolor import colored
from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_test_datasets():
    # Download the language files
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # define the tokenizer
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    # Create the pytext's Field
    source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    # Splits the data in Train, Test and Validation data
    _, _, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))

    return test_data


class Encoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, encoder_hidden_dim, n_layers=1, dropout_prob=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, n_layers, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embedded = self.dropout(self.embedding(input_batch))
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.attn_vector = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size, src_len = encoder_outputs.shape[1], encoder_outputs.shape[0]

        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))

        attn_vector = self.attn_vector(attn_hidden).squeeze(2)

        return F.softmax(attn_vector, dim=1)


class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, dropout_prob=0.5):
        super().__init__()

        self.output_dim = input_output_dim
        self.attention = attention

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.rnn = nn.GRU(encoder_hidden_dim + embedding_dim, decoder_hidden_dim)

        self.fc = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, input_output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        attention_weights = torch.bmm(a, encoder_outputs)

        attention_weights = attention_weights.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, attention_weights), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = attention_weights.squeeze(0)

        predicted_token = self.fc(torch.cat((output, weighted, embedded), dim=1))

        return predicted_token, hidden, a.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, encoder_outputs, hidden, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.one_step_decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        input = target[0, :]

        for t in range(1, trg_len):
            output, hidden, a = self.one_step_decoder(input, hidden, encoder_outputs)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = target[t] if teacher_force else top1

        return outputs


class EncodeDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(source)
        return self.decoder(target, encoder_outputs, hidden, teacher_forcing_ratio)


def create_model_for_inference(source_vocab, target_vocab):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5

    # Instantiate the models
    attention_model = Attention(hidden_dim, hidden_dim)
    encoder = Encoder(len(source_vocab), embedding_dim, hidden_dim)
    one_step_decoder = OneStepDecoder(len(target_vocab), embedding_dim, hidden_dim, hidden_dim, attention_model)
    decoder = Decoder(one_step_decoder, device)

    model = EncodeDecoder(encoder, decoder)

    model = model.to(device)

    return model


def load_models_and_test_data(file_name):
    test_data = get_test_datasets()
    checkpoint = torch.load(file_name)
    source_vocab = checkpoint['source']
    target_vocab = checkpoint['target']
    model = create_model_for_inference(source_vocab, target_vocab)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, source_vocab, target_vocab, test_data


def predict(id, model, source_vocab, target_vocab, test_data, display_attention=False, debug=False):
    src = vars(test_data.examples[id])['src']
    trg = vars(test_data.examples[id])['trg']

    # Convert each source token to integer values using the vocabulary
    tokens = ['<sos>'] + [token.lower() for token in src] + ['<eos>']
    src_indexes = [source_vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    model.eval()

    # Run the forward pass of the encoder
    encoder_outputs, hidden = model.encoder(src_tensor)

    # Take the integer value of <sos> from the target vocabulary.
    trg_index = [target_vocab.stoi['<sos>']]
    next_token = torch.LongTensor(trg_index).to(device)

    attentions = torch.zeros(30, 1, len(src_indexes)).to(device)

    trg_indexes = [trg_index[0]]

    outputs = []
    with torch.no_grad():
        # Use the hidden and cell vector of the Encoder and in loop
        # run the forward pass of the OneStepDecoder until some specified
        # step (say 50) or when <eos> has been generated by the model.
        for i in range(30):
            output, hidden, a = model.decoder.one_step_decoder(next_token, hidden, encoder_outputs)

            attentions[i] = a

            # Take the most probable word
            next_token = output.argmax(1)

            trg_indexes.append(next_token.item())

            predicted = target_vocab.itos[output.argmax(1).item()]
            if predicted == '<eos>':
                break
            else:
                outputs.append(predicted)
    if debug:
        print(colored(f'Ground Truth    = {" ".join(trg)}', 'green'))
        print(colored(f'Predicted Label = {" ".join(outputs)}', 'red'))

    predicted_words = [target_vocab.itos[i] for i in trg_indexes]

    if display_attention:
        display_attention(src, predicted_words[1:], attentions[:len(predicted_words) - 1])

    return predicted_words


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                       rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def cal_bleu_score(dataset, model, source_vocab, target_vocab):
    targets = []
    predictions = []

    for i in range(len(dataset)):
        target = vars(test_data.examples[i])['trg']
        predicted_words = predict(i, model, source_vocab, target_vocab, dataset)
        predictions.append(predicted_words[1:-1])
        targets.append([target])
        if i < 10:
            print(colored(predicted_words[1:-1], 'red'))
            print(colored(target, 'green'))

    print(f'BLEU Score: {round(bleu_score(predictions, targets) * 100, 2)}')


if __name__ == '__main__':
    checkpoint_file = 'nmt-model-gru-attention-5.pth'
    model, source_vocab, target_vocab, test_data = load_models_and_test_data(checkpoint_file)
    # predict(20, model, source_vocab, target_vocab, test_data)
    # predict(14, model, source_vocab, target_vocab, test_data)

    cal_bleu_score(test_data, model, source_vocab, target_vocab)
