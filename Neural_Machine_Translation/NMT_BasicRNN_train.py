import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_datasets(batch_size=128):
    # Download the language files
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    # define the tokenizer
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    # Create the pytext's Field
    source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    # Splits the data in Train, Test and Validation data
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))

    # Build the vocabulary for both the language
    source.build_vocab(train_data, min_freq=3)
    target.build_vocab(train_data, min_freq=3)

    # Create the Iterator using builtin Bucketing
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          batch_size=batch_size,
                                                                          sort_within_batch=True,
                                                                          sort_key=lambda x: len(x.src),
                                                                          device=device)
    return train_iterator, valid_iterator, test_iterator, source, target


class Encoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_dim, n_layers, dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embed = self.dropout(self.embedding(input_batch))
        outputs, (hidden, cell) = self.rnn(embed)

        return hidden, cell


class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, hidden_dim, n_layers, dropout_prob):
        super().__init__()
        # self.input_output_dim will be used later
        self.input_output_dim = input_output_dim

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, target_token, hidden, cell):
        target_token = target_token.unsqueeze(0)
        embedding_layer = self.dropout(self.embedding(target_token))
        output, (hidden, cell) = self.rnn(embedding_layer, (hidden, cell))

        linear = self.fc(output.squeeze(0))

        return linear, hidden, cell


class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, cell, teacher_forcing_ratio=0.5):
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.one_step_decoder.input_output_dim
        # Store the predictions in an array for loss calculations
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        # Take the very first word token, which will be sos
        input = target[0, :]

        # Loop through all the time steps, starts from 1
        for t in range(1, target_len):
            predict, hidden, cell = self.one_step_decoder(input, hidden, cell)

            predictions[t] = predict
            input = predict.argmax(1)

            # Teacher forcing
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if do_teacher_forcing else input

        return predictions


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source)
        outputs = self.decoder(target, hidden, cell, teacher_forcing_ratio)

        return outputs


def create_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5

    # Instanciate the models
    encoder = Encoder(len(source.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    one_step_decoder = OneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    decoder = Decoder(one_step_decoder, device)

    model = EncoderDecoder(encoder, decoder)

    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Makes sure the CrossEntropyLoss ignores the padding tokens.
    TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

    return model, optimizer, criterion


def train(train_iterator, valid_iterator, source, target, epochs=10):
    model, optimizer, criterion = create_model(source, target)

    clip = 1

    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

        training_loss = []
        # set training mode
        model.train()

        # Loop through the training batch
        for i, batch in enumerate(train_iterator):
            # Get the source and target tokens
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            # Forward pass
            output = model(src, trg)

            # reshape the output
            output_dim = output.shape[-1]

            # Discard the first token as this will always be 0
            output = output[1:].view(-1, output_dim)

            # Discard the sos token from target
            trg = trg[1:].view(-1)

            # Calculate the loss
            loss = criterion(output, trg)

            # back propagation
            loss.backward()

            # Gradient Clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            training_loss.append(loss.item())

            pbar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}", refresh=True)
            pbar.update()

        with torch.no_grad():
            # Set the model to eval
            model.eval()

            validation_loss = []

            # Loop through the validation batch
            for i, batch in enumerate(valid_iterator):
                src = batch.src
                trg = batch.trg

                # Forward pass
                output = model(src, trg, 0)

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                # Calculate Loss
                loss = criterion(output, trg)

                validation_loss.append(loss.item())

        pbar.set_postfix(
            epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, val loss= {round(sum(validation_loss) / len(validation_loss), 4)}",
            refresh=False)
        pbar.close()

    return model


if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator, source, target = get_datasets(batch_size=512)
    model = train(train_iterator, valid_iterator, source, target, epochs=25)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'source': source.vocab,
        'target': target.vocab
    }

    torch.save(checkpoint, 'nmt-model-lstm-25.pth')
