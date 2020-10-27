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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_datasets(batch_size=128):
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

        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)

        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of [source len, batch size, decoder hidden dim]
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [source len, batch size]
        self.attn_scoring_fn = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]

        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        hidden = hidden.repeat(src_len, 1, 1)

        # Calculate Attention Hidden values
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate the Scoring function. Remove 3rd dimension.
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)

        # The attn_scoring_vector has dimension of [source len, batch size]
        # Since we need to calculate the softmax per record in the batch
        # we will switch the dimension to [batch size,source len]
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)

        # Softmax function for normalizing the weights to
        # probability distribution
        return F.softmax(attn_scoring_vector, dim=1)


class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, dropout_prob=0.5):
        super().__init__()

        self.output_dim = input_output_dim
        self.attention = attention

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)

        # Add the encoder_hidden_dim and embedding_dim
        self.rnn = nn.GRU(encoder_hidden_dim + embedding_dim, decoder_hidden_dim)
        # Combine all the features for better prediction
        self.fc = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, input_output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden, encoder_outputs):
        # Add the source len dimension
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        # Calculate the attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # We need to perform the batch wise dot product.
        # Hence need to shift the batch dimension to the front.
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Use PyTorch's bmm function to calculate the
        # weight W.
        W = torch.bmm(a, encoder_outputs)

        # Revert the batch dimension.
        W = W.permute(1, 0, 2)

        # concatenate the previous output with W
        rnn_input = torch.cat((embedded, W), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # Remove the sentence length dimension and pass them to the Linear layer
        predicted_token = self.fc(torch.cat((output.squeeze(0), W.squeeze(0), embedded.squeeze(0)), dim=1))

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
            # Pass the encoder_outputs. For the first time step the
            # hidden state comes from the encoder model.
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

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(source)
        return self.decoder(target, encoder_outputs, hidden, teacher_forcing_ratio)


def create_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5

    # Instantiate the models
    attention_model = Attention(hidden_dim, hidden_dim)
    encoder = Encoder(len(source.vocab), embedding_dim, hidden_dim)
    one_step_decoder = OneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, hidden_dim, attention_model)
    decoder = Decoder(one_step_decoder, device)

    model = EncodeDecoder(encoder, decoder)

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


train_iterator, valid_iterator, test_iterator, source, target = get_datasets(batch_size=256)
model = train(train_iterator, valid_iterator, source, target, epochs=5)

checkpoint = {
    'model_state_dict': model.state_dict(),
    'source': source.vocab,
    'target': target.vocab
}

torch.save(checkpoint, 'nmt-model-gru-attention-five.pth')
