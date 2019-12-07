import numpy as np
import torch
from torch import nn
from torch.functional import F
import matplotlib.pyplot as plt


def pre_processing():
    """
    This function reads the file and encodes the text into integer.

    :arguments:
    ----------------------------------------
        None
    :return:
     ---------------------------------------
        encoded_text:   The encoded text in integer
        int2char    :   dict object containing the int to char mapping
        char2int    :   dict object containing the char to int mapping

    """

    # Read the python file.
    with open('datasets/data.txt', 'r') as f:
        text = f.read()

    # get the unique list of characters
    chars = set(text)

    # Create hashmap with integer key and char as values
    int2char = {i: c for i, c in enumerate(chars)}
    # Create hashmap with char as key and integer as values
    char2int = {c: i for i, c in enumerate(chars)}

    # Convert the entire text to integer values and return as numpy array
    encoded_text = np.array([char2int[t] for t in text])

    return encoded_text, int2char, char2int


def one_hot_encode(batch, n_labels):
    """
    Converts the integer data to one-hot-encoding

    :arguments:
    ----------------------------------------
        batch   :   The data in nxm dimension
        n_labels:   Number of unique chars
    :return:
     ---------------------------------------
        one_hot :   Matrix having nxmxd dimension.

    """

    batch_squ_len = batch.shape[0] * batch.shape[1]

    # Create one hot placeholder with all zeros
    one_hot = np.zeros((batch_squ_len, n_labels), dtype=np.float32)

    # Update the related zeros to 1
    one_hot[np.arange(batch_squ_len), batch.flatten()] = 1

    # Convert to 3D Matrix. The 3rd dimension represents the one hot encoding
    one_hot = one_hot.reshape((batch.shape[0], batch.shape[1], n_labels))

    return one_hot


def get_batch(encoded_text, seq_len, batch_len):
    """
    Creates batches from the input data

    :arguments:
    ----------------------------------------
        encoded_text    :   The encoded text in integer
        seq_len         :   Sequence Length
        batch_len       :   Batch Length
    :return:
     ---------------------------------------
        x   :   Input Data
        y   :   Target Data

    """

    # Calculate the batch size
    batch_size = seq_len * batch_len

    # Find number of total number of batch
    possible_batches = len(encoded_text) // batch_size

    # Trim the encoded text array and reshape it to have seq_len number of rows
    encoded_text = encoded_text[:possible_batches * batch_size].reshape((seq_len, -1))

    # Loop through the array
    for n in range(0, encoded_text.shape[1], batch_len):

        # Get the input x with batch_len cols
        x = encoded_text[:, n:n + batch_len]

        # The target (y) for the last batch needs to overlap and the last col should be the first column of the input(x)
        # Verify we have reached at the last batch
        if n + batch_len >= encoded_text.shape[1]:

            # Copy only the last batch_len-1 columns
            y = encoded_text[:, n + 1:n + batch_len]
            # Copy only the first column
            y2 = encoded_text[:, 0].reshape((1, -1))

            # Append the data by column
            y = np.append(y, y2.T, axis=1)
        else:
            # Copy the next batch_len columns
            y = encoded_text[:, n + 1:n + batch_len + 1]

        # Return input and target
        yield x, y


class CharRNN(nn.Module):

    def __init__(self, int2char, char2int, batch_len=100, n_hidden=256, n_layers=2, drop_prob=0.5):

        """
        Initializes the CharRNN class

        :arguments:
        ----------------------------------------
            int2char    :   dict object containing the int to char mapping
            char2int    :   dict object containing the char to int mapping
            batch_len   :   Batch Length
            n_hidden    :   # of hidden layers
            n_layers    :   # of stacked LSTM
            drop_prob   :   Dropout Probability
        :return:
         ---------------------------------------
            None

        """

        super().__init__()

        self.int2char = int2char
        self.char2int = char2int
        self.batch_len = batch_len
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # LSTM Inputs:
        #   Input length    : This should be the length of one hot encoding. i.e # of unique chars
        #   Hidden Layer    : Number of hidden layer passed
        #   Stacked Layers  : Number of stacked LSTM Layers
        #   dropout         : Dropout ratio
        #   batch_first     : The first dimension will indicate # of batches
        self.lstm = nn.LSTM(len(self.int2char), self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define the final fully connected later. The input will be same size as the # of hidden layer,
        # the output will be same as the encoding size ( input size = target/out size )
        self.fc = nn.Linear(self.n_hidden, len(self.int2char))

        # Manually initialize the weights of the FC layer
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the FC Layer
        :return:
        """

        # Initialize the weights using uniform distribution
        self.fc.weight.data.uniform_(-1, 1)
        # Initialize the weights using 0
        self.fc.bias.data.fill_(0)

    def init_hidden(self, seq_len):

        """
        Initializes the hidden state
        :return:
        """

        # Initialize the hidden state
        weight = next(self.parameters()).data

        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        return (weight.new(self.n_layers, seq_len, self.n_hidden).zero_(),
                weight.new(self.n_layers, seq_len, self.n_hidden).zero_())

    def forward(self, x, hc):
        """
        Forward propagation through the network

        :arguments:
        ----------------------------------------
            x   :   Input Batch Data
            hc  :   Hidden/cell state
        :return:
         ---------------------------------------
            None

        """

        # Inputs: input, (h_0, c_0)
        #    input (seq_len, batch, input_size): tensor containing the features of the input sequence.
        #       The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
        #    h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        #    c_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        #    If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (h, c) = self.lstm(x, hc)

        # Dropout Layer
        x = self.dropout(x)

        # Flatten the batch and seq
        x = x.view(x.size()[0] * x.size()[1], self.n_hidden)

        # FC Layer
        x = self.fc(x)

        return x, (h, c)

    def predict(self, char, h=None, top_k=None):
        """
        Predicts the next char in sequence

        :arguments:
        ----------------------------------------
           char     :   Current char
           h        :   Hidden/cell state
           top_k    :   Top K Probability
        :return:
        ---------------------------------------
           output   :   Predicted sequence
           hc       :   Hidden/cell state

        """

        # If GPU is available then use it
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        # In case hidden layer is not available, it needs to be initialized
        if h is None:
            # n_seqs should be 1, as we are going to predict only one char at a time
            h = self.init_hidden(1)

        # Convert the input char to integer
        x = np.array([[self.char2int[char]]])

        # Perform the one hot encoding
        x = one_hot_encode(x, len(self.char2int))

        # Convert Numpy Array to Tensor
        input = torch.from_numpy(x)

        if torch.cuda.is_available():
            input = input.cuda()

        h = tuple([each.data for each in h])
        out, hc = self.forward(input, h)

        # Calculate the probablity using softmax
        p = F.softmax(out, dim=1).data

        # Move p to CPU
        if torch.cuda.is_available():
            p = p.cpu()

        # Get top k predicted values
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

        # Randomly choose from top k prediction
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        # Return the char and hc ( to be used in next prediction )
        return self.int2char[char], hc


def train(model: CharRNN, input, epochs=10, seq_len=10, batch_len=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """
    The training loop for the model

    :arguments:
    ----------------------------------------
        model       :  CharRNN class
        input       :  Encoded input text
        epochs      :  epochs
        seq_len     :  Sequence length
        batch_len   :  Batch Length
        lr          :  Learning Rate
        clip        :  Gradient Clipping Value
        val_frac    :  Validation set ratio
        print_every :  Print logs
    :return:
    ---------------------------------------
       None

    """

    # Set the model to training mode
    model.train()

    # Define optimization process
    optimization = torch.optim.Adam(model.parameters(), lr=lr)

    # Define Loss function
    error_func = nn.CrossEntropyLoss()

    # Create training and validation data
    train_index = int(len(input) * (1 - val_frac))
    train_data, val_data = input[:train_index], input[train_index:]

    # Move the model to GPU ( if available )
    if torch.cuda.is_available():
        model.cuda()

    training_loss = []
    validation_loss = []

    # Loop through the epochs
    for i in range(epochs):

        # Initialize the hidden layers
        hc = model.init_hidden(seq_len)

        # Loop though the batches [ Mini-Batch SGD ]
        for index, (x, y) in enumerate(get_batch(train_data, seq_len, batch_len)):

            # Perform one hot encoding
            x = one_hot_encode(x, len(model.int2char))

            # Convert Numpy Arrays to PyTorch Tensors
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            # Move the input and targets to GPU ( if available )
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            hc = tuple([each.data for each in hc])

            # Remove the gradients from the model
            model.zero_grad()

            # Forward Propagation
            output, hc = model.forward(inputs, hc)

            # Calculate the Loss
            # The output will be of dim (128x100x162) the targets will be of dim (128x100)
            loss = error_func(output, targets.view(seq_len * batch_len))

            # Backprop
            loss.backward()

            # Gradient clipping ( needed to avoid exploding gradients )
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimization.step()

            if index % print_every == 0:

                # Calculate validation loss

                val_hc = model.init_hidden(seq_len)
                val_losses = []

                # Loop through validation batches
                for val_index, (x, y) in enumerate(get_batch(val_data, seq_len, batch_len)):

                    # One hot encode and convert to torch tensors
                    x = one_hot_encode(x, len(model.int2char))
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_hc = tuple([each.data for each in val_hc])

                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_hc = model.forward(inputs, val_hc)

                    val_loss = error_func(output, targets.view(seq_len * batch_len))
                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(index),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

        validation_loss.append(np.mean(val_losses))
        training_loss.append(loss.item())

    # print Training/Validation Loss
    plot_loss(training_loss, validation_loss)


def plot_loss(training_loss, validation_loss):
    """
    Plots the training vs validation loss

    :arguments:
    ----------------------------------------
        model       :  CharRNN class
        train_loader:  Training batch data loader
        val_loader  :  Validation batch data loader
        batch_size  :  Batch Size
        epochs      :  epochs
        lr          :  Learning Rate
        clip        :  Gradient Clipping Value
        print_every :  Print logs
    :return:
    ---------------------------------------
       None

    """
    plt.plot(training_loss, 'r--')
    plt.plot(validation_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def sample(model: CharRNN, size, prime='import ', top_k=5):
    """
    Generate sample output

    :arguments:
    ----------------------------------------
        model       :  CharRNN class
        size        :  Number of predicted char
        prime       :  Initial values to start with
        top_k       :  Top K predicted values

    :return:
    ---------------------------------------
       prediction   :   Predicted String

    """

    if torch.cuda.is_available():
        model.cuda()

    # set the model to evaluation mode
    model.eval()

    # Initially run through the prime data
    chars = [c for c in prime]
    hc = model.init_hidden(1)
    for c in prime:
        char, hc = model.predict(c, hc, top_k=top_k)

    # Predict by passing the previous char
    for index in range(size):
        char, hc = model.predict(chars[-1], hc, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


if __name__ == '__main__':

    training = True

    if training:

        # Perform Pre-processing
        encoded_text, int2char, char2int = pre_processing()

        # Define the CharRNN Class
        model = CharRNN(int2char, char2int, n_hidden=512, n_layers=2)
        print(model)

        # Train the model
        train(model, encoded_text, epochs=50, seq_len=128, batch_len=200, lr=0.001, print_every=100)

        # Save the model
        model_name = 'model/rnn_50_epoch_new.net'

        checkpoint = {'n_hidden': model.n_hidden,
                      'n_layers': model.n_layers,
                      'state_dict': model.state_dict(),
                      'int2char': model.int2char,
                      'char2int': model.char2int,
                      'batch_len': model.batch_len
                      }

        with open(model_name, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        # Open the model checkpoint
        with open('model/rnn_50_epoch.net', 'rb') as f:
            checkpoint = torch.load(f)
        # Initialize the CharRNN class
        model = CharRNN(checkpoint['int2char'], checkpoint['char2int'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        # Load the trained weights and biases
        model.load_state_dict(checkpoint['state_dict'])

        # Prediction
        print(sample(model, 5000))
