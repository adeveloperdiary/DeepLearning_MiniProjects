import numpy as np
import torch.nn as nn
from torch.functional import F
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class SentimentRNN(nn.Module):
    def __init__(self, word2int, embedding_dim, n_hidden, n_layers, output_size=1, drop_prob=0.5):
        """
        Initializes the SentimentRNN class

        :arguments:
        ----------------------------------------
            word2int        :   dict object containing the word to int mapping
            embedding_dim   :   Embedding Length
            n_hidden        :   # of hidden layers
            n_layers        :   # of stacked LSTM
            output_size     :   Output Size ( should be default to 1 here )
            drop_prob       :   Dropout Probability
        :return:
         ---------------------------------------
            None

        """

        super(SentimentRNN, self).__init__()

        # Initialize the variables
        self.word2int = word2int
        self.vocab_size = len(self.word2int) + 1  # Add one for the 0 paddings
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.drop_prob = drop_prob

        # Initialize the layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM Inputs:
        #   Input length    : This should be the length of one embedding dim. i.e # of unique chars
        #   Hidden Layer    : Number of hidden layer passed
        #   Stacked Layers  : Number of stacked LSTM Layers
        #   dropout         : Dropout ratio
        #   batch_first     : The first dimension will indicate # of batches
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        # Define a dropout layer
        self.dropout = nn.Dropout(0.3)

        # Define the final fully connected later. The input will be same size as the # of hidden layer,
        # the output will be always 1
        self.fc = nn.Linear(self.n_hidden, self.output_size)

        self.sigmoid = nn.Sigmoid()

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

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state
        :return:
        """

        # Initialize the hidden state
        weight = next(self.parameters()).data

        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        if torch.cuda.is_available():
            return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

    def forward(self, x, hidden):
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
        # Get the batch size, we need it for the sigmoid layer
        batch_size = x.size(0)

        # Forward Propagation through the embedding layer
        x = self.embedding(x.long())

        # Invoke the lstm layer
        x, hidden = self.lstm(x, hidden)

        # Flatten the output of LSTM
        x = x.contiguous().view(-1, self.n_hidden)

        # Dropout Layer
        x = self.dropout(x)

        # FC Layer
        x = self.fc(x)

        # Sigmoid Layer
        x = self.sigmoid(x)

        # Reshape the sigmoid output
        x = x.view(batch_size, -1)

        # Take only the last sigmoid output, ignore the rest
        # as we are interested only in the sentiment which is 0/1
        x = x[:, -1]

        return x, hidden


def pre_process():
    """
    This function reads the data files and encodes the text into integer.

    :arguments:
    ----------------------------------------
        None
    :return:
     ---------------------------------------
        reviews_encoded :   The encoded text in integer
        labels_encoded  :   Target Sentiment
        word2int        :   dict object containing the word to int mapping

    """

    # Read the reviews.txt and labels.txt, then store them in variables
    with open('datasets/reviews.txt', 'r') as f:
        reviews_text = f.read()

    with open('datasets/labels.txt', 'r') as f:
        sentiment = f.read()

    # Convert the reviews text to lowercase,
    # then remove all the punctuations
    reviews_text = reviews_text.lower()
    reviews_text = ''.join([c for c in reviews_text if c not in punctuation])

    # Each line contains one review. Create a list of reviews.
    reviews_list = reviews_text.split('\n')

    # Merge all the text in one string
    # This will be required for creating the char2int dict
    words = (' '.join(reviews_list)).split()

    # Build a dictionary that maps words to integers
    # Using Counter to set the values as number of word occurrences
    # High frequency words will have high occurrences
    counts = Counter(words)

    # Sort the dict based on # of occurrences from most to least
    sorted_dict = sorted(counts, key=counts.get, reverse=True)

    # Create the word2int dict with most frequent word starting from index 1
    # Reserve 0 for padding
    word2int = {word: index for index, word in enumerate(sorted_dict, 1)}

    # Encode the reviews text using word2int
    reviews_encoded = []
    for review in reviews_list:
        reviews_encoded.append([word2int[word] for word in review.split()])

    # Convert the labels to 0/1
    labels_encoded = []
    sentiment_list = sentiment.split('\n')
    for sentiment in sentiment_list:
        if sentiment == 'positive':
            labels_encoded.append(1)
        else:
            labels_encoded.append(0)

    return reviews_encoded, labels_encoded, word2int


def process_reviews(reviews_encoded, labels_encoded):
    """
    Process the encoded reviews:
        1. Remove empty reviews
        2. Convert the target to Numpy array


    :arguments:
    ----------------------------------------
        reviews_encoded     :   The encoded review text
        labels_encoded      :   target list
    :return:
     ---------------------------------------
        reviews_encoded     :   The encoded review text
        labels_ndarray      :   targets as Numpy array

    """

    # Remove entries with empty reviews
    count = 0
    for i, review in enumerate(reviews_encoded):
        if len(review) == 0:
            count = 1
            del reviews_encoded[i]
            del labels_encoded[i]

    print("Removed %d reviews" % count)

    # Convert the labels_encoded from list to Numpy Array
    labels_ndarray = np.array(labels_encoded)

    return reviews_encoded, labels_ndarray


def pad_features(reviews_encoded, seq_length):
    """
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    Also convert the reviews_encoded to Numpy Array

    :arguments:
    ----------------------------------------
        reviews_encoded :   The encoded review text
        seq_length      :   Length of sequence
    :return:
     ---------------------------------------
        reviews_ndarray     :   The encoded review text

    """

    features = list()

    # Loop through the reviews
    for i, reviews in enumerate(reviews_encoded):

        if len(reviews) > seq_length:
            # Truncate the reviews
            features.append(reviews[0:seq_length])
        elif len(reviews) < seq_length:
            # Calculate padding and prepend
            difference = seq_length - len(reviews)
            arr = [0 for i in range(difference)]
            arr.extend(reviews)
            features.append(arr)
        else:
            features.append(reviews)

    return np.array(features)


def create_traing_test_val_set(reviews_ndarray, labels_ndarray, train_frac=0.8):
    """
    Create Train/Validation/Test sets

    :arguments:
    ----------------------------------------
        reviews_ndarray :   The encoded review text
        reviews_ndarray :   The encoded review text
        train_frac      :   Training data percentage
    :return:
     ---------------------------------------
        train_x     :   Training features
        train_y     :   Training targets
        val_x       :   Validation features
        val_y       :   Validation targets
        test_x      :   Test features
        test_y      :   Test targets
    """

    # Find the index to split
    split = int(len(reviews_ndarray) * train_frac)

    # Get the train split
    train_x = reviews_ndarray[:split, :]
    train_y = labels_ndarray[:split]

    # Get the remaining split
    remaining_x = reviews_ndarray[split:, :]
    remaining_y = labels_ndarray[split:]

    # Split the remaining data in half
    split = int(len(remaining_x) * 0.5)

    # Use first half for validation
    val_x = remaining_x[:split, :]
    val_y = remaining_y[:split]

    # Use 2nd half for Testing
    test_x = remaining_x[split:, :]
    test_y = remaining_y[split:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def create_batches(train_x, train_y, val_x, val_y, test_x, test_y, batch_size):
    """
    Create batches using torch's TensorDataset and DataLoader

    :arguments:
    ----------------------------------------
        train_x     :   Training features
        train_y     :   Training targets
        val_x       :   Validation features
        val_y       :   Validation targets
        test_x      :   Test features
        test_y      :   Test targets
        batch_size  :   Batch Size
    :return:
     ---------------------------------------
        train_loader   :   Training batch data loader
        val_loader     :   Validation batch data loader
        test_loader    :   Test batch data loader
    """

    # Create TensorDataset
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # Use the DataLoader. Also shuffle the data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train(model: SentimentRNN, train_loader, val_loader, batch_size=50, epochs=10, lr=0.001, clip=5, print_every=100):
    """
    The training loop for the model

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

    # Set the model to training mode
    model.train()

    # Define optimization process
    optimization = torch.optim.Adam(model.parameters(), lr=lr)

    # Define Loss function as Binary Cross Entropy
    error_func = nn.BCELoss()

    if torch.cuda.is_available():
        model.cuda()

    training_loss = []
    validation_loss = []

    # Loop through the epochs
    for i in range(epochs):

        # Initialize the hidden layers
        hidden = model.init_hidden(batch_size)

        # Loop though the batches [ Mini-Batch SGD ]
        for counter, (inputs, labels) in enumerate(train_loader):

            # Move the input and targets to GPU ( if available )
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            hidden = tuple([each.data for each in hidden])

            # Remove the gradients from the model
            model.zero_grad()

            # Forward Propagation
            output, hidden = model.forward(inputs, hidden)

            # Calculate the Loss
            loss = error_func(output.squeeze(), labels.float())

            # Backprop
            loss.backward()

            # Gradient clipping ( needed to avoid exploding gradients )
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimization.step()

            if counter % print_every == 0:
                # Calculate validation loss

                val_hc = model.init_hidden(batch_size)
                val_losses = []

                # Set the model to evaluation state
                model.eval()

                for inputs, labels in val_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_hc = tuple([each.data for each in val_hc])

                    # Move the input and targets to GPU ( if available )
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # Forward Propagation
                    output, val_hc = model.forward(inputs, val_hc)

                    val_loss = error_func(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                # Set the model to training mode again
                model.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
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


def test(model: SentimentRNN, test_loader, batch_size=50):
    """
    Predict using the test preprocessing

    :arguments:
    ----------------------------------------
        model       :  CharRNN class
        test_loader :  Test batch data loader
        batch_size  :  Batch Size
    :return:
    ---------------------------------------
       None

    """

    if torch.cuda.is_available():
        model.cuda()
        train_on_gpu = True

    # Set the model to training mode
    model.eval()

    # Define Loss function as Binary Cross Entropy
    error_func = nn.BCELoss()

    h = model.init_hidden(batch_size)
    test_losses = []

    # Set the model to evaluation state
    model.eval()

    num_correct = 0

    for inputs, labels in test_loader:
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # Move the input and targets to GPU ( if available )
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward Propagation
        output, h = model.forward(inputs, h)

        test_loss = error_func(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        predictions = torch.round(output.squeeze())

        # Compare predictions with true labels
        correct_tensor = predictions.eq(labels.float().view_as(predictions))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


def prediction(model: SentimentRNN, input):
    # Pre-Process the text
    input = input.lower()
    input = ''.join([c for c in input if c not in punctuation])

    # Convert to list of words
    input = input.split()

    # Encode the text
    input_encoded = [[model.word2int[word] for word in input]]

    # Apply padding
    features = pad_features(input_encoded, 200)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = model.init_hidden(batch_size)

    if torch.cuda.is_available():
        feature_tensor = feature_tensor.cuda()

    # initialize hidden state
    h = model.init_hidden(batch_size)

    # get the output from the model
    output, h = model(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())

    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if pred.item() == 1:
        return "Positive Review"
    else:
        return "Negative Review"


if __name__ == '__main__':

    batch_size = 50

    reviews_encoded, labels_encoded, word2int = pre_process()
    reviews_encoded, labels_ndarray = process_reviews(reviews_encoded, labels_encoded)
    reviews_ndarray = pad_features(reviews_encoded, 200)

    train_x, train_y, val_x, val_y, test_x, test_y = create_traing_test_val_set(reviews_ndarray, labels_ndarray)

    train_loader, val_loader, test_loader = create_batches(train_x, train_y, val_x, val_y, test_x, test_y, batch_size)

    mode = "PREDICTION"  # TRAIN/TEST/PREDICTION

    if mode == "TRAIN":
        # Instantiate the model w/ hyperparams
        output_size = 1
        embedding_dim = 512
        n_hidden = 256
        n_layers = 2

        model = SentimentRNN(word2int, embedding_dim, n_hidden, n_layers)

        print(model)

        train(model, train_loader, val_loader, batch_size=batch_size, epochs=4)
        # Save the model
        model_name = 'model/rnn_sentiment_4_epoch.net'

        checkpoint = {'n_hidden': model.n_hidden,
                      'n_layers': model.n_layers,
                      'state_dict': model.state_dict(),
                      'word2int': model.word2int,
                      'embedding_dim': model.embedding_dim
                      }

        with open(model_name, 'wb') as f:
            torch.save(checkpoint, f)
    elif mode == "PREDICTION":
        # Open the model checkpoint
        with open('model/rnn_sentiment_4_epoch.net', 'rb') as f:
            checkpoint = torch.load(f)
        # Initialize the CharRNN class
        model = SentimentRNN(checkpoint['word2int'], checkpoint['embedding_dim'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        # Load the trained weights and biases
        model.load_state_dict(checkpoint['state_dict'])

        positive = "This movie had the best acting and the dialogue was so good. I loved it."
        negative = "The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow."

        inputs = [positive, negative]
        for input in inputs:
            print(prediction(model, input))

    elif mode == "TEST":
        # Open the model checkpoint
        with open('model/rnn_sentiment_4_epoch.net', 'rb') as f:
            checkpoint = torch.load(f)
        # Initialize the CharRNN class
        model = SentimentRNN(checkpoint['word2int'], checkpoint['embedding_dim'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        # Load the trained weights and biases
        model.load_state_dict(checkpoint['state_dict'])

        test(model, test_loader)
