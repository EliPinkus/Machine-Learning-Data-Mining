# Solution set for CS 155 Set 6, 2017
# Authors: Suraj Nair, Sid Murching

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from P3CHelpers import *
import sys

def get_word_repr(word_to_index, word):
    """
    Returns one-hot-encoded feature representation of the specified word given
    a dictionary mapping words to their one-hot-encoded index.

    Arguments:
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        word:          Word whose feature representation we wish to compute.

    Returns:
        feature_representation:     Feature representation of the passed-in word.
    """
    unique_words = word_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_words))
    feature_representation[word_to_index[word]] = 1
    return feature_representation    

def generate_traindata(word_list, word_to_index, window_size=4):
    """
    Generates training data for Skipgram model.

    Arguments:
        word_list:     Sequential list of words (strings).
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        window_size:   Size of Skipgram window. Defaults to 2 
                       (use the default value when running your code).

    Returns:
        (trainX, trainY):     A pair of matrices (trainX, trainY) containing training 
                              points (one-hot-encoded vectors) and their corresponding labels
                              (also one-hot-encoded vectors)

    """
    trainX = []
    trainY = []
    vocab_size = len(word_to_index)
    for i in range(len(word_list)):
        # Extracts the words at each spot
        curr_word = word_list[i]
        # Loop over window of words near index i (index of current word)
        # Add pairs (x = curr_word, y = window_word) to our training data
        # for each window_word in the window.
        for j in range(1, window_size):
            ahead_idx = i + j
            behind_idx = i - j
            if ahead_idx < len(word_list):
                ahead_word = word_list[ahead_idx]
                y = get_word_repr(word_to_index, ahead_word)
                x = get_word_repr(word_to_index, curr_word)
                trainX.append(x)
                trainY.append(y)
            if behind_idx > 0:
                behind_word = word_list[behind_idx]
                y = get_word_repr(word_to_index, behind_word)
                x = get_word_repr(word_to_index, curr_word)
                trainX.append(x)
                trainY.append(y)
    return np.array(trainX), np.array(trainY)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python P3C.py <path_to_textfile> <num_latent_factors>")
        sys.exit(1)

    filename = sys.argv[1]
    num_latent_factors = int(sys.argv[2])
    
    sample_text = load_word_list(filename)

    # Create word dictionary
    word_to_index = generate_onehot_dict(sample_text)
    print("Textfile contains %s unique words"%len(word_to_index))
    # Create training data
    trainX, trainY = generate_traindata(sample_text, word_to_index)
    # Build our model
    vocab_size = len(word_to_index)
    model = Sequential()
    # <hidden_layer> contains our latent factors (vector representation of each word)	
    hidden_layer = Dense(num_latent_factors, input_dim = vocab_size)
    model.add(hidden_layer)
    # <output_layer> transforms the outputs of <hidden_layer> into a vector of size <vocab_size>.
    output_layer = Dense(vocab_size)
    model.add(output_layer) 	
    model.add(Activation('softmax'))

    # Compile and fit our model
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY, batch_size = 100, nb_epoch = 50)
    weights, biases = hidden_layer.get_weights()
    print("Hidden layer weight matrix shape: ", weights.shape)
    output_weights, output_biases = output_layer.get_weights()
    print("Output layer weight matrix shape: ", output_weights.shape)

    # Find and print most similar pairs
    similar_pairs = most_similar_pairs(weights, word_to_index)
    for pair in similar_pairs[:30]:
        print(pair)
