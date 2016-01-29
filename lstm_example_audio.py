import numpy as np
import os
from lstm_network.LSTMNetwork import LSTMNetwork
from DataStore import DataStore

"""Configuration:"""

# List of tuples of tag name and according pnm file representing audio
path_tag_list = [
    ("auch", "src/auch1.pnm"),
    ("auch", "src/auch2.pnm"),
    ("auch", "src/auch3.pnm"),
    ("auch", "src/auch4.pnm"),
    ("dein", "src/dein1.pnm"),
    ("dein", "src/dein2.pnm"),
    ("dein", "src/dein3.pnm"),
    ("dein", "src/dein4.pnm"),
    ("dir", "src/dir1.pnm"),
    ("dir", "src/dir2.pnm"),
    ("dir", "src/dir3.pnm"),
    ("dir", "src/dir4.pnm"),
    ("duper", "src/duper.pnm"),
    ("es", "src/es1.pnm"),
    ("es", "src/es2.pnm"),
    ("es", "src/es3.pnm"),
    ("es", "src/es4.pnm"),
    ("geht", "src/geht1.pnm"),
    ("geht", "src/geht2.pnm"),
    ("geht", "src/geht3.pnm"),
    ("geht", "src/geht4.pnm"),
    ("hallo", "src/hallo1.pnm"),
    ("hallo", "src/hallo2.pnm"),
    ("hallo", "src/hallo3.pnm"),
    ("hallo", "src/hallo4.pnm"),
    ("ignatz", "src/ignatz1.pnm"),
    ("ignatz", "src/ignatz2.pnm"),
    ("isa", "src/ISA1.pnm"),
    ("isa", "src/ISA2.pnm"),
    ("ist", "src/ist1.pnm"),
    ("ist", "src/ist2.pnm"),
    ("ist", "src/ist3.pnm"),
    ("mein", "src/mein1.pnm"),
    ("mein", "src/mein2.pnm"),
    ("mein", "src/mein3.pnm"),
    ("mir", "src/mir1.pnm"),
    ("mir", "src/mir2.pnm"),
    ("mir", "src/mir3.pnm"),
    ("name", "src/name1.pnm"),
    ("name", "src/name2.pnm"),
    ("name", "src/name3.pnm"),
    ("name", "src/name4.pnm"),
    ("super", "src/super1.pnm"),
    ("super", "src/super2.pnm"),
    ("super", "src/super3.pnm"),
    ("und", "src/und1.pnm"),
    ("und", "src/und2.pnm"),
    ("und", "src/und3.pnm"),
    ("wie", "src/wie1.pnm"),
    ("yasin", "src/yasin1.pnm"),
    ("yasin", "src/yasin2.pnm")
]

# get instance of datastore for current dataset
data_store = DataStore(path_tag_list, out_size=0)
# retrieve list of samples from data store
samples = data_store.get_samples()

"""Training parameters"""
# Number of time steps for backpropagation in time. Also defines the packet size to train at once
sequence_length = 20  # 'length' of memory
# learning rate for backpropagation
learning_rate = 0.06
#size of hidden state of LSTM network
memory_size = 600
# input size for lstm network (retrieved from input data)
in_size = np.shape(samples[0]["inputs"][0])[0]
# output size for lstm network (retrieved from target output data)
out_size = np.shape(samples[0]["target"])[0]
# maximum iterations after which training will terminate
max_iterations = 20000
# target loss: if accomplished, training will be terminated
target_loss = 0.051
# directory to save lstm state to
save_dir = "lstm_isa_save"


# create instance of LSTMNetwork
lstm = LSTMNetwork()
# use output layer (necessary for independent hidden state and output sizes)
lstm.use_output_layer = True

# create actual lstm layers
lstm.populate(in_size, out_size, layer_sizes=[memory_size])
# load training state from directory (comment to start training over)
# lstm.load("lstm_isa_save")

#define decode/encode methods for training progress output
lstm.decode = data_store.decode
lstm.encode = data_store.encode

# train lstm
lstm.train(
    samples,
    seq_length=sequence_length,
    iterations=max_iterations,
    target_loss=target_loss,
    learning_rate=learning_rate,
    save_dir=os.path.join(os.getcwd(), save_dir))

# simple function to get the 5 most probable of a list of predictions (tags)
def top_5(tag_list):
    tag = np.sum(tag_list, axis=0)
    result = [data_store.decode(tag) + str(tag[np.argmax(tag)])]
    tag[np.argmax(tag)] = 0
    result.append(data_store.decode(tag) + str(tag[np.argmax(tag)]))
    tag[np.argmax(tag)] = 0
    result.append(data_store.decode(tag) + str(tag[np.argmax(tag)]))
    tag[np.argmax(tag)] = 0
    result.append(data_store.decode(tag) + str(tag[np.argmax(tag)]))
    tag[np.argmax(tag)] = 0
    result.append(data_store.decode(tag) + str(tag[np.argmax(tag)]))
    return result

# validation output - add extra validation set for more meaningful results.
match_counter = 0
for tag, path in path_tag_list:
    test_sample = DataStore.load(path)
    prediction = lstm.predict(test_sample, seq_length=sequence_length)
    lstm.lstm_layer_in.clear_cache()
    average = data_store.decode(np.sum(prediction, axis=0))
    last = data_store.decode(prediction[-1])
    combo = data_store.decode(np.sum(prediction[-5:], axis=0))
    print("\nPrediction for '" + os.path.basename(path) + "':")
    print("Expected: " + tag)
    print("All: " + "[ " + ', '.join([data_store.decode(p) + " (" + str(p[np.argmax(p)][0]) + ")" for p in prediction]) + " ]")
    print("Average: " + average)
    print("Last: " + last)
    print("Combo: " + combo)
    print("Top 5: " + ", ".join(top_5(prediction)))

    if tag in [average, last, combo]:
        match_counter += 1
print("\n" + str(match_counter) + "/" + str(len(path_tag_list)) + " matches!")
