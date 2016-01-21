import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork
from lstm_network.utils import decode, encode


input_data = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam"

sequence_length = 50  # 'length' of memory
learning_rate = 0.1
data_set = set(input_data + str.join("", [chr(x) for x in range(90 - 18)]))
memory_size = len(data_set)

data_to_int = {dat: i for i, dat in enumerate(data_set)}
int_to_data = {i: dat for i, dat in enumerate(data_set)}

encoded_data = []
for t in xrange(len(input_data)):
    # encode character in 1-of-k representation
    encoded_data.append(encode(input_data[t], data_to_int, len(data_set)))


lstm = LSTMNetwork()
lstm.int_to_data = int_to_data

lstm.populate(data_set, layer_sizes=[memory_size])

lstm.train(encoded_data, 5, iterations=20000, learning_rate=learning_rate)


#print("result: " + str(lstm.feedforward(np.array([1]))))
#print("result: " + str(lstm.feedforward(np.array([1]))))

while True:
    test_data_str = raw_input("Enter data to be predicted by the LSTM Network (char of alphabet: " +
                          str(data_set) + "):\n")
    if not len(test_data_str):
        break
    num = int(raw_input("Enter amount of characters to predict."))
    inp = np.zeros((len(data_set), 1))
    inp[data_to_int[test_data_str]] = 1
    results = lstm.predict(inp, num)
    out = ""
    for i in range(num):
        out += decode(results[i], int_to_data)
    print("Result: " + out)
