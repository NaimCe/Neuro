import numpy as np
from lstm_network.LSTMNetwork import LSTMNetwork
from lstm_network.utils import decode, encode
from isa.DataStore import DataStore
import os
path_tag_list = [
    ("auch", "isa/src/auch1.pnm"),
    ("auch", "isa/src/auch2.pnm"),
    ("auch", "isa/src/auch3.pnm"),
    ("auch", "isa/src/auch4.pnm"),
    ("dein", "isa/src/dein1.pnm"),
    ("dein", "isa/src/dein2.pnm"),
    ("dein", "isa/src/dein3.pnm"),
    ("dein", "isa/src/dein4.pnm"),
    ("dir", "isa/src/dir1.pnm"),
    ("dir", "isa/src/dir2.pnm"),
    ("dir", "isa/src/dir3.pnm"),
    ("dir", "isa/src/dir4.pnm"),
    ("duper", "isa/src/duper.pnm"),
    ("es", "isa/src/es1.pnm"),
    ("es", "isa/src/es2.pnm"),
    ("es", "isa/src/es3.pnm"),
    ("es", "isa/src/es4.pnm"),
    ("geht", "isa/src/geht1.pnm"),
    ("geht", "isa/src/geht2.pnm"),
    ("geht", "isa/src/geht3.pnm"),
    ("geht", "isa/src/geht4.pnm"),
    ("hallo", "isa/src/hallo1.pnm"),
    ("hallo", "isa/src/hallo2.pnm"),
    ("hallo", "isa/src/hallo3.pnm"),
    ("hallo", "isa/src/hallo4.pnm"),
    ("ignatz", "isa/src/ignatz1.pnm"),
    ("ignatz", "isa/src/ignatz2.pnm"),
    ("isa", "isa/src/ISA1.pnm"),
    ("isa", "isa/src/ISA2.pnm"),
    ("ist", "isa/src/ist1.pnm"),
    ("ist", "isa/src/ist2.pnm"),
    ("ist", "isa/src/ist3.pnm"),
    ("mein", "isa/src/mein1.pnm"),
    ("mein", "isa/src/mein2.pnm"),
    ("mein", "isa/src/mein3.pnm"),
    ("mir", "isa/src/mir1.pnm"),
    ("mir", "isa/src/mir2.pnm"),
    ("mir", "isa/src/mir3.pnm"),
    ("name", "isa/src/name1.pnm"),
    ("name", "isa/src/name2.pnm"),
    ("name", "isa/src/name3.pnm"),
    ("name", "isa/src/name4.pnm"),
    ("super", "isa/src/super1.pnm"),
    ("super", "isa/src/super2.pnm"),
    ("super", "isa/src/super3.pnm"),
    ("und", "isa/src/und1.pnm"),
    ("und", "isa/src/und2.pnm"),
    ("und", "isa/src/und3.pnm"),
    ("wie", "isa/src/wie1.pnm"),
    ("yasin", "isa/src/yasin1.pnm"),
    ("yasin", "isa/src/yasin2.pnm")
]
data_store = DataStore(path_tag_list, out_size=0)
samples = data_store.get_samples()

sequence_length = 20  # 'length' of memory
learning_rate = 0.1
data_set = data_store.tag_set
#data_set = data_set.union(set([(x + 256) for x in range(abs(memory_size - len(set(data_set))))]))
memory_size = 600 #np.shape(samples[0]["target"])[0]
in_size = np.shape(samples[0]["inputs"][0])[0]
out_size = np.shape(samples[0]["target"])[0]

#print("target: " + str(np.shape(samples[0]["target"])))
#print("inputs: " + str(np.shape(samples[0]["inputs"][0])))

"""ids = range(len(input_data) - 1)

random_samples = []
for i in range(200):
    rand_num = random.randint(0, len(ids)-1)
    rand_id = ids.pop(rand_num)
    random_samples.append(input_data[rand_id])
random_samples = [ord(x) for x in ''.join(random_samples)]
#print(random_samples)"""
encoded_data = []

#print(str(encoded_data[-1]))
lstm = LSTMNetwork()
lstm.use_output_layer = False

#print(str(len(data_set)) + "\nmemsize " + str(memory_size) + "\nmaxInt")

lstm.populate(in_size, out_size, layer_sizes=[memory_size, memory_size])
lstm.load("lstm_isa_save_bk")

lstm._int_to_data = data_store.tag_to_data
lstm._data_to_int = data_store.data_to_tag
lstm.decode = data_store.decode
lstm.encode = data_store.encode

lstm.train(samples, seq_length=sequence_length, iterations=20000, target_loss=0.08, learning_rate=learning_rate, save_dir=os.path.join(os.getcwd(), "lstm_isa_save"))

#print("result: " + str(lstm.feedforward(np.array([1]))))
#print("result: " + str(lstm.feedforward(np.array([1]))))


for tag, path in path_tag_list:
    test_sample = DataStore.load(path)
    prediction = lstm.predict(test_sample, seq_length=sequence_length)
    print("Prediction for '" + os.path.basename(path) + "':")
    print("Expected: " + tag)
    print("All: " + "[ " + ', '.join([data_store.decode(p) for p in prediction]) + " ]")
    print("Average: " + data_store.decode(np.average(prediction)))
    print("Last: " + data_store.decode(prediction[-1]))