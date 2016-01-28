import os
import numpy as np
from LSTMLayer import LSTMLayer
from utils import encode, decode, decode_sequence
import math
import time

class LSTMNetwork:

    def __init__(self):
        self.layers_spec = None
        self.layers = None
        self.populated = False
        self.trained = False
        self.chars = None
        self._int_to_data = None
        self._data_to_int = None
        self.state_history = None
        self.learning_rate = 0.001
        self.verbose = True
        self.lstm_layer_in = None
        self.lstm_layer_out = None
        self.encode = None
        self.decode = None
        self.use_output_layer = False
        self.use_output_slicing = False

    def populate(self, in_size, out_size, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []

        self.layers = []

        self.lstm_layer_in = LSTMLayer(
            in_size=in_size,
            out_size=out_size,
            memory_size=layer_sizes[0],
            use_output_layer=True)
        self.lstm_layer_out = LSTMLayer(
            in_size=layer_sizes[0],
            out_size=out_size,
            memory_size=layer_sizes[1],
            use_output_layer=True)

        self.populated = True

    def feedforward(self, inputs, caching_depth=1):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        hidden_outputs = []
        i = 0
        #print(str(np.shape(inputs)))
        # get initial state of all lstm layers
        for data_in in inputs:
            i += 1
            # feed forward through all layers
            # get output of lstm network
            hidden_outputs.append(self.lstm_layer_in.feed(np.array([data_in]).T, caching_depth))
            # get updated state from all lstm layers
        #print("feeded: " + str(i) + "/" + str(len(inputs)))
        #outputs = []
        #for hidden_out in hidden_outputs:
        #    outputs.append(self.lstm_layer_out.feed(hidden_out, caching_depth))

        return hidden_outputs

    def learn(self, targets, learning_rate):
        #print("targets: " + str(np.shape(targets)))
        losses = self.lstm_layer_in.learn(targets, learning_rate)
        #deltas = []
        #cache = self.lstm_layer_in.first_cache
        #while not cache.is_last_cache:
        #    cache = cache.successor
        #    deltas.append(cache.loss_input)
        #print("deltas: " + str(np.shape(deltas[0])))
        #self.lstm_layer_in.learn(deltas, learning_rate)
        return losses

    def train(self, inputs_list, seq_length, iterations=100, target_loss=0, learning_rate=0.1, save_dir="lstm_save", iteration_start=0):
        #print("inputs_list: " + str(inputs_list))
        loss_diff = 0
        self.lstm_layer_in.visualize("visualize/lstm1/")
        self.lstm_layer_out.visualize("visualize/lstm2/")
        raw_input("Press Enter to proceed...")
        iteration_loss = []
        for i in xrange(iteration_start, iterations):
            output_string = ""
            input_string = ""
            loss_list = []
            sample_loss_list = []
            iteration_loss = 0
            for sample in inputs_list:
                snippet_length = min(seq_length, len(sample))
                for inputs in [sample["inputs"][j:j+snippet_length] for j in xrange(0, len(sample["inputs"]) - 1, seq_length)]:
                    targets = []
                    for j in range((np.shape(inputs)[0])):
                        targets.append(np.array([sample["target"]]).T)
                    outputs = self.feedforward(inputs, seq_length)
                    loss = self.learn(targets, learning_rate)
                    output_string += " " + self.decode(outputs[-1])
                    input_string += " " + self.decode(targets[-1])
                    sample_loss_list.append(loss)
                loss_list.append(sample_loss_list[-1])
                self.lstm_layer_in.clear_cache()
                #self.lstm_layer_out.clear_cache()
            iteration_loss = np.average(loss_list)

            if not i % 10 or iteration_loss < target_loss:
                self.lstm_layer_in.save(os.path.join(os.getcwd(), save_dir, "lstm_in"))
                self.lstm_layer_out.save(os.path.join(os.getcwd(), save_dir, "lstm_out"))
                self.lstm_layer_in.visualize("visualize/lstm1")
                self.lstm_layer_out.visualize("visualize/lstm2")
                loss_diff -= iteration_loss
                if self.verbose:
                    print("\nIteration " + str(i) + " - learning rate: " + str(learning_rate) + "  ")
                    print("loss: " + str(iteration_loss) + "  "
                          + ("\\/" if loss_diff > 0 else ("--" if loss_diff == 0 else "/\\"))
                          + " " + str(loss_diff)[1:])
                    print("in: " + input_string.replace("\n", "\\n"))
                    print("out: " + output_string.replace("\n", "\\n"))

                loss_diff = iteration_loss
                #time.sleep(1)
                if iteration_loss < target_loss:
                    return


    def roll_weights(self):
        self.populate(self.chars, [self.lstm_layer.size])

    def predict(self, inputs, seq_length=1):
        outputs = self.feedforward(inputs, seq_length)
        return outputs


    def load(self, path):
        self.lstm_layer_in.load(os.path.join(path, "lstm_in"))

