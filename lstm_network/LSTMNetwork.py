import os
import numpy as np
from LSTMLayer import LSTMLayer
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
        self.lstm_layer = None
        self.encode = None
        self.decode = None
        self.use_output_layer = False
        self.use_output_slicing = False

    def populate(self, in_size, out_size, layer_sizes=None):
        layer_sizes = layer_sizes or self.layers_spec or []

        self.layers = []

        self.lstm_layer = LSTMLayer(
            in_size=in_size,
            out_size=out_size,
            memory_size=layer_sizes[0],
            use_output_layer=self.use_output_layer)
        self.populated = True

    def feedforward(self, inputs, caching_depth=1):
        if not self.populated:
            raise Exception("LSTM network must be populated first (Have a look at method LSTMNetwork.populate)!")

        hidden_outputs = []
        i = 0
        # get initial state of all lstm layers
        for data_in in inputs:
            i += 1
            # feed forward through all layers
            # get output of lstm network
            hidden_outputs.append(self.lstm_layer.feed(np.array([data_in]).T, caching_depth))
            # get updated state from all lstm layers

        return hidden_outputs

    def learn(self, targets, learning_rate):
        losses = self.lstm_layer.learn(targets, learning_rate)
        return losses

    def train(self, inputs_list, seq_length, iterations=100, target_loss=0, learning_rate=0.1, save_dir="lstm_save", iteration_start=0, dry_run=False):
        loss_diff = 0
        self.lstm_layer.visualize("visualize")
        raw_input("Press Enter to proceed...")
        for i in xrange(iteration_start, iterations):
            output_string = ""
            input_string = ""
            loss_list = []
            sample_loss_list = []
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
                self.lstm_layer.clear_cache()
            iteration_loss = np.average(loss_list)

            if not i % 10 or iteration_loss < target_loss:
                if dry_run:
                    self.load(os.path.join(os.getcwd(), save_dir))
                else:
                    self.lstm_layer.save(os.path.join(os.getcwd(), save_dir))
                self.lstm_layer.visualize("visualize")
                loss_diff -= iteration_loss
                if self.verbose:
                    print("\nIteration " + str(i) + " - learning rate: " + str(learning_rate) + "  ")
                    print("loss: " + str(iteration_loss) + "  "
                          + ("\\/" if loss_diff > 0 else ("--" if loss_diff == 0 else "/\\"))
                          + " " + str(loss_diff)[1:])
                    print("target: " + input_string.replace("\n", "\\n"))
                    print("out:    " + output_string.replace("\n", "\\n"))

                loss_diff = iteration_loss
                if iteration_loss < target_loss:
                    print("Target loss reached! Training finished.")
                    self.lstm_layer.clear_cache()
                    return


    def roll_weights(self):
        self.populate(self.chars, [self.lstm_layer.size])

    def predict(self, inputs, seq_length=1):
        outputs = self.feedforward(inputs, seq_length)
        return outputs


    def load(self, path):
        self.lstm_layer.load(path)

