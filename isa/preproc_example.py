from DataStore import DataStore

data_store = DataStore([
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
])

samples = data_store.get_samples()

print(str(samples[0]))

class RecurrentNeuralNet():
    """Sample RNN class"""
    def train(self, input_list, target_list):
        pass


# feed to neural network like this:
nn = RecurrentNeuralNet()
# add more samples...
for sample in samples:
    nn.train(sample["inputs"], sample["target"])




