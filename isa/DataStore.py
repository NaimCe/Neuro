import numpy as np
from scipy.misc import imresize as sp_imresize

class DataStore(object):
    def __init__(self, tag_path_list):
        self.tag_set = None
        self.data_list = []
        self.tag_to_data = None
        self.data_to_tag = None

        self.generate_data(tag_path_list)

    @classmethod
    def read_pnm(cls, path_to_pnm_file):
        img_lines = [line.rstrip('\n') for line in open(path_to_pnm_file)]
        dimensions = img_lines[1].split(" ")
        return int(dimensions[0]), int(dimensions[1]), img_lines[3]

    @classmethod
    def load(self, path_to_pnm):
        width, height, img_str = DataStore.read_pnm(path_to_pnm)
        img_data = np.array([int(num) for num in filter(None, img_str.split(" "))]).reshape((height, width))
        img_data /= np.max(img_data)

        return sp_imresize(img_data, (width / 4, 30))

    """def add_img(self, tag, path):
        self.path_list[tag] = path"""

    def length(self):
        return len(self.tag_set)

    def decode(self, data):
        #print(str(np.argmax(data)) + "/" + str(len(self.tag_set)) + "\n" + str(data))
        return self.data_to_tag[np.argmax(data)]

    def encode(self, tag):
        tag_id = self.tag_to_data[tag]
        encoded = np.zeros(len(self.tag_set))
        encoded[tag_id] = 1
        return encoded

    def get_samples(self):
        return self.data_list

    def generate_data(self, tag_path_list):
        self.tag_set = set([tag for tag, _ in tag_path_list])
        self.tag_to_data = {tag: idx for idx, tag in enumerate(self.tag_set)}
        self.data_to_tag = {idx: tag for idx, tag in enumerate(self.tag_set)}

        for tag, path in tag_path_list:
            self.data_list.append({
                "target": self.encode(tag),
                "inputs": self.load(path)
            })
