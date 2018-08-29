import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.append(os.path.abspath('../'))
print(os.path.abspath(''))
import torch

torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

import json
import numpy as np
from torch.utils import data
from collections import Counter
import time
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from TorchHelper import TorchHelper
from sklearn.metrics import accuracy_score
from tf_logger import Logger
from Dataset import Dataset
from report import EvaluationReports
from models import EmotionFlowModel

torch_helper = TorchHelper()
EvaluationReports = EvaluationReports()
print('gpu', torch.cuda.is_available())
# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
vocab_size = 5000
embedding_dim = 300
max_sequence_length = 1500
batch_size = 32
max_epochs = 200
emotion_sequence_dim = 10
emotion_sequence_length = 20
target_classes = 71
top_n_list = [1, 3, 5, 8, 10]

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
# Load Data using the data generator
root_data_path = '../data/MPST/'
processed_data_path = '../processed_data/'
imdb_id_list = open(root_data_path + '/final_plots_wiki_imdb_combined/imdb_id_list.txt').read().split('\n')

# Load Partition Data
partition_dict = json.load(open(root_data_path + 'partition.json', 'r'))
train_id_list, val_id_list, test_id_list = partition_dict['train'], partition_dict['validation'], \
                                                          partition_dict['test']
training_set = Dataset(train_id_list)
validation_set = Dataset(val_id_list)
test_set = Dataset(test_id_list)

#print(test_id_list)
index_to_tag_dict = json.load(open(processed_data_path + 'index_to_tag.json', 'r'))
class_weights = json.load(open(processed_data_path + '/class_weights.json', 'r'))

print('Data Split: Train (%d), Dev (%d), Test (%d)' %(len(train_id_list), len(val_id_list), len(test_id_list)) )

params = {'batch_size': batch_size, 'shuffle': False}

#train_generator = data.DataLoader(training_set, batch_size, shuffle=False)
val_generator = data.DataLoader(validation_set, batch_size=1)
#test_generator = data.DataLoader(test_set, batch_size=1)

dumped_model_path = '../outputs/2/best.pth'


def get_tags(model, word_sequence, emotion_vector, top_n):
    model.eval()

    model_output = model([word_sequence, emotion_vector])
    probabilities, indexes = torch.sort(model_output, 1)

    for i in range(70, 70-top_n, -1):
        print(indexes[0][i].item(), index_to_tag_dict[str(indexes[0][i].item())], probabilities[0][i].item())


def load_model():
    m = EmotionFlowModel(embedding_dim, vocab_size, max_sequence_length, emotion_sequence_dim, emotion_sequence_length,
                             target_classes, class_weights)
    m.load_state_dict(torch.load(dumped_model_path, map_location={'cuda:0': 'cpu'})['model_state'])
    return m


if __name__ == '__main__':
    model = load_model()

    for i, data in enumerate(val_generator):
        word_sequence = data[0]
        emotion_sequence = data[1]
        print(data[3])
        get_tags(model, word_sequence, emotion_sequence, 10)
        print()
        if i>10:
            break