import json
import torch
from torch.utils import data

padded_sequence_list = json.load( open('../processed_data/vectors/padded_word_sequences_1500.json', 'r'))
emotion_vectors_list = json.load( open('../processed_data/vectors/emotion_score_dict_20_chunks.json', 'r'))
binary_labels = json.load( open('../processed_data/vectors/labels_binary_dict.json', 'r') )


class Dataset(data.Dataset):
    def __init__(self, imdb_id_list):
        self.imdb_id_list = imdb_id_list

    def __len__(self):
        return len(self.imdb_id_list)

    def __getitem__(self, index):
        imdb_id = str(self.imdb_id_list[index])
        word_sequence = torch.LongTensor(padded_sequence_list[imdb_id])
        emotion_vector = torch.FloatTensor(emotion_vectors_list[imdb_id])
        one_hot_tags = torch.FloatTensor(binary_labels[imdb_id])

        return word_sequence, emotion_vector, one_hot_tags, imdb_id