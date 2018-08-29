import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

import json
import time
import numpy as np
from torch import optim
from torch.utils import data
from torch.nn import functional as F

from TorchHelper import TorchHelper
from tf_logger import Logger
from Dataset import Dataset
from report import EvaluationReports
from models import EmotionFlowModel

torch_helper = TorchHelper()
EvaluationReports = EvaluationReports()

# ----------------------------------------------------------------------------
# Model CONFIGURATION
# ----------------------------------------------------------------------------
vocab_size = 5000
embedding_dim = 300
text_sequence_dim = 1500
batch_size = 32
max_epochs = 300
emotion_sequence_dim = 10
emotion_sequence_length = 20
target_classes = 71
top_n_list = [1, 3, 5, 8, 10]

# Creates the directory where the results, logs, and models will be dumped.
output_dir_path = 'outputs/model_1/'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
logger = Logger(output_dir_path + 'logs')

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
# Load Data using the data generator
root_data_path = '../data/MPST/'
processed_data_path = '../processed_data/'
imdb_id_list = open(root_data_path + '/final_plots_wiki_imdb_combined/imdb_id_list.txt').read().split('\n')

# Loads a dictionary like {1:murder, 2: violence ....}
index_to_tag_dict = json.load(open(processed_data_path + 'index_to_tag.json', 'r'))
class_weights = torch.FloatTensor(json.load(open(processed_data_path + '/class_weights_sk.json', 'r')))

# Load Partition Information
partition_dict = json.load(open(root_data_path + 'partition.json', 'r'))
train_id_list, val_id_list, test_id_list = partition_dict['train'], partition_dict['validation'], \
                                                          partition_dict['test']

print('Data Split: Train (%d), Dev (%d), Test (%d)' %(len(train_id_list), len(val_id_list), len(test_id_list)) )

# Create the data loaders for all splits
training_set = Dataset(train_id_list)
train_generator = data.DataLoader(training_set, batch_size, shuffle=True)

validation_set = Dataset(val_id_list)
val_generator = data.DataLoader(validation_set, batch_size)

test_set = Dataset(test_id_list)
test_generator = data.DataLoader(test_set, batch_size)


# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model():
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:
    """
    model = EmotionFlowModel(text_sequence_dim, emotion_sequence_dim, embedding_dim, target_classes, vocab_size,
                 batch_size, emotion_sequence_length, class_weights)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, optimizer):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    start_time = time.time()

    model.train()

    total_loss = 0

    for batch_idx, (word_sequence, emotion_vector, target_one_hot_vector, imdb_id) in enumerate(train_generator):

        if torch.cuda.is_available():
            word_sequence, emotion_vector, target_one_hot_vector = word_sequence.cuda(), emotion_vector.cuda(), \
                                                                   target_one_hot_vector.cuda()

        optimizer.zero_grad()

        model_output = model([word_sequence, emotion_vector])

        loss = F.kl_div(torch.log(model_output), target_one_hot_vector)
        total_loss += loss.data.item()
        loss.backward(retain_graph=True)

        optimizer.step()

        torch_helper.show_progress(batch_idx+1, np.ceil(len(train_id_list)/batch_size), start_time, loss.data.item())
        # break
    # print('Total training loss', total_loss)


# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, data_generator):
    """

    :param model:
    :param data_generator:
    :return: average loss {float},
    """
    predicted_probabilities_list = []
    target_tags_list = []
    imdb_ids_list = []
    total_loss = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (word_sequence, emotion_vector, target_one_hot_vector, imdb_id) in enumerate(data_generator):

            if torch.cuda.is_available():
                word_sequence, emotion_vector = word_sequence.cuda(), emotion_vector.cuda()

            model_output = model([word_sequence, emotion_vector])

            total_loss += F.kl_div(torch.log(model_output), target_one_hot_vector.cuda()).data.item()

            if torch.cuda.is_available():
                predicted_probabilities_list.append(model_output)
                target_tags_list.append(target_one_hot_vector)
            else:
                predicted_probabilities_list.append(model_output.numpy().tolist())
                target_tags_list.append(target_one_hot_vector.squeeze().numpy().tolist())

            imdb_ids_list.append(imdb_id)

    predicted_probabilities_list = torch.cat(predicted_probabilities_list, dim=0)
    target_tags_list = torch.cat(target_tags_list, dim=0)

    avg_loss = total_loss / len(data_generator.dataset)

    results = EvaluationReports.get_f1_and_tl(np.array(predicted_probabilities_list).squeeze(),
                                              np.array(target_tags_list).squeeze(),
                                              top_n_list)

    return avg_loss, results


def training_loop():
    """

    :return:
    """
    model = create_model()

    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(max_epochs):
        print('[Epoch %d]' % (epoch + 1))

        train(model, optimizer)

        val_loss, val_results = evaluate(model, val_generator)
        train_loss, train_results = evaluate(model, train_generator)

        print('Training Loss %.5f, Validation Loss %.5f' % (train_loss, val_loss))

        for tn, t_rs, v_rs in zip(top_n_list, train_results, val_results):
            print('Top %d: Train F1: %.4f, Val F1: %.4f, Train TL: %d, Val TL %d' % (tn, t_rs[0], v_rs[0], t_rs[1], v_rs[1]))

        val_top_3_f1 = val_results[2][0]
        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, val_top_3_f1, epoch, 'max')

        #if is_best:
        #    json.dump(val_predictions, open(output_dir_path + 'validation_predictions.json', 'w'))

        print()

        # -------------------------------------------------------------
        # Tensorboard Logging
        # -------------------------------------------------------------
        info = {'training loss': train_loss,
                'validation loss': val_loss,
                'Top 1: Train F1': train_results[0][0],
                'Top 1: Train TL': train_results[0][1],
                'Top 1: Val F1'  : val_results[0][0],
                'Top 1: Val TL'  : val_results[0][1],
                'Top 3: Train F1': train_results[2][0],
                'Top 3: Train TL': train_results[2][1],
                'Top 3: Val F1': val_results[2][0],
                'Top 3: Val TL': val_results[2][1],
                'Top 5: Train F1': train_results[3][0],
                'Top 5: Train TL': train_results[3][1],
                'Top 5: Val F1': val_results[3][0],
                'Top 5: Val TL': val_results[3][1],
                }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # Log values and gradients of the model parameters
        for tag, value in model.named_parameters():
            if value.grad is not None:
                tag = tag.replace('.', '/')

                if torch.cuda.is_available():
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
                else:
                    logger.histo_summary(tag, value.data.numpy(), epoch + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.numpy(), epoch + 1)


def test():
    model = create_model()

    torch_helper.load_saved_model(model, output_dir_path + 'best.pth')

    loss, result = evaluate(model, test_generator)
    print(loss, result)

    loss, result = evaluate(model, val_generator)
    print(loss, result)

    loss, result = evaluate(model, train_generator)
    print(loss, result)


if __name__ == '__main__':
    training_loop()
    test()
