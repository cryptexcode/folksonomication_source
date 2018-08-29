from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error
import numpy as np
import pandas as pd
# import config


class EvaluationReports:

    @staticmethod
    def get_top_n_predictions(pred_prob_matrix, top_n):
        """

        >>> er = EvaluationReports()
        >>> inp = np.array([[1,2,4,5,7,4,3,7], [12,26,421,25,47,74,32,73], [567,3,87,23,54,12,65,12]])
        >>> er.get_top_n_predictions(inp)

        :param top_n:
        :param pred_prob_matrix:
        :return:
        """
        print(pred_prob_matrix.shape)
        sorted_idx = np.argsort(pred_prob_matrix)
        for i in range(sorted_idx.shape[0]):
            sorted_idx[i] = sorted_idx[i][::-1]

        outputs = []
        unique_predictions = []

        for tn in top_n:
            sliced = sorted_idx[:, :tn]
            unique_set = set(sliced.flatten())

            one_hot = np.zeros(pred_prob_matrix.shape)
            for idx in range(sliced.shape[0]):
                one_hot[idx][sliced[idx]] = 1
            outputs.append(one_hot)
            unique_predictions.append(len(unique_set))

        return outputs, unique_predictions

    def get_f1_and_tl(self, y_pred, y_true, top_n):
        y_pred = y_pred
        y_true = y_true
        top_preds, learned_tags = self.get_top_n_predictions(y_pred, top_n)
        output = []

        for tp, lt in zip(top_preds, learned_tags):
            micro_f1 = f1_score(y_true, tp, average='micro')
            output.append([round(micro_f1, 5), round(lt, 5)])

        return output
