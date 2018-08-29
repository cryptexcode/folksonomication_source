# Implementation of "Folksonomication: Predicting Tags for Movies from Plot Synopses using Emotion Flow Encoded Neural Network"

[Project Home and Live Demo] (http://ritual.uh.edu/folksonomication-2018) 

## Contributors
- Sudipta Kar
- Suraj Maharjan
- Thamar Solorio




## Dependencies
The code is written in Python 3.
- [PyTorch] (http://pytorch.org) (Our used version is <b>0.4.0a0+396637c</b>)
- [JSON] (https://docs.python.org/3.1/library/json.html)
- [Joblib] (http://pypi.python.org/pypi/joblib)
- [Pandas] (https://pandas.pydata.org)
- [Jupyter Notebook] (http://jupyter.org)
- CUDA 8.0

Optional
- [Tensorflow] (https://www.tensorflow.org) (If you want to monitor the experiment with Tensorboard)

## Resource Map
```
├── data
├── LICENSE
├── processed_data
│   ├── all_sequence_dict.json
│   ├── class_weights.json
│   ├── class_weights_sk.json
│   ├── emotion_lexicons_dict.pkl
│   ├── idx2word_no_process.json
│   ├── index_to_tag.json
│   ├── tag_to_index.json
│   ├── test_sequences_dict.json
│   ├── test_sequences_list.json
│   ├── train_sequences_dict.json
│   ├── train_sequences_list.json
│   ├── vectors
│   │   ├── emotion_score_dict_20_chunks.json
│   │   ├── labels_binary_dict.json
│   │   └── padded_word_sequences_1500.json
│   ├── vocab_5k_no_process.json
│   └── word2idx_no_process.json
├── README.md
└── source
    ├── Dataset.py
    ├── misc.py
    ├── models.py
    ├── notebooks
    │   └── Prepare Data.ipynb
    ├── outputs
    │   └── 4_shorter_rand_emb_rmsprop
    │       ├── best.pth
    │       └── logs
    │           ├── events.out.tfevents.1529639175.ritual-nlpdigits
    │           └── events.out.tfevents.1529640895.ritual-nlpdigits
    ├── predict_tags.py
    ├── report.py
    ├── tf_logger.py
    ├── TorchHelper.py
    ├── t.py
    └── train.py
```



## Usage
1. Download the [MPST Corpus] (http://ritual.uh.edu/mpst-2018).
2. Unzip the data and put the <b>MPST</b> directory inside the <b>data</b> directory.
3. Use the data processor notebook located at <b>source/notebooks/Prepare Data.ipynb</b> to prepare the data for the model.
    The processed data would be dumped inside <b>processed_data</b> directory.
4. After completing the processing, <b>processed_data</b> directory should look like below.

```
├── processed_data
│   ├── all_sequence_dict.json
│   ├── class_weights.json
│   ├── class_weights_sk.json
│   ├── emotion_lexicons_dict.pkl
│   ├── idx2word_no_process.json
│   ├── index_to_tag.json
│   ├── tag_to_index.json
│   ├── test_sequences_dict.json
│   ├── test_sequences_list.json
│   ├── train_sequences_dict.json
│   ├── train_sequences_list.json
│   ├── vectors
│   │   ├── emotion_score_dict_20_chunks.json
│   │   ├── labels_binary_dict.json
│   │   └── padded_word_sequences_1500.json
│   ├── vocab_5k_no_process.json
│   └── word2idx_no_process.json
```

5.
6. 