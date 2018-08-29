import joblib
import numpy as np

old_seq = joblib.load('/home/sk/SK/Works/NarrativeAnalysis/experiments/classification/features/sequence/sequences_docs.pkl')
old_seq = np.array(old_seq)

id_list = open('/home/sk/SK/Works/NarrativeAnalysis/experiments/classification/data/data_exp/final_plots_wiki_imdb_combined/imdb_id_list.txt').read().split('\n')

print(old_seq.shape)
print(len(id_list))

luther_seq = old_seq[11869]

print(len(luther_seq))

luther_txt = open('/home/sk/SK/Works/NarrativeAnalysis/paper/coling_2018/Folksonomication-Coling-2018/data/MPST/final_plots_wiki_imdb_combined/cleaned/tt0309820.txt').read().split()

print(len(luther_txt))

print(luther_txt)

print(luther_seq)