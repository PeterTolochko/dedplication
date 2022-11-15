import spacy
import pandas as pd
from math import sqrt, pow, exp
import numpy as np


nlp = spacy.load("de_core_news_sm")

def squared_sum(x):
    return round(sqrt(sum([a*a for a in x])), 3)

def cos_similartiy(x, y):
    """ return cosine similarity between two lists """
    numerator = sum(a*b for a, b in zip(x,y))
    denominator = squared_sum(x) * squared_sum(y)
    return round(numerator / float(denominator), 3)


my_data = pd.read_csv("")

word_vectors = []

for i, j in enumerate(my_data.text):
    word_vectors.append(nlp(j).vector)

def get_sim_matrix(word_vectors):
    sim_matrix = np.zeros((len(word_vectors), len(word_vectors)))

    for i, vector_i in enumerate(word_vectors):
        for j, vector_j in enumerate(word_vectors):
            sim_matrix[i, j] = cos_similartiy(vector_i, vector_j)
    return sim_matrix

sim_matrix = get_sim_matrix(word_vectors)

sim_ids = []

for i in range(0, sim_matrix.shape[0]):
    for j in range(0, sim_matrix.shape[0]):
        if i < j:
            if sim_matrix[i, j] >= .995:
                sim_ids.append([i, j])


to_remove = [sim_ids[x][1] for x in range(0, len(sim_ids))]

my_data_dedup = my_data
my_data_dedup = my_data_dedup.drop(to_remove)

my_data_dedup.to_csv("")