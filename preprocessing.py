import bcolz
import numpy as np
from Project import project
import pickle
from torch import nn
import sys
import time
import torch

def transter_embed_file():
    words = [] # 定义单词
    idx = 0 
    word2idx = {} # 将单词映射为索引

    with open(project.embedding_dir, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1

    pickle.dump(words, open(project.words_dir, 'wb'))
    pickle.dump(word2idx, open(project.idx_dir, 'wb'))

# Test
transter_embed_file()

def get_datadict_vector(target_vocab):
    """
    为数据集中的词典指定相应的向量,最终形状应为
    (dataset’s vocabulary length, word vectors dimension)
    1. 如果单词在预训练词表中存在，就返回对应的向量
    2. 如果不存在，就返回随机产生的权重矩阵
    """
    vectors = bcolz.open(project.bcolz_dir)
    words = pickle.load(open(project.words_dir,'rb'))
    word2idx = pickle.load(open(project.idx_dir,'rb'))

    glove = {w:vectors[word2idx[w]] for w in words}
    matrix_len = len(target_vocab)
    embed_size = vectors.shape[1] # 载入的词向量维度
    weight_matrix = np.zeros((matrix_len, embed_size))
    words_found = 0

    for i, word in enumerate(target_vocab):
        if i%1000==0:
            sys.stdout.write("\r 已完成向量转换: %d / %d" %(i, len(target_vocab)))
        try:
            weight_matrix[i]  = glove[word]
            words_found += 1
        except KeyError:
            print(word)
            weight_matrix[i] = np.random.normal(scale=0.6, size=(embed_size, ))
    vocab_vectors = bcolz.carray(weight_matrix, rootdir=project.vector_dir, mode = 'w')
    vocab_vectors.flush()

def create_emb_layer(non_trainable = True):
    weight_matrix = bcolz.open(project.vector_dir)
    weight_matrix = np.asarray(weight_matrix)
    print(weight_matrix.shape)
    num_embeddings, embedding_dim = weight_matrix.shape
    emb_layer = nn.Embedding.from_pretrained(embeddings = torch.FloatTensor(weight_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    
    return emb_layer, num_embeddings, embedding_dim

# test
# emb_layer, num_embeddings, embedding_dim = create_emb_layer()
# print(emb_layer(torch.LongTensor([1,2])))


import nltk
import random
from nltk.corpus import movie_reviews


def read_corpus():
    documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)
    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    all_words = nltk.FreqDist(all_words)
    all_words = list(all_words)
    word_dict = {word:key for key, word in enumerate(all_words)}

    pickle.dump(word_dict, open(project.vocab_dir,'wb'))

    return all_words
# Test
# read_corpus()

import math

def doc2vec():
    word2idx = pickle.load(open(project.vocab_dir,'rb'))
    documents = [{'sentence':movie_reviews.words(fileid),'label':category}
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    documents_vector = []
    print_count = 0
    label_dict = {'pos':0, 'neg':1}
    padd_len = math.ceil(2000)
    for document in documents:
        try:
            doc_list = [word2idx[word] for word in document['sentence']]
            if len(doc_list)>=padd_len:
                doc_list = doc_list[0:padd_len]
            else:
                doc_list += [0 for _ in range(padd_len-len(doc_list))]
            documents_vector.append({'vector':doc_list, 'label':label_dict[document['label']]})
        except KeyError:
            continue
    print(documents_vector[0])
    return documents_vector

# Test
# doc2vec()