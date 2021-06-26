# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import random
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module
from gensim.models import KeyedVectors
import pickle
from utils.DataHelper import DataHelper


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
label_dict = {"P": 0, "I": 1, "O": 2, "N": 3}


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    files = os.listdir(file_path)
    for file in files:
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(file_path, file), 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin
                for word in tokenizer(content):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config):

    dh_train = DataHelper(config.train_data_root, train=True)
    x_train, _, sentences_train, y_train, _, _ = dh_train.load_data()

    dh_val = DataHelper(config.val_data_root, train=True)
    x_meta, _, sentences_meta, y_meta, _, _ = dh_val.load_data()

    dh_test = DataHelper(config.test_data_root, test=True)
    x_test, _, sentences_test, y_test, vocab, vocabulary_inv = dh_test.load_data()

    ids_train = torch.from_numpy(np.array(range(len(y_train))))
    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    y_train = y_train.view(-1)
    train_data = TensorDataset(x_train, y_train, ids_train)
    # todo: debug
    print("train data len: ", len(train_data))
    print("train sentences len ", len(sentences_train))

    ids_meta = torch.from_numpy(np.array(range(len(y_meta))))
    x_meta = torch.from_numpy(x_meta).long()
    y_meta = torch.from_numpy(y_meta).long()
    y_meta = y_meta.view(-1)
    meta_data = TensorDataset(x_meta, y_meta, ids_meta)

    ids_test = torch.from_numpy(np.array(range(len(y_test))))
    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()
    y_test = y_test.view(-1)
    test_data = TensorDataset(x_test, y_test, ids_test)

    train_dataloader = DataLoaderHolder(DataLoader(train_data, config.batch_size, shuffle=True, num_workers=1), sentences_train)
    meta_dataloader = DataLoaderHolder(DataLoader(meta_data, config.batch_size, shuffle=True, num_workers=1), sentences_meta)
    test_dataloader = DataLoaderHolder(DataLoader(test_data, config.batch_size, shuffle=False, num_workers=1), sentences_test)

    return vocab, train_dataloader, meta_dataloader, test_dataloader


class DataLoaderHolder(object):
    def __init__(self, data_loader, sentences):
        self.data_loader = data_loader
        self.sentences = sentences

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches




def buildPretrainedEmbedding(embedding):
    x = import_module('models.TextCNN')
    config = x.Config('datasets/')
    w2v = KeyedVectors.load_word2vec_format('materials/embeddings/' + embedding, binary=True)
    config.embed = w2v['good'].size
    weight = {}
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    for word, index in vocab.items():
        if word in w2v:
            weight[index] = w2v.word_vec(word)
        else:
            weight[index] = np.random.uniform(-0.25, 0.25, config.embed).astype(np.float32)
    weight = np.array(list(weight.values()))
    return torch.from_numpy(weight)


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./CancerDataset/data/cancerTrain"
    vocab_dir = "./CancerDataset/data/vocab.pkl"
    pretrain_dir = "./CancerDataset/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=10)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
