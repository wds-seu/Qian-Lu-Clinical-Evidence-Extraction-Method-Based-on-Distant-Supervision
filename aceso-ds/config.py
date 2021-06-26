# -*- coding: utf-8 -*-
import warnings
import logging
import os
import torch

path = os.path.abspath("..")


class DefaultConfig(object):
    """ default parameters and settings """

    env = 'word_only_20210518_1'  # default visdom env name
    model = 'CNNNet'  # default model
    class_list = ['P', 'I', 'O', 'N']

    # train_data_root = 'datasets/PICO'  # train data location
    path_prefix = '/home/gp/stu/MTCUGE/'
    train_data_root = path_prefix + 'datasets/aceso-ds/train'
    val_data_root = path_prefix + 'datasets/aceso-ds/meta'
    all_data_root = path_prefix + 'datasets/aceso-ds/trainmeta'
    #train_data_root = 'datasets/PICOEBMNLP1220'
    test_data_root = path_prefix + 'datasets/aceso-ds/test'  # test data location
    #train_data_root = "active_learning/orthopedic_test"

    load_model_path = 'checkpoints/CNNNet_epoch0_2021-05-24_18-05-07.pth'  # pre-trained model, or model for prediction
    save_model_path = 'checkpoints/CNNNet_mtu.pth'
    # load_model_path = "/home/tenyun/Documents/GitHome/MTCUGE/checkpoints/26.pth"
    pred_PubMed_vector = path_prefix + "materials/bio_nlp_vec/PubMed-shuffle-win-30.bin"
    pred_umls_vector = path_prefix + "materials/deepwalk.embeddings"
    pred_hs_umls_vector = "/home/gp/AcesoE_use/AcesoE/embeddings/umls.embeddings"
    customize_word_embeddings = "materials/PubMed_extracted.pl"
    customize_umls_embeddings = "materials/umls_extracted.pl"
    customize_mixing_embeddings = "materials/mixing_extracted.pl"

    words_save = "materials/words.dat"
    cuis_save = "materials/cuis.dat"
    word2cui = "materials/word2cui.dat"
    vocabulary_store = "materials/vocabulary_store.dat"

    batch_size = 128  # batch size
    use_gpu = False  # use gpu or not
    num_workers = 4  # how many workers for loading data
    print_freq = 16  # print info every N batch
    # embedding_dim = 200
    embedding_dim = 308
    word_embedding_dim = 200
    umls_embedding_dim = 108
    kernel_num = 256
    kernel_sizes = [2, 3, 4]
    class_num = 4
    max_epoch = 30
    # RNN
    hidden_dim = 100

    lr = 1e-3
    lr_decay = 0.95  # when val loss increase, lr = lr * 0.95
    mode = "static"
    weight_decay = 5e-4  # 损失函数
    dropout = 0.5
    device = 0

    use_shuffle = False
    use_drop = False
    together_calculate = True
    mixing_train = True
    use_hs = True
    pretraining = True

    # log
    log_location = "/home/gp/stu/MTCUGE/log/MTCUGE.log"
    # log_location = "D:\\ubuntu备份\GitHome\MTCUGE\log\MTCUGE.log"

    # active learning
    unlabel_data_root = "active_learning/unlabel/"  # data for active learning location
    # unlabel_data_root = "active_learning/orthopedic_test/"
    # predict_orthopedic_root = "active_learning/orthopedic_test/"
    ac_unlabel_bak_data_root = "active_learning/ac_unlabel_bak/"
    label_file_col_list = ['sentence', 'washedsentence', 'cui_list', 'pmid', 'pm_sentence_id', 'section_label',
                           'section_num', 'knn_density', 'min_dist', 'prob']
    label_file_col_num = 10
    pad_size = 28
    prefix = 'ac0'
    ac_batch = 100


    def parse(self, kwargs):
        """
        update config parameters according kwargs dict
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        # logging.info("user config:  ")
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('__'):
        #         logging.info(k, getattr(self, k))


opt = DefaultConfig()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
