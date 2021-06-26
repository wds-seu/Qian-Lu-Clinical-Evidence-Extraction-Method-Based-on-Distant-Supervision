# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchnet import meter
from tqdm import tqdm
from utils.visualize import Visualizer
import models
from config import DefaultConfig
from utils.DataHelper import DataHelper
from utils.Embed import EmbedUtil
import logging
import torch.nn.functional as F
import pandas as pd
import numpy as np
from active_learning import ac
import os

opt = DefaultConfig()
emb_utils = EmbedUtil()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')

for_error_analysis = []

def train(**kwargs):
    vis = Visualizer(opt.env)
    opt.parse(kwargs)
    # step1: prepare data
    print("start")
    # dh = DataHelper(opt.train_data_root, train=True)
    # x_text, cuis, sentences_origin, y, vocabulary, vocabulary_inv = dh.load_data()
    # x_train, x_val, y_train, y_val = train_test_split(x_text, y, test_size=0.3, random_state=1, shuffle=True)
    print("data")
    # test, train
    dh_train = DataHelper(opt.train_data_root, train=True)
    x_train, _, _, y_train, _, _ = dh_train.load_data()

    dh_val = DataHelper(opt.val_data_root, train=True)
    x_val, _, _, y_val, _, _ = dh_val.load_data()

    dh_test = DataHelper(opt.test_data_root, test=True)
    x_test, cuis, _, y_test, vocabulary, vocabulary_inv = dh_test.load_data()

    print("vocab: ", len(vocabulary))

    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).long()
    y_train = y_train.view(-1)
    train_data = TensorDataset(x_train, y_train)

    x_val = torch.from_numpy(x_val).long()
    y_val = torch.from_numpy(y_val).long()
    y_val = y_val.view(-1)
    val_data = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step2: model
    if opt.mixing_train:
        print("embedding...")
        pretrained_embeddings = emb_utils.load_mixing_embedding()
    else:
        pretrained_embeddings = emb_utils.load_words_embedding()

    model = getattr(models, opt.model)(vocab_size=len(vocabulary), pretrained_embeddings=pretrained_embeddings)
    # todo: 载入预训练的模型
    # if opt.load_model_path:
    #    model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # setp3 : loss function and optim
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    # fix the emb parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                 weight_decay=opt.weight_decay)

    # step4 : CM
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(4)
    previous_loss = 1e100

    print("train start...")
    max_result_single = [0, 0, 0]
    max_result_all = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # step5 : train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        for ii, (data, label) in (enumerate(train_dataloader)):
            # train_dataloader.batch_size
            # 训练模型参数d
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                torch.cuda.set_device(opt.device)
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            if "BiLSTMNet" == opt.model:
                score, _ = model(input.t_())
            elif "LSTMNet" == opt.model or "RNNNet" == opt.model:
                score, _ = model(input.t_())
            elif "CNNNet" == opt.model:
                score = model(input)
            else:
                score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            m = torch.max(score, 1)[1]
            confusion_matrix.add(m.view(target.size()).data, target.data)

            if (ii + 1) % opt.print_freq == 0:
                # 训练集指标可视化
                cm_value = confusion_matrix.value()
                if not opt.together_calculate:
                    result_p, result_i, result_o, result_n = vis.calculate_and_show(cm_value, together_calculate=False)
                    data = [result_p, result_i, result_o, result_n]

                    vis.plot_lprf_dependent(data, env="train")
                    vis.plot('train_loss', loss_meter.value()[0])

                    # if result_p[2] > max_result_all[0][2]:
                    #     max_result_all[0] = result_p
                    # if result_i[2] > max_result_all[1][2]:
                    #     max_result_all[1] = result_i
                    # if result_o[2] > max_result_all[2][2]:
                    #     max_result_all[2] = result_o
                    # if result_n[2] > max_result_all[3][2]:
                    #     max_result_all[3] = result_n
                else:
                    train_accuracy, train_precision, train_recall, train_f1 = vis.calculate_and_show(cm_value)
                    data = [train_accuracy, train_precision, train_recall, train_f1]
                    # if train_f1 > max_result_single[2]:
                    #     max_result_single[2] = train_f1
                    #     max_result_single[0] = train_precision
                    #     max_result_single[1] = train_recall
                    vis.plot_laprf(data, env="train")
                    vis.plot('train_loss', loss_meter.value()[0])
        model.save(epoch=epoch)
        # 计算验证集上的指标以及可视化
        vocabulary_inv = {index: word for word, index in vocabulary.items()}
        val_data = val(model, val_dataloader, loss_meter, vis, epoch, vocabulary_inv)

        # 记录验证集上最高的f1值和对应的p、r
        if not opt.together_calculate:
            if val_data[0][2] > max_result_all[0][2]:
                max_result_all[0] = val_data[0]
            if val_data[1][2] > max_result_all[1][2]:
                max_result_all[1] = val_data[1]
            if val_data[2][2] > max_result_all[2][2]:
                max_result_all[2] = val_data[2]
            if val_data[3][2] > max_result_all[3][2]:
                max_result_all[3] = val_data[3]
        else:
            if val_data[3] > max_result_single[2]:
                max_result_single[2] = val_data[3]
                max_result_single[0] = val_data[1]
                max_result_single[1] = val_data[2]

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
    # save the visdom data
    vis.save()
    # np.savetxt("for_error_analysis_" + opt.env + ".txt", np.array(for_error_analysis), fmt="%s", encoding="utf-8")
    if not opt.together_calculate:
        print("current epoch max result is train_precision %f %f %f" %(max_result_all[0][0], max_result_all[0][1],max_result_all[0][2]))
        print(" train_recall %f %f %f" %(max_result_all[1][0], max_result_all[1][1], max_result_all[1][2]))
        print(" train_f1 %f %f %f" % (max_result_all[2][0], max_result_all[2][1], max_result_all[2][2]))
    else:
        print("current epoch max result are show as: ")
        print(max_result_single)


def val(model, dataloader, loss_meter, vis, epoch, vocabulary_inv):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(4)
    max_result_single = [0, 0, 0, 0]  # together_calculate = True时  记录最大的值  accuracy  precision  recall  f1
    max_result_all = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # # together_calculate = False 时  记录最大的值 [p,r,f1] [pion]
    for ii, data in (enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        if opt.use_gpu:
            torch.cuda.set_device(opt.device)
            val_input = val_input.cuda()
        if "BiLSTMNet" == opt.model:
            score, _ = model(input.t_())
        elif "LSTMNet" == opt.model or "RNNNet" == opt.model:
            score, _ = model(input.t_())
        elif "CNNNet" == opt.model:
            score = model(input)
        else:
            score = model(input)
        if epoch == opt.max_epoch - 1:
            build_input_active_learning(val_input, vocabulary_inv)
        m = torch.max(score, 1)[1]
        # 保存验证集结果，用于错误分析
        # temp_arr = np.c_[np.array(input), np.array(label), np.array(m)]
        # for_error_analysis.append(temp_arr)
        confusion_matrix.add(m.view(label.size()).data, label.type(torch.LongTensor))

        if (ii + 1) % opt.print_freq == 0:
            cm_value = confusion_matrix.value()
            if not opt.together_calculate:
                result_p, result_i, result_o, result_n = vis.calculate_and_show(cm_value, together_calculate=False)
                data = [result_p, result_i, result_o, result_n]
                vis.plot('val_loss', loss_meter.value()[0])
                vis.plot_lprf_dependent(data, env="val")
                # val_data = data
                if data[0][2] > max_result_all[0][2]:
                    max_result_all[0] = data[0]
                if data[1][2] > max_result_all[1][2]:
                    max_result_all[1] = data[1]
                if data[2][2] > max_result_all[2][2]:
                    max_result_all[2] = data[2]
                if data[3][2] > max_result_all[3][2]:
                    max_result_all[3] = data[3]
            else:
                train_accuracy, train_precision, train_recall, train_f1 = vis.calculate_and_show(cm_value)
                data = [train_accuracy, train_precision, train_recall, train_f1]
                vis.plot_laprf(data, env="val")
                vis.plot('val_loss', loss_meter.value()[0])
                # val_data = data
                if data[3] > max_result_single[3]:
                    max_result_single[3] = data[3]
                    max_result_single[0] = data[0]
                    max_result_single[1] = data[1]
                    max_result_single[2] = data[2]

    model.train()
    if not opt.together_calculate:
        return max_result_all
    else:
        return max_result_single


def test_many(**kwargs):
    opt.parse(kwargs)
    files = os.listdir(opt.test_data_root)
    import datetime
    today = datetime.date.today()
    savepath = "output/" + str(today)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    dh = DataHelper(opt.test_data_root, test=True)
    ans = dh.load_test_data(opt.test_data_root)
    vocabulary, vocabulary_inv = dh.build_vocab()

    pretrained_embeddings = emb_utils.load_mixing_embedding()
    model = getattr(models, opt.model)(vocab_size=len(vocabulary), pretrained_embeddings=pretrained_embeddings)
    for file in files:
        test_data, cuis, sentences_origin = ans[file]
        model.eval()
        if opt.load_model_path:
            model.load(opt.load_model_path)
        if opt.use_gpu:
            model.cuda()
        sentences_id = np.array([i for i in range(0, len(sentences_origin))])
        test_data_torch = torch.from_numpy(test_data).long()
        sentences_id = torch.from_numpy(sentences_id).view(-1)
        test_data_torch = TensorDataset(test_data_torch, sentences_id)
        test_dataloader = DataLoader(test_data_torch, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        label_map = {0: "P", 1: "I", 2: "O", 3: "N"}
        res_df = pd.DataFrame(columns=["id", "sentences", "cuis", "label", "prob", "score"],
                              index=range(0, len(test_data_torch)))
        for ii, (data, sentences_id) in tqdm(enumerate(test_dataloader)):
            input = Variable(data, volatile=True)
            if opt.use_gpu:
                input = input.cuda()
            score = model(input)
            probability = F.softmax(score)
            # for index, prob in zip(sentences_id.numpy(), probability.data.numpy()):
            #     prob_dict[index] = prob
            m = torch.max(probability, 1)[1]
            for index, prob, probs, s in zip(sentences_id.numpy(), m.data, probability.data, score.data):
                res_df.loc[index]["id"] = index
                res_df.loc[index]["sentences"] = sentences_origin[index]
                res_df.loc[index]["cuis"] = cuis[index]
                res_df.loc[index]["label"] = label_map[prob]
                res_df.loc[index]["prob"] = list(probs.numpy())
                res_df.loc[index]["score"] = list(s.numpy())
        res_df.to_csv(savepath + "/ " + file, encoding="utf-8", index=False)


def test(**kwargs):
    opt.parse(kwargs)
    # data
    dh = DataHelper(opt.test_data_root, test=True)
    test_data, cuis, sentences_origin, y, vocabulary, vocabulary_inv = dh.load_data()
    # model
    if opt.mixing_train:
        print("embedding...")
        pretrained_embeddings = emb_utils.load_mixing_embedding()
    else:
        pretrained_embeddings = emb_utils.load_words_embedding()

    model = getattr(models, opt.model)(vocab_size=len(vocabulary), pretrained_embeddings=pretrained_embeddings)
    model.eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    sentences_id = np.array([i for i in range(0, len(sentences_origin))])
    test_data = torch.from_numpy(test_data).long()
    sentences_id = torch.from_numpy(sentences_id)
    sentences_id = sentences_id.view(-1)
    test_data = TensorDataset(test_data, sentences_id)
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    label_map = {0: "P", 1: "I", 2: "O", 3: "N"}
    df = pd.DataFrame(columns=["sentences", "label"], index=range(0, len(test_data)))
    for ii, (data, sentences_id) in enumerate(test_dataloader):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = F.softmax(score)
        m = torch.max(probability, 1)[1]
        for index, prob in zip(sentences_id.numpy(), m.data):
            prob = prob.item()
            df.loc[index]["sentences"] = sentences_origin[index]
            df.loc[index]["label"] = label_map[prob]
    df.to_csv("output/result.csv", encoding="utf-8", index=False)

def help():
    """
    打印帮助信息
    python file.py help
    """

    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example:
        python {0} train --env='env0701' --lr=0.01
        python {0} test --dataset='path/to/dataset/root'
        python {0} help
    avaiable args:
    """.format(__file__))
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


def build_input_active_learning(sentences, vocabulary_inv):
    """build middle input for active learning"""
    sentences = sentences.cpu().data.numpy()
    input = [[vocabulary_inv[word] for word in sentence] for sentence in sentences]
    df = pd.DataFrame(columns=["id", "sentence", "cuis", "is_label"], index=range(len(input)))
    for ii in range(len(input)):
        col1 = " ".join(map(str, input[ii]))
        df.loc[ii]["id"] = ii
        df.loc[ii]["sentence"] = " "
        df.loc[ii]["cuis"] = col1
        df.loc[ii]["is_label"] = 1
    df.to_csv("active_learning/label_data_with_prob.csv", sep=",", index=False, encoding="utf-8", mode='a',
              header=False)


def build_active_learning_data(**kwargs):
    # 根据命令行参数更新配置
    print('build active learning data')
    opt.parse(kwargs)
    print('model: ', opt.load_model_path)
    print('mlp root:', opt.unlabel_data_root)
    ac.build_ac_data()


if __name__ == '__main__':
    import fire
    fire.Fire()
    # build_active_learning_data()
    # train()
    # test()
