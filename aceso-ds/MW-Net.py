# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
import models
from sklearn import metrics
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

from utils.Embed import EmbedUtil
from models.resnet import VNet
from config import DefaultConfig
from utils.util import build_dataset

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.6,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=316)
parser.add_argument('--epochs', default=150, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.5, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

parser.set_defaults(augment=True)

args = parser.parse_args()
torch.manual_seed(args.seed)

opt = DefaultConfig()
emb_utils = EmbedUtil()

use_cuda = opt.use_gpu
device = torch.device("cuda" if use_cuda else "cpu")
aceso = True


def build_dataset_and_model():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    vocabulary, train_dataloader, meta_dataloader, test_dataloader = build_dataset(opt)
    print("vocab: ", len(vocabulary))

    # prepare model
    print("mixing embedding (pubmed word2vec + umls) ...")
    pretrained_embeddings = emb_utils.load_mixing_embedding()

    model = getattr(models, opt.model)(vocab_size=len(vocabulary), pretrained_embeddings=pretrained_embeddings)
    # x = import_module('models.TextCNN')
    # config = x.Config('datasetsV4/')
    # config.embedding_pretrained = pretrained_embeddings
    # config.n_vocab = len(vocabulary)
    # model = x.Model(config).to(config.device)
    # todo: 载入训练好的模型
    if opt.pretraining:
         model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    # print(model.parameters)
    return train_dataloader, meta_dataloader, test_dataloader, model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model, test_loader):
    test_loader = test_loader.data_loader
    opt = DefaultConfig()
    model.eval()
    test_loss = 0

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for batch_idx, (inputs, targets, ids) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            # correct += predicted.eq(targets).sum().item()

            labels = targets.data.cpu().numpy()
            predicts = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predicts)

    test_loss /= len(test_loader.dataset)
    # accuracy = 100. * correct / len(test_loader.dataset)

    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=opt.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     accuracy))
    return acc, test_loss, report, confusion, labels_all, predict_all


def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    print('begin training')
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    ids_train = []
    weights_train = []
    targets_train = []
    losses_train = []

    train_sentences = train_loader.sentences
    print('sentences: ', len(train_sentences))
    train_loader = train_loader.data_loader
    train_meta_loader = train_meta_loader.data_loader
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, ids) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        if use_cuda:
            meta_model = model.cuda()
        else:
            meta_model = model
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)

        cost = F.cross_entropy(outputs, targets, reduce=False)  # loss： f(w)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)   # weight f(w, theta)
        l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)   # 加权训练loss
        meta_model.zero_grad()
        # todo: grad这里修改了CNNNet的mode
        grads = torch.autograd.grad(l_f_meta, (filter(lambda p: p.requires_grad, model.parameters())), create_graph=True)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        #                              weight_decay=opt.weight_decay)

        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))   # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)   # 调整w
        del grads

        try:
            inputs_val, targets_val, ids_val = next(train_meta_loader_iter)
        except StopIteration:
            # meta的batch size为64，滚动获取
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val, ids_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)  # meta data的target
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)   # meta data的loss
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()   # 调整vnet，使得meta data loss最小

        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)

        loss = torch.sum(cost_v * w_new)/len(cost_v)

        # todo:detach好像会影响模型更新，如何赋值可更新的张量cost_v,之前打印出來，已經验证了结论
        # print("cost_v: ", cost_v)  # 损失
        # print("weight:", w_new)

        # todo: get weight and id, 把模型和vnet都存起来，再载入预测所有
        w_new = tensor2list(w_new)
        ids = tensor2list(ids)
        targets = tensor2list(targets)
        # loss1 = cost_v.view(-1).detach().numpy().tolist()
        loss1 = np.zeros(len(ids))
        ids_train.extend(ids)
        weights_train.extend(w_new)
        targets_train.extend(targets)
        losses_train.extend(loss1)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
    print("ids:", len(ids_train))
    print("weights: ", len(weights_train))
    print("sentences: ", len(train_sentences))
    assert len(ids_train) == len(weights_train) == len(train_sentences)
    sentences_with_weight = []
    for i in range(len(ids_train)):
        id = ids_train[i]
        sentences_with_weight.append({'id': ids_train[i], 'sentence': train_sentences[id],
                                      'weight': weights_train[i], 'target': targets_train[i], "loss": losses_train[i]})
    return sentences_with_weight

def tensor2list(tensor):
    tensor = tensor.view(-1)
    list_result = np.array(tensor).tolist()
    return list_result


def train_bak(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    inputs_train = []
    loss_train = []

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        if use_cuda:
            meta_model = model.cuda()
        else:
            meta_model = model
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)

        cost = F.cross_entropy(outputs, targets, reduce=False)  # loss： f(w)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)   # weight f(w, theta)
        l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)   # 加权训练loss
        meta_model.zero_grad()
        # todo: grad这里修改了CNNNet的mode
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)

        meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))   # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)   # 调整w
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)  # meta data的target
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)   # meta data的loss
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()   # 调整vnet，使得meta data loss最小

        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]



        with torch.no_grad():
            w_new = vnet(cost_v)

        loss = torch.sum(cost_v * w_new)/len(cost_v)

        # todo: print loss top
        # inputs_train.extend(inputs)
        # loss_train.extend(loss_curr)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


        train_loss += loss.item()
        meta_loss += l_g_meta.item()


        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))


train_loader, train_meta_loader, test_loader, model = build_dataset_and_model()


def main():
    if use_cuda:
        vnet = VNet(1, 100, 1).cuda()
    else:
        vnet = VNet(1, 100, 1)

    optimizer_model = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    # optimizer_model = torch.optim.SGD(model.params(), args.lr,
    #                                   momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
                                      weight_decay=1e-4)
    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        sentences_with_weight = train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc, test_loss, report, confusion, labels, predicts = test(model=model, test_loader=test_loader)
        print('test acc: ', test_acc)
        if test_acc > best_acc:
            write_test_result(test_acc, test_loss, report, confusion, labels, predicts)
            write_ac_result(sentences_with_weight)
            write_vnet_result(sentences_with_weight)
            best_acc = test_acc
            torch.save(model.state_dict(), opt.save_model_path)
    print('best accuracy:', best_acc)


def test_only():
    test_acc, test_loss, report, confusion, labels, predicts = test(model=model, test_loader=test_loader)
    print('test acc: ', test_acc)
    write_test_result(test_acc, test_loss, report, confusion, labels, predicts)


def write_vnet_result(sentences_with_weight):
    ids = []
    sentences = []
    weights = []
    losses = []

    for entry in sentences_with_weight:
        ids.append(entry['id'])
        sentences.append(entry['sentence'])
        weights.append(entry['weight'])
        losses.append(entry['loss'])

    df = pd.DataFrame({'id': ids, 'sentence': sentences, 'loss': losses, 'weight': weights})
    df.to_csv('vnet_result.csv', index=False)


def write_test_result(test_acc, test_loss, report, confusion, labels, predicts):
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    df = pd.DataFrame({"labels": labels.tolist(), "predicts": predicts.tolist()})
    df.to_csv("output/test_result.csv", index=False)


# todo：从train_set.csv里找到ac的所有句子，摘出来
def write_ac_result(sentences_with_weight):
    sentences_with_weight.sort(key=lambda x: x['weight'])
    sentences_ac = []
    with open('output/%s_meta_result.txt' % opt.prefix, 'w') as f:
        for entry in sentences_with_weight[:opt.ac_batch]:
            tmp = entry['sentence']
            sentences_ac.append(tmp)
            f.write(tmp + '\n')
    # 从train_set.csv里找到ac的所有句子，摘出来
    # ac_file = pd.



if __name__ == '__main__':
    # test_only()
    main()
