#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch as t
import time
import logging
from config import DefaultConfig
from torch.nn.parameter import Parameter
opt = DefaultConfig()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=opt.log_location,
                    filemode='a+')
from torch.autograd import Variable


def to_var(x, requires_grad=True):
    if opt.device:
        if t.cuda.is_available():
            x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要提供了save和load两个方法
    提供快速加载和保存模型的接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))   # 模型的默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        :param path: 路径
        :return: 模型
        """
        return self.load_state_dict(t.load(path))

    def save(self, name=None, epoch=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        :param name: 模型名字
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + "_epoch" + str(epoch) + '_'
            name = time.strftime(prefix + '%Y-%m-%d_%H-%M-%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, Parameter(param))

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class Flat(t.nn.Module):
    """
    reshape to (batch_size, dim_length)
    """
    def __init__(self):
        super(Flat, self).__init__()

    def foward(self, x):
        return x.view(x.size(0), -1)

