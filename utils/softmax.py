#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-19 下午4:47
# @Author  : zheng
# @Site    : 
# @File    : softmax.py
# @Software: PyCharm

import numpy as np
def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape(x.shape[0], 1)

        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


