#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import math
import time
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import layers as L

def _variable_on_cpu(name, shape, initializer,wd=None):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def intfVGG_use_layer(input_tensor, n_classes=1000, rgb_mean=None, training=True):
    # assuming 224x224x3 input_tensor
    # define image mean
    if rgb_mean is None:
        rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    mu = tf.constant(rgb_mean, name="rgb_mean")
    keep_prob = 0.5

    # subtract image mean
    net = tf.subtract(input_tensor, mu, name="input_mean_centered")

    # block 1 -- outputs 112x112x64
    net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
    net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
    net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
    net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
    net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
    net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
    net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
    net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
    net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    net = L.fully_connected(net, name="fc6", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc7", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc8", n_out=n_classes)
    return net

def intfVGG(images,n_classes=1000,wd=0.0005,train=True):
    # wd = weight decay
    with tf.variable_scope('conv1_1') as scope:
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 3, 64],
          tf.random_normal_initializer(stddev=1e-2),wd)

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv1_1)

    with tf.variable_scope('conv1_2') :
        # shape[0]*shape[1] kernelsize shape[2] kernelIN shape[3]kernelnum
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 64, 64],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu1_2
        conv1_2 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv1_2)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 64, 128],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu2_1
        conv2_1 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv2_1)

    # conv2_2
    with tf.variable_scope('conv2_2') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 128, 128],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu2_2
        conv2_2 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv2_2)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3_1
    with tf.variable_scope('conv3_1') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 128, 256],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu3_1
        conv3_1 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv3_1)

        # conv3_2
    with tf.variable_scope('conv3_2') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 256, 256],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu3_2
        conv3_2 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv3_2)

    # conv3_3
    with tf.variable_scope('conv3_3') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 256, 256],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu3_3
        conv3_3 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv3_3)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    # conv4_1
    with tf.variable_scope('conv4_1') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 256, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu4_1
        conv4_1 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv4_1)

    # conv4_2
    with tf.variable_scope('conv4_2') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 512, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu4_2
        conv4_2 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv4_2)

    # conv4_3
    with tf.variable_scope('conv4_3') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 512, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu4_3
        conv4_3 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv4_3)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')

    # conv5_1
    with tf.variable_scope('conv5_1') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 512, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu5_1
        conv5_1 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv5_1)

    # conv5_2
    with tf.variable_scope('conv5_2') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 512, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu5_2
        conv5_2 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv5_2)


    # conv5_3
    with tf.variable_scope('conv5_3') :
        kernel = _variable_on_cpu(
          'weights',
          [3, 3, 512, 512],
          tf.random_normal_initializer(stddev=1e-2),wd)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
        #relu5_3
        conv5_3 = tf.nn.relu(out, name=scope.name)
        # _activation_summary(conv5_3)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # fc1
    # 输入是3通道，三通道一起卷积了相加，最后输出是1维？
    with tf.variable_scope('fc6') :
        # np.prod 元素逐个相乘
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = _variable_on_cpu(
          'weights',
          [shape,4096],
          tf.random_normal_initializer(stddev=1e-2),wd)
        
        fc1b = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
        # -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        #relu6
        if train:
            fc1 = tf.nn.dropout(tf.nn.relu(fc1l), 0.5,name=scope.name)
        else:
            fc1 = tf.nn.relu(fc1l,name=scope.name)
        # _activation_summary(fc1)

    # fc2
    with tf.variable_scope('fc7') :
        fc2w = _variable_on_cpu(
          'weights',
          [4096,4096],
          tf.random_normal_initializer(stddev=1e-2),wd)
        fc2b = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        #relu7
        if train:
            fc2 = tf.nn.dropout(tf.nn.relu(fc2l), 0.5,name=scope.name)
        else:
            fc2 = tf.nn.relu(fc2l,name=scope.name)
        # _activation_summary(fc2)

    # fc3
    with tf.variable_scope('fc8') :
        fc3w = _variable_on_cpu(
          'weights',
          [4096,n_classes],
          tf.random_normal_initializer(stddev=1e-2),wd)
        fc3b = _variable_on_cpu('biases', [n_classes], tf.constant_initializer(0.0))
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b,name=scope.name)
        # _activation_summary(fc3l)

    return fc3l

def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
