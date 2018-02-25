"""

FastBuildNet

Some general functions help building the neural network more easily

"""

import tensorflow as tf


def createLayer(tensor, lays_info):
    """
    Create a layer
    ----------------------------
    :param tensor: input data
    :param weight: a list of weight
    :param lays_info: a dict of {layerId:("layerName",(w1,w2,w3,...,wn),(b1,b2,b3,...), activationFunc)}
    :return: Layer's activation function g(z)
    """
    global Z,g
    para_sum_list = []
    for i in range(1, len(lays_info)+1):
        layerName = lays_info[i][0]
        with tf.name_scope(layerName):
            with tf.name_scope('Weights'):
                weight = lays_info[i][1]
                mean_plot, std_plot, max_plot, min_plot, para_plot = para_summaries(para=weight, name=layerName + '/Weights')
                para_sum_list.extend([mean_plot, std_plot, max_plot, min_plot, para_plot])

            with tf.name_scope('Biases'):
                bias = lays_info[i][2]
                b_mean_plot, b_std_plot, b_max_plot, b_min_plot, b_para_plot = para_summaries(para=bias, name=layerName + '/Biases')
                para_sum_list.extend([b_mean_plot, b_std_plot, b_max_plot, b_min_plot, b_para_plot])

        with tf.name_scope('Weight_Sum'):
            Z = tf.matmul(tensor, weight) + bias # tensor is data X
            tf.summary.histogram(layerName + '/Weight_sum', Z)
        if lays_info[i][3] is not None:
            g = lays_info[i][3](Z)  # activation function
            tf.summary.histogram(layerName + '/Activation_function', g)
            tensor = g
        else:
            g = Z
            tf.summary.histogram(layerName + '/Linear_Activation_Function', g)
    return g, para_sum_list


def para_summaries(para, name):
    """
    Attach a lot of summaries to a Tensor.
    --------------------------------------
    :param para: Wi or Bi (ith layer)
    :param name: Layer name + Parameter name
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(para)
        mean_plot = tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(para - mean)))
        std_plot = tf.summary.scalar('sttdev/' + name, stddev)
        max_plot = tf.summary.scalar('max/' + name, tf.reduce_max(para))
        min_plot = tf.summary.scalar('min/' + name, tf.reduce_min(para))
        para_plot = tf.summary.histogram(name, para)

    return mean_plot,std_plot,max_plot,min_plot,para_plot



