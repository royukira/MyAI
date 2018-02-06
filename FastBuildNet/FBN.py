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
    for i in range(1, len(lays_info)+1):
        layerName = lays_info[i][0]
        with tf.name_scope(layerName):
            with tf.name_scope('Weights'):
                weightSet = lays_info[i][1]
                for p in weightSet:
                    para_summaries(para=p, name=layerName + '/Weights')

            with tf.name_scope('Biases'):
                biaSet = lays_info[i][2]
                for b in biaSet:
                    para_summaries(para=b, name=biaSet + '/Biases')

        with tf.name_scope('Weight_Sum'):
            for wb in range(len(weightSet)):  # the number of weightSet is equal to #bias
                Z = tf.matmul(tensor, weightSet[i]) + biaSet[i] # tensor is data X
                tf.summary.histogram(layerName + '/Weight_sum', Z)

        g = lays_info[i][3](Z)  # activation function
        tf.summary.histogram(layerName + '/Activation_function', g)
        tensor = g
    return g


def para_summaries(para, name):
    """
    Attach a lot of summaries to a Tensor.
    --------------------------------------
    :param para: Wi or Bi (ith layer)
    :param name: Layer name + Parameter name
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(para)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(para - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(para))
        tf.summary.scalar('min/' + name, tf.reduce_min(para))
        tf.summary.histogram(name, para)


