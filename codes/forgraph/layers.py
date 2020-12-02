from inits import *
import tensorflow as tf
from config import args,dtype
from tensorflow.python.keras import layers



class GraphConvolution(layers.Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, bias=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)


        self.act = activation
        self.bias = bias

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##


        self.weight = self.add_weight('weight', [input_dim, output_dim],dtype=dtype,  initializer=initializer)
        if self.bias:
            self.bias_weight = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")


    def call(self, inputs, training=None):
        x, support = inputs

        if training and args.dropout>0:
            x = tf.nn.dropout(x, args.dropout)


        if args.order =='AW':
            if isinstance(support, tf.Tensor):
                output = tf.matmul(support, x)
            else:
                output = tf.sparse.sparse_dense_matmul(support, x)
            output = tf.matmul(output, self.weight)

        else:
            pre_sup = tf.matmul(x, self.weight)
            if  isinstance(support,tf.Tensor):
                output = tf.matmul(support, pre_sup)
            else:
                output = tf.sparse.sparse_dense_matmul(support, pre_sup)

        if self.bias:
            output += self.bias_weight

        if args.embnormlize:
            output,_ = tf.linalg.normalize(output,ord=2,axis=-1)

        return self.act(output)


class Dense(layers.Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, bias=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)


        self.act = activation
        self.bias = bias

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##


        self.weight = self.add_weight('weight', [input_dim, output_dim],dtype=dtype,  initializer=initializer)
        if self.bias:
            self.bias_weight = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")


    def call(self, inputs, training=None):
        x, support = inputs

        if training and args.dropout>0:
            x = tf.nn.dropout(x, args.dropout)


        if args.order =='AW':
            if isinstance(support, tf.Tensor):
                output = tf.matmul(support, x)
            else:
                output = tf.sparse.sparse_dense_matmul(support, x)
            output = tf.matmul(output, self.weight)

        else:
            pre_sup = tf.matmul(x, self.weight)
            if  isinstance(support,tf.Tensor):
                output = tf.matmul(support, pre_sup)
            else:
                output = tf.sparse.sparse_dense_matmul(support, pre_sup)

        if self.bias:
            output += self.bias_weight

        if args.embnormlize:
            output,_ = tf.linalg.normalize(output,ord=2,axis=-1)

        return self.act(output)
