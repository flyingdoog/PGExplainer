from config import *
from layers import *
from metrics import *
from tensorflow import keras
from config import *
from layers import *
from metrics import *
from tensorflow import keras

class GCN(keras.Model):
    def __init__(self, input_dim, output_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)
        usebias = args.bias
        self.bn = args.bn
        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        if self.bn:
            self.bnlayers = [tf.keras.layers.BatchNormalization()]*(len(hiddens)-1)

        self.layers_ = []

        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0],
                                  activation=tf.nn.relu,
                                  bias=usebias)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_],
                                      activation=tf.nn.relu,
                                  bias=usebias)
            self.layers_.append(layertemp)


        self.pred_layer = tf.keras.layers.Dense(output_dim)
        # self.layers_.append(pred_layer)
        self.hiddens = hiddens

    def call(self,inputs,training=None):
        out = self.getNodeEmb(inputs,training)

        out1 = tf.reduce_max(out, axis=1)
        out2 = tf.reduce_sum(out, axis=1)
        out = tf.concat([out1, out2], axis=-1)
        out = self.pred_layer(out)
        return out

    def getNodeEmb(self,inputs,training=None):
        x, support  = inputs
        nmb_layers = len(self.layers_)
        x_all = []
        for layerindex in range(nmb_layers):
            layer = self.layers_[layerindex]
            x = layer.call((x,support),training)
            if self.bn and layerindex!=nmb_layers-1:
                x = self.bnlayers[layerindex](x,training=training)
            x_all.append(x)
        if args.concat:
            x = tf.concat(x_all,axis=-1)
        out = x
        return out
