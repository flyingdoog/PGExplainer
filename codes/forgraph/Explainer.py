import tensorflow as tf
from config import args

class Explainer(tf.keras.Model):
    def __init__(self, model, nodesize, **kwargs):
        super(Explainer, self).__init__(**kwargs)
        with tf.name_scope("explainer") as scope:
            self.elayers = [tf.keras.layers.Dense(64,tf.nn.relu),
                            tf.keras.layers.Dense(1)]

        rc = tf.expand_dims(tf.range(nodesize),0)
        rc = tf.tile(rc,[nodesize,1])
        self.row = tf.reshape(tf.transpose(rc),[-1])
        self.col = tf.reshape(rc,[-1])
        # For masking diagonal entries
        self.nodesize = nodesize
        self.model = model

        tfones = tf.ones((nodesize, nodesize))
        self.diag_mask =  tf.cast(tfones,tf.float32)-tf.eye(nodesize)

        self.mask_act = 'sigmoid'

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            debug_var = 0.0
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var, dtype=tf.float32)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        return gate_inputs



    def call(self,inputs,training=None):
        x, embed, adj,tmp, label = inputs
        self.label = tf.argmax(tf.cast(label,tf.float32),axis=-1)

        self.tmp = tmp
        f1 = tf.gather(embed, self.row)
        f2 = tf.gather(embed, self.col)
        f12self = tf.concat([f1, f2], axis=-1)
        h = f12self
        for elayer in self.elayers:
            h = elayer(h)
        self.values = tf.reshape(h, [-1])

        values = self.concrete_sample(self.values, beta=tmp, training=training)


        sparsemask = tf.sparse.SparseTensor(
            indices=tf.cast(tf.concat([tf.expand_dims(self.row, -1), tf.expand_dims(self.col, -1)], axis=-1),tf.int64),
            values=values,
            dense_shape=[self.nodesize,self.nodesize]
        )

        sym_mask = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(sparsemask)), tf.float32)
        self.mask = sym_mask

        sym_mask = (sym_mask + tf.transpose(sym_mask)) / 2
        masked_adj = tf.multiply(adj, sym_mask)

        self.masked_adj = masked_adj
        x = tf.expand_dims(x,0)
        adj = tf.expand_dims(self.masked_adj,0)

        output = self.model((x,adj))
        res = tf.nn.softmax(output)
        return res


    def loss(self, pred, pred_label):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        pred_reduce = pred[0]
        gt_label_node = self.label
        logit = pred_reduce[gt_label_node]
        pred_loss = -tf.math.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = tf.nn.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = tf.nn.relu(self.mask)
        size_loss = args.coff_size * tf.reduce_sum(mask)

        # entropy
        mask = mask*0.99+0.005
        mask_ent = -mask * tf.math.log(mask) - (1 - mask) * tf.math.log(1 - mask)
        mask_ent_loss = args.coff_ent * tf.reduce_mean(mask_ent)


        loss = pred_loss + size_loss + mask_ent_loss

        return loss