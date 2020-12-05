import tensorflow as tf
from config import args


class Explainer(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(Explainer, self).__init__(**kwargs)

        with tf.name_scope("explainer") as scope:
            self.elayers = [tf.keras.layers.Dense(64,tf.nn.relu),
                            tf.keras.layers.Dense(1)]


        self.model = model
        self.mask_act = 'sigmoid'
        # self.label = tf.argmax(tf.cast(label,tf.float32),axis=-1)
        self.params = []

        self.coeffs = {
            "size": args.coff_size,
            "weight_decay": args.weight_decay,
            "ent": args.coff_ent
        }

    def _masked_adj(self,mask,adj):
        sym_mask = mask
        sym_mask = (sym_mask + tf.transpose(sym_mask)) / 2

        sparseadj = tf.sparse.SparseTensor(
            indices=tf.cast(tf.concat([tf.expand_dims(adj.row,-1), tf.expand_dims(adj.col,-1)], axis=-1),tf.int64),
            values = adj.data,
            dense_shape=adj.shape
        )
        adj = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(sparseadj)),tf.float32)
        self.adj = adj

        masked_adj = tf.multiply(adj,sym_mask)

        num_nodes = adj.shape[0]
        tfones = tf.ones((num_nodes, num_nodes))
        diag_mask =  tf.cast(tfones,tf.float32)-tf.eye(num_nodes)

        return tf.multiply(masked_adj,diag_mask)


    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            bias = args.sample_bias
            random_noise = tf.random.uniform(tf.shape(log_alpha), minval=bias, maxval=1.0-bias, dtype=tf.float32)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        # gate_inputs = tf.sigmoid(log_alpha)
        return gate_inputs


    def call(self,inputs,training=False):

        x,adj,nodeid, embed, tmp = inputs
        self.tmp = tmp
        f1 = tf.gather(embed, adj.row)
        f2 = tf.gather(embed, adj.col)
        selfemb = embed[nodeid]
        selfemb = tf.expand_dims(selfemb,0)
        selfemb = tf.tile(selfemb, [f1.shape[0], 1])
        f12self = tf.concat([f1, f2, selfemb], axis=-1)
        h = f12self
        for elayer in self.elayers:
            h = elayer(h)
        self.values = tf.reshape(h,[-1])

        values = self.concrete_sample(self.values,beta=tmp,training=training)
        # if not training:
        #     print(values)

        sparsemask = tf.sparse.SparseTensor(
            indices= tf.cast(tf.concat([tf.expand_dims(adj.row,-1), tf.expand_dims(adj.col,-1)], axis=-1),tf.int64),
            values = values,
            dense_shape=adj.shape
        )
        mask = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(sparsemask)),tf.float32)
        masked_adj = self._masked_adj(mask,adj)

        self.mask = mask
        self.masked_adj = masked_adj

        output = self.model((x,masked_adj))

        node_pred = output[nodeid, :]
        res = tf.nn.softmax(node_pred,axis=0)

        return res


    def loss(self, pred, pred_label, label, node_idx, adj_tensor=None):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        label = tf.argmax(tf.cast(label, tf.float32), axis=-1)

        pred_label_node = pred_label[node_idx]
        logit = pred[pred_label_node]

        if args.miGroudTruth:
            gt_label_node = label[node_idx]
            logit = pred[gt_label_node]

        logit += 1e-6
        pred_loss = -tf.math.log(logit)

        if args.budget<=0:
            size_loss = self.coeffs["size"] * tf.reduce_sum(self.mask)
        else:
            size_loss = self.coeffs["size"] * tf.nn.relu(tf.reduce_sum(self.mask)-args.budget)
        scale=0.99
        mask = self.mask*(2*scale-1.0)+(1.0-scale)
        mask_ent = -mask * tf.math.log(mask) - (1 - mask) * tf.math.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * tf.reduce_mean(mask_ent)

        l2norm = 0
        for weight in self.elayers.weights:
            if 'kernel' in weight.name:
                l2norm += tf.norm(weight)
        l2norm = self.coeffs['weight_decay']*l2norm
        loss = pred_loss +size_loss+l2norm+mask_ent_loss

        if args.budget>0 and args.coff_connect>0:

            # sample args.connect_sample adjacency pairs
            adj_tensor_dense = tf.sparse.to_dense(adj_tensor,validate_indices=False) # need to figure out
            noise = tf.random.uniform(adj_tensor_dense.shape,minval=0, maxval=0.001)
            adj_tensor_dense += noise
            cols = tf.argsort(adj_tensor_dense,direction='DESCENDING',axis=-1)
            sampled_rows = tf.expand_dims(tf.range(adj_tensor_dense.shape[0]),-1)
            sampled_cols_0 = tf.expand_dims(cols[:,0],-1)
            sampled_cols_1 = tf.expand_dims(cols[:,1],-1)
            sampled0 = tf.concat((sampled_rows,sampled_cols_0),-1)
            sampled1 = tf.concat((sampled_rows,sampled_cols_1),-1)

            sample0_score = tf.gather_nd(mask,sampled0)
            sample1_score = tf.gather_nd(mask,sampled1)
            connect_loss = tf.reduce_sum(-(1.0-sample0_score)*tf.math.log(1.0-sample1_score)-sample0_score*tf.math.log(sample1_score))
            connect_loss = connect_loss* args.coff_connect
            loss += connect_loss


        return loss,pred_loss,size_loss
