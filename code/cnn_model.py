import os
import cPickle
import random
import numpy as np
import scipy.sparse as sp
from w2v import load_word2vec

import theano
import theano.tensor as T
import lasagne

class CNN_model(object):
    def __init__(self, args):
        # Model Hyperparameters
        self.sequence_length = args.sequence_length
        self.embedding_dim = args.embedding_dim
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.pooling_units = args.pooling_units
        self.pooling_type = args.pooling_type
        self.hidden_dims = args.hidden_dims

        # Training Hyperparameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        # Model variation and w2v pretrain type
        self.model_variation = args.model_variation        # CNN-rand | CNN-pretrain
        self.pretrain_type = args.pretrain_type

        # Fix random seed
        np.random.seed(1126)
        random.seed(1126)
        self.rng = T.shared_randomstreams.RandomStreams(seed=1126)

    def add_data(self, X_trn, Y_trn):
        """ addd data to the model.
        """
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.output_dim = Y_trn.shape[1]

    def add_pretrain(self, vocabulary, vocabulary_inv):
        print 'model_variaton:', self.model_variation
        if self.model_variation=='pretrain':
            embedding_weights = load_word2vec(self.pretrain_type, vocabulary_inv, self.embedding_dim)
        elif self.model_variation=='random':
            embedding_weights = None
        else:
            raise ValueError('Unknown model variation')
        self.embedding_weights = embedding_weights
        self.vocab_size = len(vocabulary)


    def build_train(self):
        """build the model. This method should be called after add_data()
        """
        X = T.imatrix('X')                  # batch_size * sequence_length
        Y = T.imatrix('Y')                  # batch_size * output_dim

        # input layer, word embedding layer, and dropout
        l_x_in = lasagne.layers.InputLayer(shape=(None, self.sequence_length), input_var=X)
        if self.model_variation == 'pretrain':
            l_embed = lasagne.layers.EmbeddingLayer(l_x_in, input_size=self.vocab_size, output_size=self.embedding_dim, W=self.embedding_weights)
        elif self.model_variation == 'random':
            l_embed = lasagne.layers.EmbeddingLayer(l_x_in, input_size=self.vocab_size, output_size=self.embedding_dim)
        else:
            raise NotImplementedError('Unknown model_variation!')
        l_embed_dropout = lasagne.layers.DropoutLayer(l_embed, p=0.25)

        # convolution layer
        convs = []
        for fsz in self.filter_sizes:
            l_conv = lasagne.layers.Conv1DLayer(l_embed_dropout,
                                                num_filters=self.num_filters,
                                                filter_size=fsz,
                                                stride=2)
            l_conv_shape = lasagne.layers.get_output_shape(l_conv)
            pool_size = l_conv_shape[-1] // self.pooling_units
            if self.pooling_type == 'average':
                l_pool = lasagne.layers.Pool1DLayer(l_conv, pool_size, stride=None, mode='average_inc_pad')
            elif self.pooling_type == 'max':
                l_pool = lasagne.layers.MaxPool1DLayer(l_conv, 2, stride=1)
            else:
                raise NotImplementedError('Unknown pooling_type!')
            l_flat = lasagne.layers.flatten(l_pool)
            convs.append(l_flat)
        if len(self.filter_sizes)>1:
            l_conv_final = lasagne.layers.concat(convs)
        else:
            l_conv_final = convs[0]

        # Final hidden layer
        l_hidden = lasagne.layers.DenseLayer(l_conv_final, num_units=self.hidden_dims, nonlinearity=lasagne.nonlinearities.rectify)
        l_hidden_dropout = lasagne.layers.DropoutLayer(l_hidden, p=0.5)

        l_y = lasagne.layers.DenseLayer(l_hidden_dropout, num_units=self.output_dim, nonlinearity=lasagne.nonlinearities.sigmoid)
        params = lasagne.layers.get_all_params(l_y, trainable=True)
        self.network = l_y

        # Objective function and update params
        Y_pred = lasagne.layers.get_output(l_y)
        loss = lasagne.objectives.binary_crossentropy(Y_pred, Y).mean()
        updates = lasagne.updates.adam(loss, params)
        self.train_fn = theano.function([X, Y], [loss], updates=updates)

    def build_predict(self, epoch):
        """build the model. This method should be called after add_data()
        """
        X = T.imatrix('X')                  # batch_size * sequence_length
        Y = T.imatrix('Y')                  # batch_size * output_dim

        # input layer, word embedding layer, and dropout
        l_x_in = lasagne.layers.InputLayer(shape=(None, self.sequence_length), input_var=X)
        if self.model_variation == 'pretrain':
            l_embed = lasagne.layers.EmbeddingLayer(l_x_in, input_size=self.vocab_size, output_size=self.embedding_dim, W=self.embedding_weights)
        elif self.model_variation == 'random':
            l_embed = lasagne.layers.EmbeddingLayer(l_x_in, input_size=self.vocab_size, output_size=self.embedding_dim)
        else:
            raise NotImplementedError('Unknown model_variation!')
        l_embed_dropout = lasagne.layers.DropoutLayer(l_embed, p=0.25)

        # convolution layer
        convs = []
        for fsz in self.filter_sizes:
            l_conv = lasagne.layers.Conv1DLayer(l_embed_dropout,
                                                num_filters=self.num_filters,
                                                filter_size=fsz,
                                                stride=2)
            l_conv_shape = lasagne.layers.get_output_shape(l_conv)
            pool_size = l_conv_shape[-1] // self.pooling_units
            if self.pooling_type == 'average':
                l_pool = lasagne.layers.Pool1DLayer(l_conv, pool_size, stride=None, mode='average_inc_pad')
            elif self.pooling_type == 'max':
                l_pool = lasagne.layers.MaxPool1DLayer(l_conv, 2, stride=1)
            else:
                raise NotImplementedError('Unknown pooling_type!')
            l_flat = lasagne.layers.flatten(l_pool)
            convs.append(l_flat)
        if len(self.filter_sizes)>1:
            l_conv_final = lasagne.layers.concat(convs)
        else:
            l_conv_final = convs[0]

        # Final hidden layer
        l_hidden = lasagne.layers.DenseLayer(l_conv_final, num_units=self.hidden_dims, nonlinearity=lasagne.nonlinearities.rectify)
        l_hidden_dropout = lasagne.layers.DropoutLayer(l_hidden, p=0.5)
        l_y = lasagne.layers.DenseLayer(l_hidden_dropout, num_units=self.output_dim, nonlinearity=lasagne.nonlinearities.sigmoid)
        self.network = l_y
        self.load_params(epoch)

        # predict output and hidden layer
        X_hidden = lasagne.layers.get_output(l_hidden, deterministic=True)
        Y_pred = lasagne.layers.get_output(l_y, deterministic=True)
        self.predict_fn = theano.function([X], [X_hidden, Y_pred])

    def train(self):
        nr_trn_num = self.X_trn.shape[0]
        nr_batches = int(np.ceil(nr_trn_num / float(self.batch_size)))
        #nr_batches = nr_trn_num // self.batch_size
        trn_loss = []
        for batch_idx in np.random.permutation(xrange(nr_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, nr_trn_num)
            #end_idx = (batch_idx + 1) * self.batch_size
            X = self.X_trn[start_idx:end_idx]
            Y = self.Y_trn[start_idx:end_idx].toarray()
            loss = self.train_fn(X, Y)
            trn_loss.append(loss)
        return np.mean(loss)

    # For large output_dim cases (mini-batch prediction for top-k classes)
    def predict(self, X_tst=None, batch_size=8192, top_k=50, max_k=None):
        nr_tst_num = X_tst.shape[0]
        nr_batches = int(np.ceil(nr_tst_num / float(batch_size)))
        row_idx_list, col_idx_list, val_idx_list = [], [], []
        #X_pred = np.zeros((X_tst.shape[0], self.hidden_dims))
        for batch_idx in xrange(nr_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nr_tst_num)
            X_hidden, Y_pred = self.predict_fn(X_tst[start_idx:end_idx])
            #X_pred[start_idx:end_idx, :] = X_hidden
            for i in xrange(Y_pred.shape[0]):
                sorted_idx = np.argpartition(-Y_pred[i, :], top_k)[:top_k]
                row_idx_list += [i + start_idx] * top_k
                col_idx_list += (sorted_idx).tolist()
                val_idx_list += Y_pred[i, sorted_idx].tolist()
        m = max(row_idx_list) + 1
        n = max_k
        Y_pred = sp.csr_matrix((val_idx_list, (row_idx_list, col_idx_list)), shape=(m, n))
        return Y_pred
        #return X_pred, Y_pred

    def store_params(self, epoch):
        """serialize the model parameters in self.model_file.
        """
        assert(os.path.isdir(self.model_file))
        model_path = os.path.join(self.model_file, 'iter-%d' % (epoch))
        fout = open(model_path, 'wb')
        params = lasagne.layers.get_all_param_values(self.network)
        cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
        fout.close()

    def load_params(self, epoch):
        """load the model parameters in self.model_file.
        """
        assert(os.path.isdir(self.model_file))
        model_path = os.path.join(self.model_file, 'iter-%d' % (epoch))
        fin = open(model_path, 'rb')
        params = cPickle.load(fin)
        lasagne.layers.set_all_param_values(self.network, params)
        fin.close()


