import os
import argparse
import numpy as np
import timeit

import data_helpers
from cnn_model import CNN_model


def load_data(args):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv = data_helpers.load_data(args.data_path, max_length=args.sequence_length, vocab_size=args.vocab_size)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv


def gen_model_file(args):
    data_name = args.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in args.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s' % \
        (data_name, args.sequence_length, args.embedding_dim,
         fs_string, args.num_filters, args.pooling_units,
         args.pooling_type, args.hidden_dims, args.batch_size,
         args.model_variation, args.pretrain_type)
    return file_name


def main(args):
    print('-'*50)
    print('Loading data...'); start_time = timeit.default_timer();
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv = load_data(args)
    print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))

    # Building model
    # ==================================================
    print('-'*50)
    print("Building model..."); start_time = timeit.default_timer();
    model = CNN_model(args)
    model.model_file = os.path.join('./CNN_runtime_models', gen_model_file(args))
    if not os.path.isdir(model.model_file):
        os.makedirs(model.model_file)
    else:
        print('Warning: model file already exist!\n %s' % (model.model_file))

    model.add_data(X_trn, Y_trn)
    model.add_pretrain(vocabulary, vocabulary_inv)
    model.build_train()
    print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))

    # Training model
    # ==================================================
    print('-'*50)
    print("Training model..."); start_time = timeit.default_timer();
    store_params_time = 0.0;
    for epoch_idx in xrange(args.num_epochs + 1):
        loss = model.train()
        print 'Iter:', epoch_idx, 'Trn loss ', loss
        if epoch_idx % 5 == 0:
            print 'saving model...'; tmp_time = timeit.default_timer();
            model.store_params(epoch_idx)
            store_params_time += timeit.default_timer() - tmp_time
    total_time = timeit.default_timer() - start_time
    print('Total time %.4f (secs), training time %.4f (secs), IO time %.4f (secs)' \
          % (total_time, total_time - store_params_time, store_params_time))


if __name__ == '__main__':
    # Parameters
    # ==================================================
    # Model Variations. See Kim Yoon's Convolutional Neural Networks for
    # Sentence Classification, Section 3 for detail.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        help='raw data path in CPickle format', type=str,
                        default='../sample_data/rcv1_raw_small.p')
    parser.add_argument('--sequence_length',
                        help='max sequence length of a document', type=int,
                        default=500)
    parser.add_argument('--embedding_dim',
                        help='dimension of word embedding representation', type=int,
                        default=300)
    parser.add_argument('--filter_sizes',
                        help='number of filter sizes (could be a list of integer)', type=int,
                        default=[2, 4, 8], nargs='+')
    parser.add_argument('--num_filters',
                        help='number of filters (i.e. kernels) in CNN model', type=int,
                        default=32)
    parser.add_argument('--pooling_units',
                        help='number of pooling units in 1D pooling layer', type=int,
                        default=32)
    parser.add_argument('--pooling_type',
                        help='max or average', type=str,
                        default='max')
    parser.add_argument('--hidden_dims',
                        help='number of hidden units', type=int,
                        default=512)
    parser.add_argument('--model_variation',
                        help='model variation: CNN-rand or CNN-pretrain', type=str,
                        default='pretrain')
    parser.add_argument('--pretrain_type',
                        help='pretrain model: GoogleNews or glove', type=str,
                        default='glove')
    parser.add_argument('--batch_size',
                        help='number of batch size', type=int,
                        default=256)
    parser.add_argument('--num_epochs',
                        help='number of epcohs for training', type=int,
                        default=50)
    parser.add_argument('--vocab_size',
                        help='size of vocabulary keeping the most frequent words', type=int,
                        default=30000)
    args = parser.parse_args()
    main(args)



