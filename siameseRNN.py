import os
import time
import itertools as it
import math

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss

import tools
from tools import EpochFinished


class SiameseRNN(object):

    def __init__(self, embedding_size, n_hidden_RNN=256, do_train=True):

        self.embedding_size = embedding_size
        self.n_hidden_RNN = n_hidden_RNN
        self.create_graph()
        if do_train: self.create_optimizer_graph(self.cost)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d)+'/train/')
        self.test_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d)+'/test/')
        self.merged = tf.summary.merge_all()
        
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(init_op)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(),
                                    max_to_keep = 1000)

    # --------------------------------------------------------------------------
    def __enter__(self):
        return self

    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()

    # --------------------------------------------------------------------------
    def create_graph(self):

        print('Create graph')
        self.input_graph()

        self.RNN_out1, self.RNN_out2 = self.create_RNN_graph(n_layers=3)

        self.out1, self.out2 = self.create_FC_graph(self.RNN_out1, self.RNN_out2)

        self.preds = tf.exp(-tf.reduce_mean(tf.abs(self.out1 - self.out2), axis=1))

        self.cost = self.create_cost_graph(preds=self.preds, targets=self.targets)

        print('Done!')

    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        self.question1 = tf.placeholder(tf.float32,
            shape=[None, None, self.embedding_size],
            name='question1')

        self.question2 = tf.placeholder(tf.float32,
            shape=[None, None, self.embedding_size],
            name='question1')

        self.seq_lengths1 = tf.placeholder(tf.int32, name='seq_lengths1')
        
        self.seq_lengths2 = tf.placeholder(tf.int32, name='seq_lengths2')

        self.targets = tf.placeholder(tf.float32, name='targets')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
    
    # --------------------------------------------------------------------------
    def biRNN(self, inputs, seq_lengths, n_layers, scope):
        print('\t\tbiRNN')
        with tf.variable_scope(scope):
            # inputs b x h x embedding_size (h is variable value)
            # sequence_length b
            cell = tf.contrib.rnn.GRUCell(self.n_hidden_RNN)

            fw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                output_keep_prob=self.keep_prob)

            bw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                output_keep_prob=self.keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=seq_lengths,
                dtype=tf.float32)
            return states[0][-1] + states[1][-1] # tuple of fw and bw states with shape b x hRNN

    # --------------------------------------------------------------------------
    def create_RNN_graph(self, n_layers):
        print('\tcreate_RNN_graph')
        with tf.variable_scope('RNN_graph'):
            out1 = self.biRNN(inputs=self.question1,
                seq_lengths=self.seq_lengths1, n_layers=n_layers, scope='RNN1')
            out2 = self.biRNN(inputs=self.question2,
                seq_lengths=self.seq_lengths2, n_layers=n_layers, scope='RNN2')
            return out1, out2

    # --------------------------------------------------------------------------
    def FC_out(self, inputs):
        print('\t\tFC_out')
        with tf.variable_scope(None, 'FC_out'):
            out_FC1 = tf.layers.dense(
                inputs=inputs,
                units=self.n_hidden_RNN,
                activation=tf.nn.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            out_FC1_d = tf.nn.dropout(out_FC1, self.keep_prob)

            out_FC2 = tf.layers.dense(
                inputs=out_FC1_d,
                units=self.n_hidden_RNN,
                activation=tf.nn.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            return out_FC2
    # --------------------------------------------------------------------------
    def create_FC_graph(self, out1, out2):
        print('\tcreate_FC_graph')
        with tf.variable_scope('FC_graph'):
            out_FC1 = self.FC_out(out1)
            out_FC2 = self.FC_out(out2)
            return out_FC1, out_FC2
    

    # --------------------------------------------------------------------------
    def create_cost_graph(self, preds, targets):
        print('\tcreate_cost_graph')
        preds = tf.clip_by_value(preds, 1e-5, 1-1e-5)
        self.cross_entropy = -tf.reduce_mean(targets*tf.log(preds)+
            (1-targets)*tf.log(1-preds))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])

        preds_ = tf.cast(tf.greater(preds, 0.5), tf.float32)
        tp = tf.reduce_sum(preds_*targets)
        fp = tf.reduce_sum((1-targets)*preds_)
        fn = tf.reduce_sum((1-preds_)*targets)
        precission = tp/(tp+fp+1e-5)
        recall = tp/(tp+fn+1e-5)
        self.pr_re = [precission, recall]

        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('L2 loss', self.L2_loss)
        tf.summary.scalar('precission', precission)
        tf.summary.scalar('recall', recall)
        
        return self.cross_entropy + self.L2_loss
    
    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('\tcreate_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            self.train = optimizer.minimize(cost)

    ############################################################################  
    def save_model(self, path = 'beat_detector_model', step = None):
        p = self.saver.save(self.sess, path, global_step = step)
        print("\tModel saved in file: %s" % p)

    ############################################################################
    def load_model(self, path):
        #path is path to file or path to directory
        #if path it is path to directory will be load latest model
        load_path = os.path.splitext(path)[0]\
        if os.path.isfile(path) else tf.train.latest_checkpoint(path)
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)

    ############################################################################
    def train_(self, data_loader, keep_prob, weight_decay, learn_rate_start,
        learn_rate_end, batch_size, n_iter, save_model_every_n_iter=1000,
        path_to_model='classifier'):

        print('\t\t\t\t----==== Training ====----')
        try:
            self.load_model(os.path.dirname(path_to_model))
        except:
            print('Can not load model {0}, starting new train'.format(path_to_model))
            
        start_time = time.time()
        b = math.log(learn_rate_start/learn_rate_end, n_iter) 
        a = learn_rate_start*math.pow(1, b)
        for current_iter in tqdm(range(n_iter)):
            learn_rate = a/math.pow((current_iter+1), b)
            batch = data_loader.train.sample_batch(batch_size)
            feedDict = {self.question1 : batch['questions_1'],
                        self.question2 : batch['questions_2'],
                        self.targets : batch['targets'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.seq_lengths2 : batch['seq_lengths_2'],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate}
            _, summary, _ = self.sess.run([self.train, self.merged, self.pr_re],
                feed_dict = feedDict)
            self.train_writer.add_summary(summary, current_iter)


            batch = data_loader.test.sample_batch(batch_size)
            feedDict = {self.question1 : batch['questions_1'],
                        self.question2 : batch['questions_2'],
                        self.targets : batch['targets'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.seq_lengths2 : batch['seq_lengths_2'],
                        self.keep_prob : 1,
                        self.weight_decay : weight_decay}
            _, summary, _ = self.sess.run([self.cost, self.merged, self.pr_re],
                feed_dict = feedDict)
            self.test_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path = path_to_model, step = current_iter+1)

        self.save_model(path = path_to_model, step = current_iter+1)
        print('\tTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))

    ############################################################################
    def eval_cost(self, data_loader, batch_size, path_to_model):
        print('\t\t\t\t----==== Evaluating cost ====----')
        self.load_model(os.path.dirname(path_to_model))
        data_loader.train.i = 0
        data_loader.test.i = 0
        

        def eval(mode):

            result = np.empty([0])
            while True:
                try:
                    if mode=='train':
                        batch = data_loader.train.next_batch(batch_size)
                    elif mode=='test':
                        batch = data_loader.test.next_batch(batch_size)
                    else:
                        raise ValueError('error of mode type')
                except EpochFinished:
                    break
                feedDict = {self.question1 : batch['questions_1'],
                            self.question2 : batch['questions_2'],
                            self.seq_lengths1 : batch['seq_lengths_1'],
                            self.seq_lengths2 : batch['seq_lengths_2'],
                            self.keep_prob : 1}

                res = self.sess.run(self.preds, feed_dict = feedDict)
                result = np.concatenate([result, res])

            if mode=='train':
                y_true = data_loader.train_data['is_duplicate']
            elif mode=='test':
                y_true = data_loader.test_data['is_duplicate']
            else:
                raise ValueError('error of mode type')

            return log_loss(y_true=y_true, y_pred=result)
        predicting_time = time.time()
        print('train loss=', eval(mode='train'))
        print('test loss=', eval(mode='test'))
        print('evaluating_time = ', time.time() - predicting_time)          


    #############################################################################################################
    def predict(self, batch_size, data_loader, path_to_save, path_to_model):
        predicting_time = time.time()
        print('\t\t\t\t----==== Predicting beats ====----')
        self.load_model(os.path.dirname(path_to_model))
        
        result = np.empty([0])
        data_loader.train.i = 0

        forward_pass_time = 0
        for current_iter in it.count():
            try:
                batch = data_loader.train.next_batch(batch_size)
            except EpochFinished:
                break
            feedDict = {self.question1 : batch['questions_1'],
                        self.question2 : batch['questions_2'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.seq_lengths2 : batch['seq_lengths_2'],
                        self.keep_prob : 1}

            start_time = time.time()
            res = self.sess.run(self.preds, feed_dict = feedDict)
            forward_pass_time = forward_pass_time + (time.time() - start_time)
            result = np.concatenate([result, res])
            
        result = pd.DataFrame(result, columns=['is_duplicate'])
        result.to_csv(path_to_save)
        print('\tfile saved ', path_to_save)

        print('forward_pass_time = ', forward_pass_time)
        print('predicting_time = ', time.time() - predicting_time)

# testing #####################################################################################################################
"""
path_to_file = '../data/test/AAO3CXJKEG.npy'
data = np.load(path_to_file).item()
n_chunks=128
overlap = 700 #in samples
chunked_data = utils.chunking_data(data)
"""
