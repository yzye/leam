#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, sys, cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from math import floor

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  #  b * s * e
    x_emb=tf.cast(x_emb,tf.float32)
    W_norm=tf.cast(W_norm,tf.float32)
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    y_emb=tf.cast(y_emb,tf.float32)
    W_class=tf.cast(W_class,tf.float32)
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc, beta = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    H_enc = tf.squeeze(H_enc)
    # H_enc=tf.cast(H_enc,tf.float32)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # b * c
    logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    prob = tf.nn.softmax(logits)
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    y_pred = tf.argmax(prob, 1)
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)) + class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_y, logits=logits_class))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_norm, global_step, beta, prob, y_pred

def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def embedding_class(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W

def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1) # b * s * 1
    x_emb_0 = tf.squeeze(x_emb,) # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask) # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, axis=2) # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, axis = 0) # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm) # b * s * c
    x_full_emb = x_emb_0
    Att_v = tf.contrib.layers.conv2d(G, num_outputs=opt.num_class,kernel_size=[opt.ngram], padding='SAME',activation_fn=tf.nn.relu) #b * s *  c

    Att_v = tf.reduce_max(Att_v, axis=-1, keepdims=True)
    Att_v_max = partial_softmax(Att_v, x_mask, 1, 'Att_v_max') # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)  
    return H_enc, Att_v_max

def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return zip(range(len(minibatches)), minibatches)

def prepare_data_for_emb(seqs_x, opt):
    maxlen = opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token
    return x, x_mask

def load_class_embedding( wordtoidx, opt):
    print("load class embedding")
    name_list = [ k.lower().split(' ') for k in opt.class_name]
    id_list = [ [ wordtoidx[i] for i in l] for l in name_list]
    value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
    value_mean = [ np.mean(l,0)  for l in value_list]
    return np.asarray(value_mean)

def partial_softmax(logits, weights, dim, name,):
    with tf.name_scope('partial_softmax'):
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keepdims=True)
        partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
        return partial_softmax_score
    
def key_words(inputs, beta, opt, ixtoword):
        
        beta = beta.flatten()
        beta.shape = (len(inputs),opt.maxlen)
    
        index = [np.argmax(a) for a,n in zip(beta,inputs)]
        int_words = [sen[i] for i,sen in zip(index,inputs)]
    
        key_words = [ixtoword[int_word] for int_word in int_words]
        
        return key_words

def key_words_number(inputs, beta, opt, ixtoword, key_num=5):
            
    beta = beta.flatten()
    beta.shape = (len(inputs),opt.maxlen)
        
    index = np.array([bb.argsort()[::-1][:key_num] for bb in beta])
    int_words = [[sen[ii] for ii in i] for i,sen in zip(index,inputs)]
        
    key_words = [[ixtoword[iw] for iw in int_w if iw != 0] for int_w in int_words]
            
    return key_words

def key_words_threshold(inputs, beta, opt, ixtoword, threshold=0.7):
        
    beta = beta.flatten()
    beta.shape = (len(inputs),opt.maxlen)
        
#   index = np.array([np.where(bb>threshold) for bb in beta])
    index = np.array([list(np.where(bb>threshold)[0]) for bb in beta])
    int_words = [[sen[ii] for ii in i] for i,sen in zip(index,inputs)]
        
    key_words = [[ixtoword[iw] for iw in int_w] for int_w in int_words]
            
    return key_words

class Options(object):
    def __init__(self):
        #self.GPUID = 1
        #self.dataset = 'yahoo'
        self.fix_emb = True
        #self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 305
        self.n_words = None
        self.embed_size = 300
        self.lr = 1e-3
        self.batch_size = 80
        self.max_epochs = 1
        self.dropout = 0.5
        #self.part_data = False
        self.portion = 1.0 
        self.save_path = "./save/"
        self.log_path = "./log/"
        
        #self.print_freq = 100
        self.valid_freq = 100
        self.embpath = "./data/mbu_glove.p"
        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

class LEAMModel(object):

    def __init__(self, opt):
        
        self.opt = opt
        self.num_class = None
        #self.embpath = "./data/yahoo_glove.p"
        self.embpath = opt.embpath
        self.wordtoix = None
        self.ixtoword = None
        self.probability = []
    
    def fit(self, X, y, wordtoix, ixtoword, class_name):
        
        self.wordtoix = wordtoix
        self.ixtoword = ixtoword
        
        opt = self.opt
        train, train_lab = X, np.array(y, dtype='float32')
        
        self.num_class = train_lab.shape[1]
        opt.num_class = train_lab.shape[1]
        opt.n_words = len(self.ixtoword)
        
        opt.class_name = class_name
        
        try:
            opt.W_emb = np.array(cPickle.load(open(self.embpath, 'rb')),dtype='float32')
            opt.W_class_emb =  load_class_embedding(self.wordtoix, opt)
        except IOError:
            print('No embedding file found.')
            opt.fix_emb = False        
        
        tf.reset_default_graph()
        
        graph = tf.Graph()
        
        #with graph.device('/gpu:1'):
        with graph.as_default():
            
            x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
            x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
            keep_prob = tf.placeholder(tf.float32,name='keep_prob')
            y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
            class_penalty_ = tf.placeholder(tf.float32, shape=())
            accuracy_, loss_, train_op, W_norm_, global_step, _, _, _ = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
        
            saver = tf.train.Saver()
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
        config.gpu_options.allow_growth = True
        
        np.set_printoptions(precision=3)
        np.set_printoptions(threshold=np.inf)
        
        #saver = tf.train.Saver()
        
        uidx = 0
        
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(tf.global_variables_initializer())
        
            try:
                for epoch in range(opt.max_epochs):
                    print("Starting epoch %d" % epoch)
                    kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                    for _, train_index in kf:
                        uidx += 1
                        sents = [train[t] for t in train_index]
                        x_labels = [train_lab[t] for t in train_index]
                        x_labels = np.array(x_labels)
                        x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                        x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                        _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:opt.class_penalty})

                        if uidx % opt.valid_freq == 0:
                            train_correct = 0.0
                            # sample evaluate accuaccy on 500 sample data
                            kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                            for _, train_index in kf_train:
                                train_sents = [train[t] for t in train_index]
                                train_labels = [train_lab[t] for t in train_index]
                                train_labels = np.array(train_labels)
                                train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                                x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)  
                                train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

                                train_correct += train_accuracy * len(train_index)

                            train_accuracy = train_correct / 500

                            print("Iteration %d: Training loss %f " % (uidx, loss))
                            print("Train accuracy %f " % train_accuracy)
                    saver.save(sess, opt.save_path, global_step=epoch)                        
            except KeyboardInterrupt:
                print('Training interupted')
    
    def predict(self, X):
        
        opt = self.opt
        test = X
        test_lab = np.zeros([np.array(test).shape[0],self.num_class,1])
        
        try:
            opt.W_emb = np.array(cPickle.load(open(self.embpath, 'rb')),dtype='float32')
            opt.W_class_emb =  load_class_embedding( self.wordtoix, opt)
        
        except IOError:
            print('No embedding file found.')
            opt.fix_emb = False
                
        graph = tf.Graph()
        
        with graph.as_default():
            
            x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
            x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
            keep_prob = tf.placeholder(tf.float32,name='keep_prob')
            y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
            class_penalty_ = tf.placeholder(tf.float32, shape=())
            accuracy_, loss_, train_op, W_norm_, global_step, beta, prob, y_pred = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
        
            saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
        config.gpu_options.allow_growth = True
        
        np.set_printoptions(precision=3)
        np.set_printoptions(threshold=np.inf)
        
        with tf.Session(graph=graph, config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('save'))
            
            kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=False)
            
            predictions = []
            keys = []
                        
            for _, test_index in kf_test:

                test_sents = [test[t] for t in test_index]
                test_labels = [test_lab[t] for t in test_index]
                test_labels = np.array(test_labels)
                test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)

                b, y_p, p = sess.run([beta, y_pred, prob],feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0, class_penalty_: 0.0})
                
                key = key_words_number(x_test_batch, b, opt, self.ixtoword)
                
                for i in y_p:
                    predictions.append(i)
                
                for j in key:
                    keys.append(j)
                
                for pp in p:
                    self.probability.append(pp)
                #predictions.append(y_p)
                #keys.append(key)
                
                #for i,j in zip(y_p,key):
                #    print("Prediction:", i,"keywords:",[jj for jj in j])
                #    print("")
        
        return predictions, keys
      
    def predict_proba(self, X=False):
        
        if (not X):            
            return self.probability
        
        else:            
            opt = self.opt
            test = X
            test_lab = np.zeros([np.array(test).shape[0],self.num_class,1],dtype=int)
    
            try:
                opt.W_emb = np.array(cPickle.load(open(self.embpath, 'rb')),dtype='float32')
                opt.W_class_emb =  load_class_embedding( self.wordtoix, opt)
            
            except IOError:
                print('No embedding file found.')
                opt.fix_emb = False        
            
            graph = tf.Graph()
            
            with graph.as_default():
                
                x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
                x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
                keep_prob = tf.placeholder(tf.float32,name='keep_prob')
                y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
                class_penalty_ = tf.placeholder(tf.float32, shape=())
                accuracy_, loss_, train_op, W_norm_, global_step, _, prob, _ = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
            
                saver = tf.train.Saver()
    
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
            config.gpu_options.allow_growth = True
            
            np.set_printoptions(precision=3)
            np.set_printoptions(threshold=np.inf)
            
            with tf.Session(graph=graph, config=config) as sess:
                saver.restore(sess, tf.train.latest_checkpoint('save'))
                
                kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=False)
                
                probability=[]
                
                for _, test_index in kf_test:
    
                    test_sents = [test[t] for t in test_index]
                    test_labels = [test_lab[t] for t in test_index]
                    test_labels = np.array(test_labels)
                    test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                    x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)
    
                    p = sess.run(prob, feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0, class_penalty_: 0.0})
                    #print(p)
                    #probability.append(p)
                    
                    for pp in p:
                        probability.append(pp)
            
            return probability

def load_data(path="./data/mbu.p"):
    
    with (open(path, "rb")) as openfile:
        while True:
            try:
                x = cPickle.load(openfile)
            except EOFError:
                break

        X, y, test_X, test_lab = x[0], x[3], x[2], x[5]
        wordtoix, ixtoword = x[6], x[7]

#    class_name = ['Society Culture',
#            'Science Mathematics',
#            'Health' ,
#            'Education Reference' ,
#            'Computers Internet' ,
#            'Sports' ,
#            'Business Finance' ,
#            'Entertainment Music' ,
#            'Family Relationships' ,
#            'Politics Government']
    class_name = ['Very Satisfied','Good','OK','Bad','Dissatisfied']
    
    return X, y, test_X, test_lab, wordtoix, ixtoword, class_name
            
def main():                
    
    GPUID = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
    
    #path = None
    X, y, test_X, test_lab, wordtoix, ixtoword, class_name = load_data()
    
    opt = Options()           
    Model=LEAMModel(opt)
    
    Model.fit(X, y, wordtoix, ixtoword, class_name)
    
    predictions, keys = Model.predict(test_X)
    print_keys(predictions, keys)
    probability = Model.predict_proba()
    print(probability)
    
    test_accuracy = compute_acc(predictions,test_lab)
    print("Test accuracy %f " % test_accuracy)
    
def compute_acc(predictions,test_lab):  
    
    tl=np.squeeze(test_lab)
    t=np.argmax(tl, axis=1)
    
    s = [1 for i,j in zip(predictions,t) if i==j]
    test_accuracy = 1.0*sum(s)/len(test_lab)
    
    return test_accuracy

def print_keys(predictions, keys):
    
    for i,j in zip(predictions, keys):
        print("Predictions: %d Keys: %s " % (i,j))
        print("")
    

if __name__ == '__main__':
    main()

