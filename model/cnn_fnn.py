import numpy as np
import theano
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import os
import time

from block.layer import ff_init,ff_build,dropout_layer,LSTM_init,LSTM_layer
from block.cnn import CNN_model,cnn_init,bulid_cnnBlock
from util import load_params,share_params,unzip,itemlist,lasagne_net_init,lasagne_net_block

import lasagne
# from google_net.googlenet import build_model as build_cnn_model
from block.cnn_1 import CNN_model as build_cnn_model
CNN_NET=build_cnn_model();
CNN_outputLayerName='conv11/3x3_s1';
CNN_block=lasagne_net_block(CNN_NET,CNN_outputLayerName);

# import pickle
# cnn_params = pickle.load(open('/home/fancy/Desktop/project/model/google_net/blvc_googlenet.pkl')); # load pretrained parameters
# lasagne.layers.set_all_param_values(CNN_NET['prob'], cnn_params['param values']);
print "cnn ready";


def init_params(options):
    featureMaps=options['featureMaps'];
    actionNum=options['actions'];
    fdim=options['fdim'];

    params=OrderedDict();

    params=ff_init(params,featureMaps,fdim,prefix="recog",name='fc0');
    params=ff_init(params,fdim,fdim,prefix="recog",name='fc1');
    
    #output
    params=ff_init(params,fdim,actionNum,prefix="recog",name='output');


    tparams=share_params(params);
    tparams=lasagne_net_init(tparams,CNN_NET,CNN_outputLayerName,prefix='recog',name='cnn');

    # loading params if need
    loadfrom=options['loadfrom'];
    if options['load'] and os.path.exists(loadfrom):
        print "loading model parameters from ",loadfrom;
        tparams = load_params(loadfrom, tparams,strict=False);

    return tparams


def build_model(options,tparams):
    """Build up the whole computation graph
    Input is the features extracted from googleNet.
    """ 
 
    last_n = options['last_n'];
    actionNum=options['actions'];
    decay_c=options['decay_c'];
    use_dropout=options['use_dropout'];
    use_wta=options['use_wta'];
    featureMaps=options['featureMaps'];
    videosize=options['videosize'];

    trng = RandomStreams(1234);
    use_noise = theano.shared(np.float32(0.));


    """combine model"""
    x=T.ftensor3('x');  
    n_steps=x.shape[0]; n_samples=x.shape[1];
    mask = T.fmatrix('mask');
    y=T.ftensor3('y');  # one hot vector,n_steps*n_samples*actionNum

    _x=x.reshape([n_steps*n_samples,3,videosize,videosize]);
    _x=_x+use_noise*trng.normal(_x.shape,avg=0,std=0.1,dtype=_x.dtype);
    feature=CNN_block(_x);
    feature=feature.reshape([n_steps,n_samples,featureMaps]);
    if use_dropout: feature = dropout_layer(feature, use_noise, trng);

    f0=ff_build(tparams,feature,prefix="recog",name='fc0',active="tanh");
    
    f1=ff_build(tparams,f0,prefix="recog",name='fc1',active="tanh");
    # if use_dropout: f0 = dropout_layer(f0, use_noise, trng);

    lin=ff_build(tparams,f1,prefix="recog",name='output',active="linear"); # n_steps*n_samples*actionNum
    probs=T.nnet.softmax(lin.reshape([-1,actionNum]));
    probs=probs.reshape([n_steps,n_samples,actionNum]);

    """compute cost"""
    cost=0;
    # cross entropy
    entropy_cost=-y*T.log(probs);
    entropy_cost = (entropy_cost.sum(2)*mask).mean(0).sum();

    cost+=entropy_cost;

    # weight decay
    weight_decay = 0.;
    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c), name='decay_c');
        for vv in itemlist(tparams):
            # if (vv.ndim==1):
            weight_decay += (vv ** 2).sum()
            # else:
                # weight_decay+=(vv.sum()-vv.shape[0])**2;
            # weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c;
        cost+=weight_decay;

    """Predictions"""
    preds = T.sum(probs[-last_n:,:,:],axis=0);
    preds = T.argmax(preds,axis=1); # n_samples
    # preds=T.argmax(probs[-last_n:,:,:],axis=2);
  
    return cost,preds,[],[x,mask,y],use_noise;

