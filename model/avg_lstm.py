import numpy as np
import theano
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import os
import time

from block.layer import LSTM_init,LSTM_layer,ff_init,ff_build,dropout_layer,WTA_Layer
from util import load_params,share_params,unzip,itemlist


def init_params(options):
    ctxdim=options['featureMaps'];
    hdim=options['hdim'];
    actionNum=options['actions'];
    fdim=options['fdim'];

    params=OrderedDict();
    params=LSTM_init(params,ctxdim,hdim,prefix="recog",name='lstm');
    params=ff_init(params,hdim,fdim,prefix="recog",name='fullconn');
    params=ff_init(params,fdim,actionNum,prefix="recog",name='output');

    tparams=share_params(params);
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

    location_dim=options['locations'];
    feature_dim=options['featureMaps'];

    trng = RandomStreams(1234);
    use_noise = theano.shared(np.float32(0.));


    """combine model"""
    x=T.ftensor3('x');  
    n_steps=x.shape[0]; n_samples=x.shape[1];
    mask = T.fmatrix('mask');
    y=T.ftensor3('y');  # one hot vector,n_steps*n_samples*actionNum

    _x=x.reshape([n_steps*n_samples,location_dim,feature_dim]);
    feature=_x.mean(1);
    # feature=feature/feature.max(1,keepdims=True);
    feature=feature.reshape([n_steps,n_samples,feature_dim]);
    feature=feature+use_noise*trng.normal(feature.shape,avg=1,std=0.01,dtype=feature.dtype);  #noisy
  	
    if use_dropout:	feature = dropout_layer(feature, use_noise, trng);
    if use_wta: feature=WTA_Layer(feature,4,2,ndim=options['featureMaps']); 
    h,c=LSTM_layer(tparams,feature,prefix='recog',name="lstm");
    if use_dropout: h = dropout_layer(h, use_noise, trng);
    if use_wta: h=WTA_Layer(h,4,2,ndim=options['featureMaps']); 


    f0=ff_build(tparams,h,prefix="recog",name='fullconn',active="tanh");

    # f0=f0+use_noise*trng.normal(f0.shape,avg=0,std=0.0001,dtype=f0.dtype);
    if use_dropout: f0 = dropout_layer(f0, use_noise, trng);

    lin=ff_build(tparams,f0,prefix="recog",name='output',active="linear"); # n_steps*n_samples*actionNum
    probs=T.nnet.softmax(lin.reshape([-1,actionNum]));
    probs=probs.reshape([n_steps,n_samples,actionNum]);

    """Predictions"""
    preds = T.sum(probs[-last_n:,:,:],axis=0);
    preds = T.argmax(preds,axis=1); # n_samples
    # preds=T.argmax(probs[-last_n:,:,:],axis=2);

    """compute cost"""
    cost=0;
    # cross entropy
    entropy_cost=-y*T.log(probs+1e-8);
    entropy_cost = (entropy_cost.sum(2)*mask).mean(0).sum()*100;

    cost+=entropy_cost;
    # weight decay
    weight_decay = 0.;
    if decay_c > 0.:
    	decay_c = theano.shared(np.float32(decay_c), name='decay_c');
        for kk, vv in tparams.iteritems():
    	    weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c;
        cost+=weight_decay;
 
  
    return cost,preds,[h,c],[x,mask,y],use_noise;

