import numpy as np
import theano
import theano.tensor as T
from util import load_params,share_params,unzip,itemlist
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import os
import time

from block.layer import LSTM_init,LSTM_layer,ff_init,ff_build,dropout_layer,Linger_init,Linger_layer

def init_params(options):
    feadim=options['featureMaps'];
    ctxdim=512;
    hdim=options['hdim'];
    actionNum=options['actions'];
    fdim=64;

    params=OrderedDict();
    #params=Saliency_init(params,ctxdim,prefix="recog");
    #params=ff_init(params,ctxdim,512,prefix="recog",name='ctx_pre');
    params=Linger_init(params,feadim,ctxdim,prefix='recog',name='linger');
    # FNN channel
    params=ff_init(params,ctxdim,fdim,prefix="recog",name='highway');
    # LSTM
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
    _x=_x+trng.normal(_x.shape,avg=0,std=0.05,dtype=_x.dtype);  #noisy
   # feature,alpha=Saliency_Layer(tparams,_x,prefix="recog");
    #alpha=alpha.reshape([n_steps,n_samples,location_dim]);
    feature=_x.mean(1);
    # feature=feature/feature.max(1,keepdims=True);
    feature=feature.reshape([n_steps,n_samples,feature_dim]);
    
 
    ctx=Linger_layer(tparams,feature,prefix='recog',name='linger');
    if use_dropout: ctx = dropout_layer(ctx, use_noise, trng);
    h,c=LSTM_layer(tparams,ctx,prefix='recog',name="lstm");
    if use_dropout: h = dropout_layer(h, use_noise, trng);
    f1=ff_build(tparams,h,prefix="recog",name='fullconn',active="linear");
    f2=ff_build(tparams,ctx,prefix="recog",name='highway',active="linear");
    # add two channels
    f1=T.tanh((f1+f2)/2);

    if use_dropout: f1 = dropout_layer(f1, use_noise, trng);
    lin=ff_build(tparams,f1,prefix="recog",name='output',active="linear"); # n_steps*n_samples*actionNum
    probs=T.nnet.softmax(lin.reshape([-1,actionNum]));
    probs=probs.reshape([n_steps,n_samples,actionNum]);


     
    """compute cost"""
    cost=0;
    # cross entropy
    entropy_cost=-y*T.log(probs+1e-8);
    entropy_cost = (entropy_cost.sum(2)*mask).mean(0).sum()*100;


    cost+=entropy_cost;
    # weight decay
    weight_decay = 0.;
    if decay_c > 0.:
    	decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        for kk, vv in tparams.iteritems():
    	    weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c;
        cost+=weight_decay;


    """Predictions"""
    preds = T.sum(probs[-last_n:,:,:],axis=0);
    preds = T.argmax(preds,axis=1); # n_samples
    # preds=T.argmax(probs[-last_n:,:,:],axis=2);
  
    

    return cost,preds,[h,c],[x,mask,y],use_noise;
