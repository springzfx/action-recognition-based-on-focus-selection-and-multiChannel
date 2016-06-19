import numpy as np
import theano
import theano.tensor as T
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
import os
import time

from block.layer import ff_init,ff_build,dropout_layer,LSTM_init,LSTM_layer,WTA_Layer,SaliencyFgbg_init,SaliencyFgbg_block
from util import load_params,share_params,unzip,itemlist


def init_params(options):
    ctxdim=options['featureMaps'];
    hdim=options['hdim'];
    actionNum=options['actions'];
    fdim=options['fdim'];

    # print actionNum;
    params=OrderedDict();

    params=SaliencyFgbg_init(params,options['locations'],options['featureMaps'],prefix="recog",name='saliencyFgbg');
    
    params=ff_init(params,ctxdim,fdim,prefix="recog",name='channel0');

    params=LSTM_init(params,ctxdim,hdim,prefix="recog",name='lstm1');
    params=ff_init(params,hdim,fdim,prefix="recog",name='channel1');

    #output
    params=ff_init(params,fdim,actionNum,prefix="recog",name='output');


    # params['recog/saliencyFgbg_w']=theano.gradient.grad_clip(params['recog/saliencyFgbg_w'],-0.1,0.1);
    tparams=share_params(params);
    # loading params if need
    loadfrom=options['loadfrom'];
    if options['load']: 
        if os.path.exists(loadfrom):
            print "loading model parameters from ",loadfrom;
            tparams = load_params(loadfrom, tparams,strict=False);
        else:
            print "Not exist ",loadfrom;

    
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


    _x=x.reshape([n_steps,n_samples,location_dim,feature_dim]);
    # _x=_x/_x.max(-1,keepdims=True);

    feature,alpha=SaliencyFgbg_block(tparams,_x,prefix="recog",name='saliencyFgbg');

    # feature=_x.mean(2);
    # feature=feature/feature.max(1,keepdims=True);
    # feature=feature.reshape([n_steps,n_samples,feature_dim]);
    # alpha=alpha.reshape([n_steps,n_samples,location_dim]);
  	
    feature=feature+use_noise*trng.normal(feature.shape,avg=0,std=0.0001,dtype=feature.dtype);
    if use_dropout:	feature = dropout_layer(feature, use_noise, trng);
    if use_wta: feature=WTA_Layer(feature,4,2,ndim=options['featureMaps']); 
    h,c=LSTM_layer(tparams,feature,prefix='recog',name="lstm1");
    if use_dropout: h = dropout_layer(h, use_noise, trng);
    if use_wta: h=WTA_Layer(h,4,2,ndim=options['featureMaps']); 


    f0=ff_build(tparams,feature,prefix="recog",name='channel0',active="linear");
    f1=ff_build(tparams,h,prefix="recog",name='channel1',active="linear");
    f0=T.tanh((f0+f1)/2);

    # f0=f0+use_noise*trng.normal(f0.shape,avg=0,std=0.01,dtype=f0.dtype);
    if use_dropout:	f0 = dropout_layer(f0, use_noise, trng);
    lin=ff_build(tparams,f0,prefix="recog",name='output',active="linear"); # n_steps*n_samples*actionNum
    # lin=T.switch(lin>5,5,lin);
    # lin=T.switch(lin<-5,-5,lin);
    probs=T.nnet.softmax(lin.reshape([-1,actionNum]));
    probs=probs.reshape([n_steps,n_samples,actionNum]);

    """compute cost"""
    cost=0;
    # cross entropy
    # _probs=T.switch(probs>0.9,1,probs);
    # _probs=T.switch(probs>0.99,1,probs);
    # _probs=T.switch(probs<0.1/actionNum,1,_probs);

    entropy_cost=-y*T.log(probs+1e-8);
    entropy_cost = (entropy_cost.sum(2)*mask).mean(0).sum()*100;
    cost+=entropy_cost;

    # direct cost
    # direct_cost=((probs-y)**2).sum(2).mean(0).sum()*100;
    # cost+=direct_cost;

    # weight decay
    weight_decay = 0.;
    if decay_c > 0.:
    	decay_c = theano.shared(np.float32(decay_c), name='decay_c');
        for kk, vv in tparams.iteritems():
    	    weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c;
        cost+=weight_decay;

    # alpha cost
    zero=theano.shared(np.float32(0.), name='zero');
    alpha_cost=theano.ifelse.ifelse( T.gt(n_steps,1),  ((alpha[1:,:,:]-alpha[:-1,:,:])**2).mean(0).sum()*100*0.1, zero);
    alpha_s=T.sort(alpha);
    alpha_cost+=((alpha_s[:,:,-9:].sum(-1)-1.0)**2).mean(0).sum()*100;
    cost+=alpha_cost;

    """Predictions"""
    preds = T.sum(probs[-last_n:,:,:],axis=0);
    preds = T.argmax(preds,axis=1); # n_samples
    # preds=T.argmax(probs[-last_n:,:,:],axis=2);
  
    return cost,preds,[alpha],[x,mask,y],use_noise;

