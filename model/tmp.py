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
from util import load_params,share_params,unzip,itemlist

cnn_net=CNN_model();
CNN_block=bulid_cnnBlock(cnn_net);




def init_params(options):
    ctxdim=256;
    hdim=options['hdim'];
    actionNum=options['actions'];
    fdim=options['fdim'];

    params=OrderedDict();

    # params=Saliency_init(params,ctxdim,prefix="recog",name='saliency');
    # params=Linger_init(params,ctxdim,ctxdim,prefix='recog',name='linger');

    # LSTM
    # params=LSTM_init(params,ctxdim,hdim,prefix="recog",name='lstm1');
    # params=LSTM_init(params,hdim,512,prefix="recog",name='lstm2');
    # params=LSTM_init(params,512,256,prefix="recog",name='lstm3');
    # params=LSTM_init(params,256,128,prefix="recog",name='lstm4');

    # multiChannel
    params=ff_init(params,ctxdim,fdim,prefix="recog",name='channel0');
    # params=ff_init(params,hdim,fdim,prefix="recog",name='channel1');
    # params=ff_init(params,512,fdim,prefix="recog",name='channel2');
    # params=ff_init(params,256,fdim,prefix="recog",name='channel3');
    # params=ff_init(params,128,fdim,prefix="recog",name='channel4');

    #output
    params=ff_init(params,fdim,actionNum,prefix="recog",name='output');


    tparams=share_params(params);
    tparams=cnn_init(tparams,cnn_net,prefix='recog',name='cnn');

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

    location_dim=4*4;
    feature_dim=256;


    trng = RandomStreams(1234);
    use_noise = theano.shared(np.float32(0.));


    """combine model"""
    x=T.ftensor3('x');  
    n_steps=x.shape[0]; n_samples=x.shape[1];
    mask = T.fmatrix('mask');
    y=T.ftensor3('y');  # one hot vector,n_steps*n_samples*actionNum

    _x=x.reshape([n_steps*n_samples,3,64,64]);
    feature=CNN_block(_x); # n_steps*featureMap*4*4

    feature=feature.reshape([n_steps, n_samples, feature_dim, location_dim]);
    feature=feature.mean(-1);
    feature=feature.reshape([n_steps,n_samples,feature_dim]);
    
    
    # feature=feature+use_noise*trng.normal(feature.shape,avg=0,std=0.01,dtype=feature.dtype);
    # if use_dropout: feature = dropout_layer(feature, use_noise, trng);

    # feature=Linger_layer(tparams,feature,prefix='recog',name='linger');


    #LSTM
    # h1,c1=LSTM_layer(tparams,feature,prefix='recog',name="lstm1");
    # if use_dropout: h1 = dropout_layer(h1, use_noise, trng);

#     h2,c2=LSTM_layer(tparams,h1,prefix='recog',name="lstm2");
#     if use_dropout: h2 = dropout_layer(h2, use_noise, trng);
# # 
#     h3,c3=LSTM_layer(tparams,h2,prefix='recog',name="lstm3");
#     if use_dropout: h3 = dropout_layer(h3, use_noise, trng);

#     h4,c4=LSTM_layer(tparams,h3,prefix='recog',name="lstm4");
#     if use_dropout: h4 = dropout_layer(h4, use_noise, trng);
    

    f0=ff_build(tparams,feature,prefix="recog",name='channel0',active="tanh");
    # f1=ff_build(tparams,h1,prefix="recog",name='channel1',active="linear");
    # f2=ff_build(tparams,h2,prefix="recog",name='channel2',active="linear");
    # f3=ff_build(tparams,h3,prefix="recog",name='channel3',active="linear");
    # f4=ff_build(tparams,h4,prefix="recog",name='channel4',active="linear");
    # add two channels
    # f0=T.tanh(f1);

    # if use_dropout:   f0 = dropout_layer(f0, use_noise, trng);
    lin=ff_build(tparams,f0,prefix="recog",name='output',active="linear"); # n_steps*n_samples*actionNum
    probs=T.nnet.softmax(lin.reshape([-1,actionNum]));
    probs=probs.reshape([n_steps,n_samples,actionNum]);

    """compute cost"""
    cost=0;
    # cross entropy
    entropy_cost=-y*T.log(probs);
    entropy_cost = (entropy_cost.sum(2)*mask).mean(0).sum()*100;

    cost+=entropy_cost;
    # weight decay
    weight_decay = 0.;
    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c), name='decay_c');
        for kk, vv in tparams.iteritems():
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

