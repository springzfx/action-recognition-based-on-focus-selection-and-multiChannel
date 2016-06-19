import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear

import theano



def CNN_model():
    net = {}
    net['input'] = InputLayer((None, 3, 64, 64)) 
    net['conv1/3x3_s1'] = ConvLayer(net['input'], 16, 3, stride=2, pad='same', W=lasagne.init.HeNormal(gain='relu'))   #  16*64*64
    # net['pool1/3x3_s2'] = PoolLayer(net['conv1/3x3_s1'], pool_size=3, stride=2, ignore_border=False) #  16*32*32
    net['pool1/norm1'] = LRNLayer(net['conv1/3x3_s2'], alpha=0.00002, k=1)

    net['conv2/3x3_s1'] = ConvLayer(net['pool1/norm1'], 32, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu')) # 32*32*32
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/3x3_s1'], pool_size=3, stride=2, ignore_border=False)     # 32*16*16
    net['pool2/norm1'] = LRNLayer(net['pool2/3x3_s2'], alpha=0.00002, k=1)


    net['conv3/3x3_s1'] = ConvLayer(net['pool2/norm1'], 64, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu')) # 64*16*16
    net['pool3/3x3_s2'] = PoolLayer(net['conv3/3x3_s1'], pool_size=3, stride=2, ignore_border=False)     # 64*8*8
    net['pool3/norm1'] = LRNLayer(net['pool3/3x3_s2'], alpha=0.00002, k=1)

    net['conv4/3x3_s1'] = ConvLayer(net['pool3/norm1'], 128, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu')) # 128*8*8
    net['pool4/3x3_s2'] = PoolLayer(net['conv4/3x3_s1'], pool_size=3, stride=2, ignore_border=False)      # 128*4*4
    net['pool4/norm1'] = LRNLayer(net['pool4/3x3_s2'], alpha=0.00002, k=1)

    net['conv5/3x3_s1'] = ConvLayer(net['pool4/norm1'], 128, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu'))
    net['pool5/norm1'] = LRNLayer(net['conv5/3x3_s1'], alpha=0.00002, k=1)    

    net['conv6/3x3_s1'] = ConvLayer(net['pool5/norm1'], 128, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu'))
    net['pool6/norm1'] = LRNLayer(net['conv6/3x3_s1'], alpha=0.00002, k=1)    

    net['conv7/3x3_s1'] = ConvLayer(net['pool6/norm1'], 128, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu'))
    net['pool7/norm1'] = LRNLayer(net['conv7/3x3_s1'], alpha=0.00002, k=1)    


    net['conv8/3x3_s1'] = ConvLayer(net['pool7/norm1'], 256, 3, stride=1, pad='same', W=lasagne.init.HeNormal(gain='relu'))  # 256*4*4
    net['pool8/4x4_s1'] = GlobalPoolLayer(net['conv8/3x3_s1']); # 256

    return net











def _p(pp, name):
    return '%s_%s'%(pp, name)

def GetPrefix(pp,name):
  if (pp==None or len(pp)==0): 
    return name;
  if name==None:
    return pp;
  return pp+'/'+name;


def cnn_init(params,net,prefix=None,name='cnn'):
  prefix=GetPrefix(prefix,name);
  params[_p(prefix,'cnn')]=lasagne.layers.get_all_params(net['conv8/3x3_s1'])
  return params


def bulid_cnnBlock(net):
  x=lasagne.layers.get_output(net['input'])
  feature=lasagne.layers.get_output(net['conv8/3x3_s1'])
  op=theano.OpFromGraph([x],[feature])

  return op


if __name__ == '__main__':
  import numpy as np
  import theano
  import theano.tensor as T

  net=CNN_model();
  cnn_params=lasagne.layers.get_all_params(net['conv5/3x3_s1'])
  print cnn_params[0].var

  # cnn_block=bulid_cnnBlock(net)
  # x=T.ftensor4('x');
  # feature=cnn_block(x);
  # f=theano.function([x],feature,allow_input_downcast=True)

  # x=np.ones([1,3,64,64])
  # y=f(x);
  # print y.shape