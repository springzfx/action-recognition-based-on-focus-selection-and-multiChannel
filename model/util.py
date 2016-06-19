import numpy as np
from collections import OrderedDict
import cv2

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        if type(vv) is list:
            new_params[kk]=[];
            for i in range(len(vv)):
                new_params[kk].append(vv[i].get_value());
        else:
            new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
  params_list=[];
  for kk, vv in tparams.iteritems():
    if type(vv) is list:
      params_list.extend(vv);
    else:
      params_list.append(vv);
  return params_list



def load_params(path, tparams,strict=True):
    pp = np.load(path)
    for kk, vv in tparams.iteritems():
        if (kk not in pp):
            if (strict):
                raise Warning('%s is not in the archive'%kk);
            else:
                print '%s is not in the archive'%kk;
        else:
            if type(vv) is list:
              for i in range(len(vv)):
                  tparams[kk][i].set_value(pp[kk][i])
            else:
                tparams[kk].set_value(pp[kk]);
    return tparams


    

def ParamsFilter(params,prefix=None):
  tparams = OrderedDict()
  for kk, pp in params.iteritems():
    if kk.startswith(prefix):
      tparams[kk]=pp;
      print kk
  return tparams;


def share_params(params):
	import theano
	tparams = OrderedDict()
	for kk, pp in params.iteritems():
	    tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams;



def GetMeanImage():
  	"""Return mean image H*W*K
  	"""
	mean_path = '/home/fancy/program/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy';
	IMAGE_MEAN=np.load(mean_path); 
	IMAGE_MEAN=np.transpose(IMAGE_MEAN,(1,2,0)); # K*H*W to H*W*K, in order to be resized by cv2
	IMAGE_MEAN=cv2.resize(IMAGE_MEAN,(224,224));
	return IMAGE_MEAN;





def _p(pp, name):
    return '%s_%s'%(pp, name)

def GetPrefix(pp,name):
  if (pp==None or len(pp)==0): 
    return name;
  if name==None:
    return pp;
  return pp+'/'+name;


def lasagne_net_init(params,net,outputName,prefix=None,name='cnn'):
  import lasagne
  prefix=GetPrefix(prefix,name);
  params[_p(prefix,'cnn')]=lasagne.layers.get_all_params(net[outputName])
  print len(params[_p(prefix,'cnn')]);
  return params


def lasagne_net_block(net,outputName):
  import lasagne
  import theano
  x=lasagne.layers.get_output(net['input'])
  feature=lasagne.layers.get_output(net[outputName])
  op=theano.OpFromGraph([x],[feature])
  return op
