from googlenet import build_model as build_googlenet_model
import pickle
import lasagne
import numpy as np
import theano
import theano.tensor as T

# ref: https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb

"""
x: range[0-255]-mean_image, samples*3*height*witdh
"""


googlenet=build_googlenet_model();
model = pickle.load(open('/home/fancy/Desktop/project/model/google_net/blvc_googlenet.pkl')); # load pretrained parameters
lasagne.layers.set_all_param_values(googlenet['prob'], model['param values']);
x=lasagne.layers.get_output(googlenet['input']);
feature=lasagne.layers.get_output(googlenet['pool5/7x7_s1']);
op=theano.OpFromGraph([x],[feature]);  #encapsulate a Theano graph

def GoogleNet_Block(hold=False):
	"""
	hold=True: Always return the same graph, otherwise not the same.
	"""
	global op;
	if (hold and (op is not None)):
		return op;

	googlenet=build_googlenet_model();
	model = pickle.load(open('/home/fancy/Desktop/Attention LSTM/google_net/blvc_googlenet.pkl')); # load pretrained parameters
	lasagne.layers.set_all_param_values(googlenet['prob'], model['param values']);
	x=lasagne.layers.get_output(googlenet['input']);
	feature=lasagne.layers.get_output(googlenet['pool5/7x7_s1']);
	op=theano.OpFromGraph(x,feature);

	return op;
