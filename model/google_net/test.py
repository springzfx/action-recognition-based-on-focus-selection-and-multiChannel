from googlenet import build_model as build_googlenet_model
import pickle
import lasagne
import numpy as np
import theano
import theano.tensor as T

# ref: https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20(VGG_S).ipynb
def CNN_Block():
	googlenet=build_googlenet_model();
	model = pickle.load(open('blvc_googlenet.pkl')); # load pretrained parameters
	lasagne.layers.set_all_param_values(googlenet['prob'], model['param values']);

	x=lasagne.layers.get_output(googlenet['input']);
	feature=lasagne.layers.get_output(googlenet['pool5/7x7_s1']);

	return x,feature




if __name__ == '__main__':
	y=T.ftensor4('y');

	x,feature=CNN_Block();
	feature1=theano.clone(feature,{x:y})

	print "compile cnn function"
	f=theano.function([x],feature);
	f1=theano.function([y],feature1);
	print "compile done"

	im=np.random.random((1,3,224,224)).astype('float32');
	prob = f(im);
	prob1 = f1(im);
	print prob[:10]
	print prob1[:10]


