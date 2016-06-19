import numpy as np
import theano
import theano.tensor as T
import lasagne
theano.config.floatX = 'float32'

def _p(pp, name):
    return '%s_%s'%(pp, name)

def tanh(x):
    return T.tanh(x)

def sigmoid(x):
	return T.nnet.sigmoid(x)
	
def softmax(x):
	return T.nnet.softmax(x)
	

def rectifier(x):
    return T.maximum(0., x)

def linear(x):
    return x

def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=1, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin

    # W = np.abs(np.random.randn(nin, nout))/np.sqrt(2/np.pi)/nout;
    W = scale * np.random.randn(nin, nout)/np.sqrt(nin);   
    
    return W.astype('float32')


def _norm_weight(nin,nout,cores,scale=1):
    """
    Random weights drawn from a Gaussian
    """
    W = scale * np.random.randn(nin,nout,cores)/np.sqrt(nin);   
    
    return W.astype('float32')




def standardize(x,axis=-1):
	return (x-x.mean(axis,keepdims=True))/T.std(x,axis=axis,keepdims=True);
	# return  x;



def GetPrefix(pp,name):
	if (pp==None or len(pp)==0): 
		return name;
	if name==None:
		return pp;
	return pp+'/'+name;




def ff_init(params,nin,nout,prefix=None,name='ff'):
	prefix=GetPrefix(prefix,name);
	params[_p(prefix,'w')]=norm_weight(nin,nout,scale=0.01);
	params[_p(prefix,'b')]=np.zeros(nout).astype('float32');
	return params;

def ff_build(tparams,state_below,active="tanh",prefix=None,name='ff',std=True):
	prefix=GetPrefix(prefix,name);
	if std:	state_below=standardize(state_below);
	return eval(active)( T.dot( state_below,tparams[_p(prefix,'w')] )+tparams[_p(prefix,'b')] );


def dropout_layer(state_before, use_noise, trng,p=0.5):
	p=1-p;
	proj = T.switch(use_noise,
	                 state_before*trng.binomial(state_before.shape, p=p, n=1, dtype=state_before.dtype)/p,
	                 state_before);
	return proj

def RNN_init(params,xdim,hdim,prefix=None,name='rnn'):
	prefix=GetPrefix(prefix,name);
	params[_p(prefix,'wh')]=norm_weight(hdim,hdim);
	params[_p(prefix,'wx')]=norm_weight(xdim,hdim);
	params[_p(prefix,'b')]=np.zeros(hdim).astype('float32');
	return params;

def RNN_layer(tparams,inputs,mask=None,init_h=None,prefix=None,name='rnn',std=True):
	"""
	inputs: n_steps*n_samples*x_size
	return h
	"""
	prefix=GetPrefix(prefix,name);
	# if length!=None: inputs=inputs[index:index+length,:,:];
	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	x_size=inputs.shape[2];

	hdim=tparams[_p(prefix,'wh')].shape[0];

	if mask == None:		
		mask = T.alloc(1., n_steps, n_samples);
	if init_h == None:       
		init_h = T.alloc(0., n_samples, hdim);


	def _step(m,x,h):
		inputs_h=( T.dot(x,tparams[_p(prefix,'wx')])+T.dot(h,tparams[_p(prefix,'wh')]) )/2+tparams[_p(prefix,'b')];
		_h=tanh(inputs_h);
		return _h;

	if std:	inputs=standardize(inputs);
	out,updates=theano.scan(lambda m,x,h:_step(m,x,h), 
							sequences=[mask,inputs], 
							outputs_info=[init_h],
							name=_p(prefix,'scan'),
							n_steps=n_steps,
							# truncate_gradient=10,
							profile=False);
	return out


def R2_RNN_block(tparams,inputs,prefix=None,name='r2_rnn',std=True):
	prefix=GetPrefix(prefix,name);
	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	x_size=inputs.shape[2];	

	r_steps=T.ceil(T.log2(n_steps)).astype('uint32');
	r_steps=T.arange(r_steps);
	# r_steps=r_steps.reshape([r_steps.shape[0],1]);

	
	def _step_inner(index,num,inps):
		index=index*2;
		index_=T.minimum(index+2,num);

		h=RNN_layer(tparams,inps[index:index_,:,:],prefix=prefix,name=None,std=False);
		return h[-1,:,:];
	
	def _step(r_step,num,inps,std=True):
		n=num;
		steps=T.arange((n+1)/2);
		# steps=steps.reshape([steps.shape[0],1]);

		out,updates=theano.scan(lambda index,num,inps:_step_inner(index,num,inps), 
							sequences=[steps], 
							outputs_info=None,
							non_sequences=[num,inps],
							name=_p(prefix,'inner_scan'),
							n_steps=steps.shape[0],
							profile=False);

		# if std:	out=standardize(out);
		num=out.shape[0];
		h=T.zeros_like(inps);
		h=T.set_subtensor(h[:num],out);
		return num,h;
		# return out;
	
	if std:	inputs=standardize(inputs);
	out,updates=theano.reduce(lambda r_step,num,inps:_step(r_step,num,inps), 
							sequences=r_steps, 
							outputs_info=[inputs.shape[0],inputs],
							# non_sequences=inputs,
							name=_p(prefix,'scan')
							);
	return out[1][:out[0]];






def LSTM_init(params,xdim,hdim,prefix=None,name='lstm'):
	prefix=GetPrefix(prefix,name);
	params[_p(prefix,'wx')]=norm_weight(xdim,4*hdim);
	# lstm to lstm
	params[_p(prefix,'wh')]=norm_weight(hdim,4*hdim);
	# lstm bias
	params[_p(prefix,'b')]=np.zeros(4*hdim).astype('float32');
	return params;

def LSTM_layer(tparams,inputs,mask=None,init_c=None,init_h=None,prefix=None,name='lstm',std=True):
	"""
	inputs: n_steps*n_samples*x_size
	return: [h],[c]
	"""
	prefix=GetPrefix(prefix,name);

	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	# x_size=inputs.shape[2];

	hdim=tparams[_p(prefix,'wh')].shape[0];

	if mask == None:		
		mask = T.alloc(1., n_steps, n_samples);
	if init_c == None:      
		init_c = T.alloc(0., n_samples, hdim);
	if init_h == None:       
		init_h = T.alloc(0., n_samples, hdim);

	# wx=theano.gradient.grad_clip(tparams[_p(prefix,'wx')],-0.1,0.1);
	# wh=theano.gradient.grad_clip(tparams[_p(prefix,'wh')],-0.1,0.1);

	wx=tparams[_p(prefix,'wx')];
	wh=tparams[_p(prefix,'wh')];

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n*dim:(n+1)*dim]
		return _x[:, n*dim:(n+1)*dim]

	def _step(m,x,h,c):
		# pass lstm
		input_lstm=T.dot(x,wx)+T.dot(h,wh)+tparams[_p(prefix,'b')];
		i=sigmoid(_slice(input_lstm,0,hdim)); 
		f=sigmoid(_slice(input_lstm,1,hdim)); 
		o=sigmoid(_slice(input_lstm,2,hdim)); 
		i_c= tanh(_slice(input_lstm,3,hdim));

		# update c and h,use mask
		_c=i*i_c+f*c;
		_c= m[:,None]*_c + (1. - m)[:,None]*c;
		_h=o*tanh(_c);
		_h= m[:,None]*_h + (1. - m)[:,None]*h;
		return [_h,_c];

	if std:	inputs=standardize(inputs);
	out,updates=theano.scan(lambda m,x,h,c:_step(m,x,h,c), 
							sequences=[mask,inputs], 
							outputs_info=[init_h, init_c],
							name=_p(prefix,'layer'),
							n_steps=n_steps,
							profile=False);
	return out

def LSTM_mcore_init(params,xdim,hdim,cores=4,prefix=None,name='lstm'):
	prefix=GetPrefix(prefix,name);
	
	params[_p(prefix,'wx')]=norm_weight(xdim,4*hdim*cores);
	# lstm to lstm
	params[_p(prefix,'wh')]=norm_weight(hdim,4*hdim*cores);
	
	# lstm bias
	params[_p(prefix,'b')]=np.zeros([4*hdim*cores]).astype('float32');

	params=ff_init(params,cores,1,prefix=prefix,name='merge');

	return params;

def LSTM_mcore_layer(tparams,inputs,cores=4,init_c=None,init_h=None,prefix=None,name='lstm',std=True):
	"""
	inputs: n_steps*n_samples*x_size
	return: [h],[c]
	"""
	prefix=GetPrefix(prefix,name);

	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	# x_size=inputs.shape[2];

	wx=tparams[_p(prefix,'wx')];
	wh=tparams[_p(prefix,'wh')];
	hdim=wh.shape[0];

	if init_c == None:      
		init_c = T.alloc(0., n_samples, hdim, cores);
	if init_h == None:       
		init_h = T.alloc(0., n_samples, hdim);

	# wx=theano.gradient.grad_clip(tparams[_p(prefix,'wx')],-0.1,0.1);
	# wh=theano.gradient.grad_clip(tparams[_p(prefix,'wh')],-0.1,0.1);

	# wx=tparams[_p(prefix,'wx')];
	# wh=tparams[_p(prefix,'wh')];

	def _slice(_x, n, dim):
		# _x=theano.printing.Print(attrs=("shape",))(_x);
		return _x[:, n*dim:(n+1)*dim];

	def _step(x,h,c):
		
		# wx=theano.printing.Print('wx',attrs=("shape",))(wx);
		# wh=theano.printing.Print('wh',attrs=("shape",))(wh);

		input_lstm=T.dot(x,wx)+T.dot(h,wh)+tparams[_p(prefix,'b')];

		
		input_lstm=input_lstm.reshape([n_samples,4*hdim,cores]);

		i=sigmoid(_slice(input_lstm,0,hdim)); 
		f=sigmoid(_slice(input_lstm,1,hdim)); 
		o=sigmoid(_slice(input_lstm,2,hdim)); 
		i_c= tanh(_slice(input_lstm,3,hdim));

		# update c and h,use mask
		_c=i*i_c+f*c;
		_h=o*tanh(_c);
		
		_h=ff_build(tparams,_h,prefix=prefix,name='merge',active='tanh');
		# _h=_h.mean(2);
		_h=_h.reshape([_h.shape[0],_h.shape[1]]);

		return _h,_c;

	if std:	inputs=standardize(inputs);
	# inputs=theano.printing.Print(attrs=("shape",))(inputs);

	out,updates=theano.scan(lambda x,h,c:_step(x,h,c), 
							sequences=[inputs], 
							outputs_info=[init_h, init_c],
							name=_p(prefix,'layer'),
							n_steps=n_steps,
							profile=False);
	return out


# def Linger_init(params,indim,outdim,prefix=None,name='linger'):
# 	prefix=GetPrefix(prefix,name);

# 	params[_p(prefix,'w')]=norm_weight(indim,outdim);
# 	#params[_p(prefix,'l')]=norm_weight(indim,outdim);
# 	# params[_p(prefix,'l')]=-params[_p(prefix,'w')];
# 	params[_p(prefix,'b')]=np.zeros(outdim).astype('float32');

# 	return params;

# def Linger_layer(tparams,inputs,linger_state=None,prefix=None,name='linger',std=True):
# 	prefix=GetPrefix(prefix,name);

# 	n_steps=inputs.shape[0];
# 	if linger_state==None:
# 		linger_state = T.alloc(0., inputs.shape[1], inputs.shape[2]);

# 	def _step(x,ctx):
# 		outputs=tanh(T.dot(x,tparams[_p(prefix,'w')])-T.dot(ctx,tparams[_p(prefix,'w')])+tparams[_p(prefix,'b')] );
# 		return x,outputs;

# 	if std:	inputs=standardize(inputs);
# 	out,updates=theano.scan(lambda x,ctx:_step(x,ctx), 
# 							sequences=inputs, 
# 							outputs_info=[linger_state,None],
# 							#non_sequences=init_ctx,
# 							name=_p(prefix,'layer'),
# 							n_steps=n_steps,
# 							profile=False);

# 	return out[1];





# def LSTM_Magnitude_init(params,xdim,hdim,prefix=None,name='lstm'):
# 	prefix=GetPrefix(prefix,name);

# 	# std LSTM
# 	params=LSTM_init(params,xdim,hdim,prefix=prefix,name='std_lstm')
	
# 	# input to lstm
# 	params=Linger_init(params,xdim,hdim,prefix=prefix,name='linger');

# 	# params[_p(prefix,'wx')]=norm_weight(xdim,hdim);
# 	# params[_p(prefix,'wctx')]=norm_weight(xdim,hdim);
# 	# params[_p(prefix,'bx')]=np.zeros(hdim).astype('float32');

# 	return params;

# def LSTM__Magnitude_layer(tparams,inputs,mask=None,init_ctx=None,init_c=None,init_h=None,prefix=None,name='lstm'):
# 	"""
# 	LSTM based on magnitude
# 	inputs: n_steps*n_samples*x_size
# 	return: [h],[c]
# 	"""
# 	prefix=GetPrefix(prefix,name);

# 	n_steps=inputs.shape[0];
# 	n_samples=inputs.shape[1];
# 	x_size=inputs.shape[2];

# 	hdim=tparams[_p(prefix,'wx')].shape[0];

# 	if init_ctx==None:
# 		init_ctx = T.alloc(0., n_samples, x_size);

# 	if mask == None:		
# 		mask = T.alloc(1., n_steps, n_samples);
# 	if init_c == None:      
# 		init_c = T.alloc(0., n_samples, hdim);
# 	if init_h == None:       
# 		init_h = T.alloc(0., n_samples, hdim);

# 	def _slice(_x, n, dim):
# 		if _x.ndim == 3:
# 			return _x[:, :, n*dim:(n+1)*dim]
# 		return _x[:, n*dim:(n+1)*dim]

# 	def _step(m,x,h,c,ctx):
# 		# pass lstm
# 		# input_i_c=T.dot(x,tparams[_p(prefix,'wx')])+T.dot(ctx,tparams[_p(prefix,'wctx')])+tparams[_p(prefix,'bx')];
# 		# i_c=sigmoid(input_i_c);

# 		x,_ctx=Linger_layer(tparams,x,linger_state=ctx,prefix=prefix,name='linger');

# 		input_lstm=T.dot(x,tparams[_p(prefix,'wx')])+T.dot(h,tparams[_p(prefix,'wh')])+tparams[_p(prefix,'b')];
# 		i=sigmoid(_slice(input_lstm,0,hdim));
# 		f=sigmoid(_slice(input_lstm,1,hdim));
# 		o=sigmoid(_slice(input_lstm,2,hdim));
# 		i_c= tanh(_slice(input_lstm,3,hdim));

# 		# update c and h,use mask
# 		_c=i*i_c+f*c;
# 		#_c= m[:,None]*_c + (1. - m)[:,None]*c;
# 		_h=f*tanh(_c);
# 		#_h= m[:,None]*_h + (1. - m)[:,None]*h;
# 		return [_h,_c,_ctx];

# 	out,updates=theano.scan(lambda m,x,h,c,ctx:_step(m,x,h,c,ctx), 
# 							sequences=[mask,inputs], 
# 							outputs_info=[init_h, init_c,init_ctx],
# 							#non_sequences=init_ctx,
# 							name=_p(prefix,'layer'),
# 							n_steps=n_steps,
# 							profile=False);

# 	return out[0],out[1];






def GoogleNet_Layer(inputs,prefix=None,name='googlenet'):
	"""
	inputs: n_samples*3*224*224
	notice that there is a scan in googlenet implementation
	"""
	from google_net.build_googlenet import GoogleNet_Block

	googlenet_op=GoogleNet_Block(hold=True);   # clone the googlenet graph is not a good choise
	return googlenet_op(inputs);


def WTA_Layer(inputs,pool_size,axis,ndim=1024):
	inl=lasagne.layers.InputLayer((None,None,ndim),inputs);
	feature=lasagne.layers.FeatureWTALayer(inl,pool_size,axis=axis);
	return lasagne.layers.get_output(feature);


def Saliency_init(params,fdim,prefix=None,name='saliency'):
	"""
	saliency with location feature, trained through target function,like action recognition
	example: n_samples*locations*feature
	"""
	prefix=GetPrefix(prefix,name);
	params=ff_init(params,fdim,128,prefix=prefix,name='alpha_pre');
	params=ff_init(params,128,1,prefix=prefix,name='alpha');
	return params;

def Saliency_Layer(tparams,state_below,prefix=None,name='saliency',std=True):
	"""
	saliency with location feature, trained through target function,like action recognition
	inputs: n_samples*locations*features, ex:100*49*1024
	"""
	prefix=GetPrefix(prefix,name);

	# pass alpha
	if std:	_state_below=standardize(state_below);
	alpha_pre=ff_build(tparams,_state_below,active='tanh',prefix=prefix,name='alpha_pre');
	alpha=ff_build(tparams,alpha_pre,active='linear',prefix=prefix,name='alpha');
	alpha=alpha.reshape([alpha.shape[0],alpha.shape[1]]);
	alpha=T.nnet.softmax(alpha);
	feature=(alpha[:,:,None]*state_below).sum(1); # n_samples*features

	return feature,alpha



def SaliencyLSTM_init(params,feaMaps,P=128,prefix=None,name='saliencyLSTM'):
	"""
	saliency with location feature, trained through target function,like action recognition
	fdim: feature dimension
	example: n_steps*n_samples*locations*feature
	"""
	prefix=GetPrefix(prefix,name);
	
	params=ff_init(params,feaMaps,P,prefix=prefix,name='alpha_pre');
	params=LSTM_init(params,P,P,prefix=prefix,name='alpha_lstm');
	params=ff_init(params,P,1,prefix=prefix,name='alpha');
	return params;

def SaliencyLSTM_block(tparams,inputs,mask=None,init_h=None,prefix=None,name='saliencyLSTM',std=True):
	"""
	saliency with location feature, trained through target function,like action recognition
	state_below: n_steps*n_samples*locations*feature
	return:
		feature: n_steps*n_samples*feature
		alpha: n_steps*n_samples*locations [0-1]
	"""
	prefix=GetPrefix(prefix,name);

	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	n_locations=inputs.shape[2];
	n_feaMaps=inputs.shape[3];

	if std:	_inputs=standardize(inputs);
	alpha_pre=ff_build(tparams,_inputs,active='tanh',prefix=prefix,name='alpha_pre');

	alpha_pre=alpha_pre.reshape([n_steps,n_samples*n_locations,-1]);
	h,c=LSTM_layer(tparams,alpha_pre,prefix=prefix,name='alpha_lstm'); 
	h=h.reshape([n_steps,n_samples,n_locations,-1]);
	alpha=ff_build(tparams,h,active='linear',prefix=prefix,name='alpha');
	alpha=alpha.reshape([n_steps*n_samples,n_locations]);
	alpha=T.nnet.softmax(alpha);
	alpha=alpha.reshape([n_steps,n_samples,n_locations]);
	feature=(alpha[:,:,:,None]*inputs).sum(2); 

	return feature,alpha
	


def SaliencyFgbg_init(params,lDim,feaMaps,P=128,prefix=None,name='saliencyFgbg'):
	prefix=GetPrefix(prefix,name);
	params[_p(prefix,'w')]=np.random.uniform(0,1,[lDim,feaMaps]).astype('float32');
	# params[_p(prefix,'b')]=np.zeros(lDim).astype('float32');
	params=ff_init(params,feaMaps,P,prefix=prefix,name='alpha_pre');
	params=ff_init(params,P,1,prefix=prefix,name='alpha');
	return params;



def SaliencyFgbg_block(tparams,inputs,init_h=None,prefix=None,name='saliencyFgbg'):
	prefix=GetPrefix(prefix,name);
	
	n_steps=inputs.shape[0];
	n_samples=inputs.shape[1];
	n_locations=inputs.shape[2];
	x_features=inputs.shape[3];
	
	if init_h == None:       
		init_h = T.alloc(0., n_samples, n_locations, x_features);

	# w=theano.gradient.grad_clip(tparams[_p(prefix,'w')],-0.1,0.1);
	w=tparams[_p(prefix,'w')];

	def _step(x,h):
		# x=x.transpose([0,2,1]); h=h.transpose([0,2,1]); 
		h=x*w+h*(1-w);
		# h=h.transpose([0,2,1]); 
		return h;
		# return oups/oups.max(-1,keepdims=True);

	inputs=standardize(inputs);
	out,updates=theano.scan(lambda x,h:_step(x,h), 
							sequences=[inputs], 
							outputs_info=[init_h],
							name=_p(prefix,'scan'),
							n_steps=n_steps,
							profile=False);
	bg=out;
	fg=inputs-bg;
	alpha_pre=ff_build(tparams,fg,active='tanh',prefix=prefix,name='alpha_pre');
	alpha=ff_build(tparams,alpha_pre,active='linear',prefix=prefix,name='alpha');
	alpha=alpha.reshape([n_steps*n_samples,n_locations]);
	alpha=T.nnet.softmax(alpha);
	alpha=alpha.reshape([n_steps,n_samples,n_locations]);
	feature=(alpha[:,:,:,None]*inputs).sum(2); # n_samples*features
	return feature,alpha



# def AGL_init(params,xsize,ctxdim,hdim,prefix=None,name='recog'):
# 	"""Attention+GoogleNet+LSTM
# 	xsize: list,[3,224,224]
# 	ctxdim: dim of input to LSTM, also the feature dim extracted by CNN.
# 	hdim: dim lstm cell and gates.
# 	"""
# 	prefix=GetPrefix(prefix,name);

# 	"""init alpha"""
# 	pp=prefix+'/alpha';
# 	channel_size=xsize[1]*xsize[2];
# 	params[_p(pp,'w')]=norm_weight(hdim,channel_size);
# 	params[_p(pp,'b')]=np.zeros(channel_size).astype('float32');

# 	params=LSTM_init(params,ctxdim,hdim,prefix=prefix);

# 	return params;


# def AGL_block(tparams,state_below,mean_image,mask=None,init_c=None,init_h=None,prefix=None,name='recog'):
# 	"""Attention+GoogleNet+LSTM
# 	state_below: require range(0,255),without mean 
# 	mask: n_steps*n_samples
# 	"""

# 	prefix=GetPrefix(prefix,name);

# 	n_steps=state_below.shape[0];
# 	n_samples=state_below.shape[1];
# 	channel=state_below.shape[2];
# 	height=state_below.shape[3];
# 	width=state_below.shape[4];

# 	hdim=tparams[_p(prefix+"/lstm",'wh')].shape[0];

# 	if mask == None:		
# 		mask = T.alloc(1., n_steps, n_samples);

# 	if init_c == None:      
# 		init_c = T.alloc(0., n_samples, hdim);
# 	if init_h == None:       
# 		init_h = T.alloc(0., n_samples, hdim);

# 	def _step(m,x,h,c):
# 		"""
# 		x: n_samples*3*224*224
# 		h: n_samples*hdim
# 		c: n_samples*hdim
# 		"""
# 		pp=prefix+"/alpha";
# 		# h->alpha
# 		input_alpha=T.dot(h,tparams[_p(pp,'w')])+tparams[_p(pp,'b')]; 
# 		active_alpha=1-sigmoid(input_alpha); # n_samples*(224*224)

# 		# attention
# 		alpha=active_alpha.reshape([n_samples,height,width])[:,None,:,:];
# 		att_x=(1-alpha)*255+alpha*x;

#         # ->cnn
# 		cnn_x=att_x.reshape([-1,channel,height,width]);
# 		cnn_x=cnn_x-mean_image.transpose([2,0,1]).astype('float32');  # mean imgae
# 		feature=GoogleNet_Layer(cnn_x,prefix=prefix);

# 		# pass lstm, only one step each time
# 		_h,_c=LSTM_layer(tparams,feature[None,:,:],prefix=prefix,mask=m[:,None,None],init_c=c,init_h=h);

# 		return [_h[0],_c[0],active_alpha];

# 	out,updates=theano.scan(lambda m,x,h,c:_step(m,x,h,c), 
# 							sequences=[mask,state_below], 
# 							outputs_info=[init_h, init_c, None], 
# 							name=_p(prefix,'layer'),
# 							n_steps=n_steps,
# 							profile=False);
# 	return out









# THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True,optimizer=fast_compile'

if __name__ == '__main__':
	from collections import OrderedDict;

	# initialize Theano shared variables according to the initial parameters
	def init_tparams(params):
	    tparams = OrderedDict()
	    for kk, pp in params.iteritems():
	        tparams[kk] = theano.shared(params[kk], name=kk)
	    return tparams


	print "test";
	params = OrderedDict();
	params=Attention_init(params,[3,224,224],1024,512);
	ftensor5 =T.TensorType('float32', (False,)*5);
	x = ftensor5('x');

	tparams=init_tparams(params);
	y=Attention_block(tparams,x);
	print "compile function"
	f=theano.function([x],y);  #about 30s
	print "compile finish";
	i=np.random.rand(30,50,3,224,224).astype('float32');
	h,c,alpha=f(i);
	print h.shape,c.shape,alpha.shape