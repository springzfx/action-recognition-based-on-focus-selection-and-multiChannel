import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import adam,momentum
from ..util import itemlist,unzip,GetMeanImage
import time
import os

mean_image=GetMeanImage();


def Predict(options,f,dh,verbose=False,result=False,train_num=None):
	"""
	Args:
	f: a theano function
	dh: datahandler
	result=True: only return predict result;
	"""
	fsize=options['videosize'];
	dh.Reset();
	res=np.array([]);
	preds_res=np.array([]);
	Tcost=0.;
	start=time.time();
	if train_num==None:
		trainNum=dh.batch_num;
	else:
		trainNum=train_num;
	for vidx in xrange(trainNum):
		x,mask,y=dh.GetSingleVideoFromSource(size=options['videosize'],scale=1);
		x=x.reshape([x.shape[0],x.shape[1],fsize,fsize,3]);
		x=x.transpose([0,1,4,2,3]);
		x=x.reshape([x.shape[0],x.shape[1],-1]);
		cost,preds=f(x,mask,y);
		#acc=((y.mean(0)).argmax(1)==preds).mean();
		#print cost,acc;
		Tcost+=cost;
		if (verbose and (vidx+1)%100==0):
			print "%d/%d, %.3f, use %.1fmin"%(vidx+1,dh.batch_num,res.mean(),(time.time()-start)/60.0);
			start=time.time();
		res=np.append(res, ((y.mean(0)).argmax(1)==preds));
		preds_res=np.append(preds_res, preds);
	   
	if result==True:
		return preds_res;
	return Tcost/trainNum,res.mean();

		

def Train(options,init_params,build_model,DataHandler):
	load=options['load'];
	loadHis=options['loadHis'];
	saveto=options['saveto'];
	loadfrom=options['loadfrom'];
	dataset=options['dataset'];
	last_n=options['last_n'];
	fsize=options['videosize'];


	print ">>>init params & build graph";
	tparams=init_params(options);
	cost,preds,inner_state,inps,use_noise=build_model(options,tparams);
	print "build done"

	print ">>>compile cost&updates function";
	start=time.time();
	f=theano.function(inps,[cost,preds],allow_input_downcast=True,on_unused_input='ignore');

	print "cost function ready"
	if options['finetune']:
		updates=momentum(cost, itemlist(tparams), options['lrate'], momentum=options['momentum']);
	else:
		updates=adam(cost, itemlist(tparams), learning_rate=options['lrate'], beta1=0.9, beta2=0.999, epsilon=1e-08); 
	print len(itemlist(tparams))
	print "updates ready",len(updates)
	f_update=theano.function(inps,[cost,preds],updates=updates,allow_input_downcast=True,on_unused_input='ignore');
	print "update function ready"
	print "compile finish, use %.1fmin"%((time.time()-start)/60);

	print '>>>Optimization'
	# ready dataset
	dh_train = DataHandler(options['dataset'],datatype=0,fps=options['fps']); dh_train.SetMode('source');
	dh_valid = DataHandler(options['dataset'],datatype=1,fps=options['fps']); dh_valid.SetMode('source');
	
	train_log=np.empty((0,4),dtype='float32');
	min_valid_cost=1e8;
	max_valid_acc=0;
	if loadHis and os.path.exists(loadfrom):
		print "load log history from",loadfrom
		train_log = np.load(loadfrom)['train_log'];
		min_valid_cost=train_log[:,2].min();
	 	max_valid_acc=train_log[:,3].max();

	train_num=dh_train.batch_num;  # should be set to dh_train.batch_num
	for epochidx in xrange(options['max_epochs']):
		use_noise.set_value(1.0);
		dh_train.Reset();
		print 'Epoch ', epochidx
		start=time.time();
		for vidx in xrange(train_num):
			x,mask,y=dh_train.GetSingleVideoFromSource(size=fsize,scale=1);
			x=x.reshape([x.shape[0],x.shape[1],fsize,fsize,3]);
			x=x.transpose([0,1,4,2,3]);
			x=x.reshape([x.shape[0],x.shape[1],-1]);
			cost,preds=f_update(x,mask,y);

			acc=((y.mean(0)).argmax(1)==preds).mean();
			print cost,acc;
			# print tparams['recog/cnn_conv2_w'].get_value().sum(),tparams['recog/cnn_conv3_w'].get_value().sum(),tparams['recog/cnn_conv4_w'].get_value().sum(),tparams['recog/cnn_conv5_w'].get_value().sum(),(tparams['recog/cnn_conv5_w'].get_value()**2).sum()
			if ((vidx+1)%100==0):
				print "%d/%d, use %.1fmin"%(vidx+1,dh_train.batch_num,(time.time()-start)/60.0);
				start=time.time();


		use_noise.set_value(0.0);
		#compute train error
		dh_train.Reset(); 
		print ">>train cost";
		tcost,tacc=Predict(options,f,dh_train,verbose=True,train_num=200);
		print "cost: %.3f, acc: %.3f"%(tcost,tacc);

		#compute valid error
		dh_valid.Reset();
		print ">>valid cost";
		vcost,vacc=Predict(options,f,dh_valid,verbose=True);
		print "cost: %.3f, acc: %.3f"%(vcost,vacc);

		print ">>save point:",options['saveto'];
		train_log=np.append(train_log,np.array([tcost,tacc,vcost,vacc])[None,...],axis=0);
		# train_log.append([tcost,tacc,vcost,vacc]);
		params = unzip(tparams);
		np.savez(saveto, train_log=train_log, options=options, **params);

		if (vcost<min_valid_cost):
			min_valid_cost=vcost;
			max_valid_acc=max(max_valid_acc,vacc);
			print ">>save best:",options['bestsaveto'];
			np.savez(options['bestsaveto'], train_log=train_log, options=options, **params);
		elif (vacc>max_valid_acc):
			max_valid_acc=vacc;
			min_valid_cost=min(min_valid_cost,vcost);
			print ">>save best:",options['bestsaveto'];
			np.savez(options['bestsaveto'], train_log=train_log, options=options, **params);
		# else:
		# 	break;
		


def Test(options,init_params,build_model,DataHandler):
	dataset=options['dataset'];
	resultSave=options['result'];
	tparams=init_params(options);
	cost,preds,inner_state,inps,use_noise=build_model(options,tparams);

	print ">>>compile cost function";
	f=theano.function(inps,[cost,preds],allow_input_downcast=True,on_unused_input='ignore');
	print "compile finish";

	dh_test = DataHandler(options['dataset'],datatype=2,fps=options['fps']); dh_test.SetMode('single');
	use_noise.set_value(0.0);
	result=Predict(options,f,dh_test,verbose=True,result=True);
	#np.savez("result.npz", result=result);

	with open(resultSave,'w') as f:
		for i in range(len(result)):
			f.write("%d\n"%result[i]);
		f.close();
	
	
	

def Visual():
	dataset=options['dataset'];
	
	tparams=init_params(options);
	cost,preds,inner_state,inps,use_noise=build_model(options,tparams);
	
	print ">>>compile cost function";
	h,c,alpha=inner_state;
	f=theano.function(inps,alpha,allow_input_downcast=True,on_unused_input='ignore');
	print "compile finish";	

	dh= DataHandler(options['dataset'],datatype=2,fps=options['fps']); 
	use_noise.set_value(0.0);
		

	index=100;  # control which video
	print ">>>compute alpha";
	dh.SetMode('single'); dh.Reset();
	x,mask,y=dh.GetSingleVideoFeature(index=index);
	# switch the last two  feature dim
	x=x.reshape([x.shape[0],x.shape[1],options['featureMaps'],options['locations']]); #1024*49
	x=x.transpose([0,1,3,2]);
	x=x.reshape([x.shape[0],x.shape[1],-1]);
	alpha=f(x,mask,y);

	print ">>>visual"
	import skimage
	import skimage.transform
	def overlay(alpha,frame):
	    """Overlay alpha over the video frame
	    alpha: 49
	    frame: 224*224*3
	    """
	    a=skimage.transform.pyramid_expand(alpha.reshape([7,7]), upscale=32,sigma=20,mode='nearest');
	    a=a/np.max(a);
	    a=a[:,:,None]
	    bg=np.zeros_like(frame);
	    #a+=0.1;  # make it clear
	    a+=0.05;
	    a[a>1]=1;
	    im=a*frame+(1-a)*bg;
	    return im

	dh.SetMode('source'); dh.Reset();
	frames,_,_=dh.GetSingleVideoFromSource(index=index,scale=255);

	import cv2
	print alpha.shape,frames.shape
	for i in range(alpha.shape[0]):
		im=overlay(alpha[i,0,...],frames[i,0,...]);
		im=im.astype('uint8');
		cv2.imshow('alpha_frame',im);
		k = cv2.waitKey(0) & 0xff;
		if k == 27:  # esc
			break
		elif k==ord('s'):
			cv2.imwrite('%d.jpg'%(i),im);