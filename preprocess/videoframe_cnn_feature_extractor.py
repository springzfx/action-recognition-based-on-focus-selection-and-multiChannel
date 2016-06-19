# -*- coding: utf-8 -*-

import numpy as np
import caffe
import cv2
import tempfile
import rarfile
import os
import time

"""
Load bvlc_googlenet model in Caffe
For each video, Extract a cnn feature, and save it to .npz file

Attention: numpy order is opposite to matlab matrix
opencv use BGR color space

my envirnment:
	python 2.7.11: Anaconda 4.0.5
	caffe
	cuda 7.5
	opencv3.1 with cuda,ffmpeg,gstreamer,ptyhon module (I compiled it by myself)
	ubuntu 14.04

"""


"""Model file"""
caffe_root='/home/fancy/program/caffe-master/';
# Model prototxt file
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
# Model caffemodel file
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# Name of the layer we want to extract 
#layer_name = 'inception_5b/output'  #1024*7*7
layer_name = 'pool5/7x7_s1'  #1024*7*7



OUTPUT_DIR="/home/fancy/process/feature/"

"""Caffe Net input define"""
BATCH_SIZE=20;  # input batch size
FRAME_WIDTH=224;  # input size
FRAME_HEIGHT=224;
CHANNELS=3; 


"""Mean image process
resize to  FRAME_HEIGHT x FRAME_WIDTH
tranpose to Height x Width x Channel
in this way, each image only needs to resize once
"""
IMAGE_MEAN=np.load(mean_path); # K*H*W to H*W*K
IMAGE_MEAN=np.transpose(IMAGE_MEAN,(1,2,0));
IMAGE_MEAN=cv2.resize(IMAGE_MEAN,(FRAME_HEIGHT,FRAME_WIDTH));



class VideoExtract(object):
	"""Retrive frames from video and preprocess it to match CNN input
	"""
	def __init__(self, videoFile):
		self.video=cv2.VideoCapture(videoFile);

	def Extract(self):
		"""
		every time you excute this function, it extract at most BATCH_SIZE frames
		you need to excute muti-times until it returns None

		Returns:
			framesExtracted: Num x Channel x Height x Width, BGR,[0-255],mean
		"""
		if (not self.video.isOpened()):return None;

		framesExtracted=np.empty((0,CHANNELS,FRAME_WIDTH,FRAME_HEIGHT));
		framenum=self.GetFrameNum();
		#print framenum
		i=0;
		while (i<framenum and i<BATCH_SIZE):
			ok,frame=self.video.read();
			if (not ok):break;

			frame_std=self.Preprocess(frame);
			framesExtracted=np.append(framesExtracted,np.expand_dims(frame_std, axis=0),axis=0);
			i+=1;

		if (framesExtracted.shape[0]==0):
			return None;
		return framesExtracted;

	def Preprocess(self,im):
		"""Resize,Mean image,Tranpose,RGB->BGR
		"""
		im=cv2.resize(im,(FRAME_HEIGHT,FRAME_WIDTH));
		im=im-IMAGE_MEAN;  # mean 
		im=np.transpose(im,(2,0,1)); # to k*H*W, no need to BGR, because it use BGR by default
		return im
	

	def GetFrameNum(self):
		return self.video.get(cv2.CAP_PROP_FRAME_COUNT);




#refer  to http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
class FeatureExtract_Caffe(object):
	"""Init a deep net in caffe,Forward the net, and extract feature
	"""
	def __init__(self):
		# load model
		self.net = caffe.Net(model_prototxt,model_trained,caffe.TEST);
		self.net.blobs['data'].reshape(BATCH_SIZE,CHANNELS,FRAME_HEIGHT,FRAME_WIDTH);

		# load input and configure preprocessing
		#self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		#self.transformer.set_mean('data', np.load(mean_path).mean(1).mean(1))
		#self.transformer.set_transpose('data', (2,0,1))  # W*H*K -> K*H*W,  K is RGB channel
		#self.transformer.set_channel_swap('data', (2,1,0)) # the model has channels in BGR order instead of RGB
		#self.transformer.set_raw_scale('data', 255.0) # scale [0,1] -> [0,255]


	def FeatureExtract(self,batchdata):
		"""Forward batchdata into the net,and return feature extracted
		Args:
			batchdata: N*K*H*W,K is the channel,N is number
		"""
		# ready data
		batchsize=batchdata.shape[0];
		if (batchsize<BATCH_SIZE):
			self.net.blobs['data'].reshape(batchsize,CHANNELS,FRAME_HEIGHT,FRAME_WIDTH);
		self.net.blobs['data'].data[...] = 	batchdata;

		# forward 
		self.net.forward();

		# init back
		if (batchsize<BATCH_SIZE):
			self.net.blobs['data'].reshape(BATCH_SIZE,CHANNELS,FRAME_HEIGHT,FRAME_WIDTH);

		# conv feature	N*1024*7*7
		return self.net.blobs[layer_name].data
		

def ExtractAndSave(videofilepath,outputfilepath):
	"""Extract a video feature through CNN
	videofilepath: video path
	outputfilepath: output path,not directory,extension will be override
	"""
	VE=VideoExtract(videofilepath);
	featuresExtracted=np.empty((0,1024,7,7));
	data=VE.Extract();
	while (data is not None):
		feat=Conv.FeatureExtract(data);
		featuresExtracted=np.append(featuresExtracted,feat,axis=0);
		data=VE.Extract();
	
	np.savez_compressed(os.path.splitext(outputfilepath)[0]+".npz", featuresExtracted);
	
	


if __name__=="__main__":
	caffe.set_device(0);
	caffe.set_mode_gpu();

	# load CNN model and param
	Conv=FeatureExtract_Caffe();
	
	# get video from rar
	rarFileName="/home/fancy/Desktop/THUMOS14/UCF11_updated_mpg.rar";
	rarF=rarfile.RarFile(rarFileName);
	tmppath=tempfile.mkdtemp()+"/";
	
	i=0;start=time.time();
	for rFilename in rarF.namelist():
		rFilename=rFilename.replace('\\','/'); # in case slash
		# check it is video file or not
		ext=os.path.splitext(rFilename)[1];
		if not (ext==".mpg" or ext==".avi"):
			continue;

		
		print "process %d: %s"% (i,rFilename);
		i+=1; v_start=time.time();

		# ready path
		inputpath=tmppath+rFilename;
		outputpath=OUTPUT_DIR+os.path.splitext(rFilename)[0]+".npz";
		outputdir=os.path.split(outputpath)[0];

		# if feature file alreay exist,then skip 
		if os.path.isfile(outputpath):
			print "feature alreay exist,skip...";
			continue;

		# check output dir
		if not os.path.isdir(outputdir):
			os.makedirs(outputdir);	

		# core part
		rarF.extract(rFilename, tmppath); # unrar to tmp dir
		ExtractAndSave(inputpath,outputpath);
		os.remove(inputpath);  # remove tmp video file

		print "time used: %.0fs " % (time.time()-v_start);
		
	print "total time used: %.0fmin "% ((time.time()-start)/60);
