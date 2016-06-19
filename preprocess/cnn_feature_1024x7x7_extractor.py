"""
python -m  preprocess.cnn_feature_1024x7x7_extractor

use google_net extract feature, And save to h5 file
"""

from model.google_net.googlenet import build_model as build_googlenet_model
import numpy as np

import theano
import theano.tensor as T
import pickle
import lasagne

import cv2
import h5py
import os

VIDEO_DIR='/home/fancy/Desktop/HMDB51/dataset/data/';
videofiles="/home/fancy/Desktop/HMDB51/split/test_filename.txt";
#TotalFrameNum=97412; #   UCF11 train:150162,valid:58485,test:97412
TotalFrameNum=143467; #   HMDB51 train:236864,valid:108066,test:143467

OUTPUT_DIR="/home/fancy/Desktop/HMDB51/feature_1024x7x7/";
h5name="test_feature.h5";


"""Net input define"""
BATCH_SIZE=100;  # input batch size
FRAME_WIDTH=224;  # input size
FRAME_HEIGHT=224;
CHANNELS=3; 


"""Mean image process
resize to  FRAME_HEIGHT x FRAME_WIDTH
tranpose to Height x Width x Channel
in this way, each image only needs to resize once
"""
mean_path = '/home/fancy/program/caffe-master/'+'python/caffe/imagenet/ilsvrc_2012_mean.npy';
IMAGE_MEAN=np.load(mean_path); # K*H*W to H*W*K
IMAGE_MEAN=np.transpose(IMAGE_MEAN,(1,2,0));
IMAGE_MEAN=cv2.resize(IMAGE_MEAN,(FRAME_HEIGHT,FRAME_WIDTH));
print "image mean process finish"


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
			ok,frame=self.video.read(); #Height x Width x Channel
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



#ready google net
googlenet=build_googlenet_model();
model = pickle.load(open('/home/fancy/Desktop/Attention LSTM/google_net/blvc_googlenet.pkl')); # load pretrained parameters
lasagne.layers.set_all_param_values(googlenet['prob'], model['param values']);
x=lasagne.layers.get_output(googlenet['input']);
feature=lasagne.layers.get_output(googlenet['inception_5b/output']);  #1024*7*7
extractor=theano.function([x],feature);
print "function compile finish"



# read h5py
h=h5py.File(OUTPUT_DIR+h5name,'w');
h.create_dataset("feature",(TotalFrameNum,1024*7*7),compression="gzip",chunks=True);
print "h5 ready",h['feature'].shape

i=0;k=0;
framenum=[];
with open(videofiles,'r') as f:
	while (True):
		line=f.readline();
		if (line=="" or line==None):
                        break;
		videoname=line.strip();

		print videoname;
		videopath=VIDEO_DIR+videoname;
		VE=VideoExtract(videopath);
		data=VE.Extract();
		while (data is not None): 
			feat=extractor(data.astype('float32'));
			k=feat.shape[0];
			framenum.append(k);
			h['feature'][i:i+k]=feat.reshape([k,-1]);
			i+=k;
			data=VE.Extract();
	f.close();
h['framenum']=framenum;
h.close();

