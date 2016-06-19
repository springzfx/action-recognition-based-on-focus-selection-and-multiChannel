"""
python -m preprocess.cnn_feature_extractor

use google_net extract feature, And save to h5 file
"""

from model.layer import GoogleNet_Layer
import numpy as np
import theano
import theano.tensor as T
import cv2
import h5py
import os

VIDEO_DIR='/home/fancy/Desktop/UCF11/data/';
videofiles="/home/fancy/Desktop/UCF11/split/test_filename.txt";
TotalFrameNum=97412; #  train:150162,valid:58485,test:97412

OUTPUT_DIR="/home/fancy/Desktop/UCF11/feature/";
h5name="test_feature.h5";


"""Net input define"""
BATCH_SIZE=20;  # input batch size
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
x=T.ftensor4('x');
feature=GoogleNet_Layer(x);
extractor=theano.function([x],feature);
print "function compile finish"

# read h5py
h=h5py.File(OUTPUT_DIR+h5name,'w');
h.create_dataset("features",(TotalFrameNum,1024),compression="gzip",chunks=True);
print "h5 ready",h['features'].shape

i=0;k=0;
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
			h['features'][i:i+k]=feat;
			i+=k;
			data=VE.Extract();
	f.close();

h.close();

