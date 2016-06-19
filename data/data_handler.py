# -*- coding: utf-8 -*-
import sys
import h5py
import numpy as np
import time
import cv2


class DataHandler(object):
	def __init__(self, data_pb,featureType='1024'):
		self.pb=data_pb;
		self.fps_ = data_pb.fps;
		self.skip = int(30.0/self.fps_);   # can not be zero
		
		self.action_num=data_pb.actionNum;
		self.frame_size=3*224*224;
		self.feature_size=49*1024;
	    # statics
		self.labels = self.GetIntFromFile(data_pb.labels_file)
		self.fnums =self.GetIntFromFile(data_pb.num_frames_file);
		self.videos=self.GetStringFromFile(data_pb.vid_name_file);
		assert len(self.videos) == len(self.labels)
		assert len(self.videos) == len(self.fnums)

		self.video_num = len(self.labels);
		self.video_indices=np.cumsum(self.fnums)-self.fnums;
		self.video_index=0;

		# load data
		if featureType=='1024':
			self.feature = h5py.File(data_pb.feature1024_file,'r')['features'];
		elif featureType=='1024x7x7':
			self.feature = h5py.File(data_pb.feature1024x7x7_file,'r')['features'];

		if data_pb.video_file!=None:
			self.videoframe = h5py.File(data_pb.video_file,'r')['frames'];

		self.mode=None;


	def SetMode(self,mode,batchSize=None):
		"""Set the data query mode, you need to call this before try to get any samples.
		"""
		if (mode=='single'):
			self.batch_size=1;
			self.batch_num=self.video_num;
		elif (mode=='batch0'):  # one video will supply only one samples with rest part drop.
			self.batch_size=batchSize;
			self.batch_num=self.video_num/self.batch_size;
			if (self.video_num%self.batch_size!=0):
				self.batch_num+=1;
		elif (mode=='batch1'):   # one video will supply multi-samples with stride.
			pass
		elif (mode=='source'):   # each time get One video from avi/mpg file
			self.batch_size=1;
			self.batch_num=self.video_num;
		else:
			raise Exception("mode invaild");
		self.mode=mode;

	def GetSingleVideoFromSource(self,scale=1,mean=None,index=None):
		"""224*224*3
		"""
		assert(self.mode=='source');
		if (index is not None):
			self.video_index=index;

		start=0;
		end=start+self.fnums[self.video_index];
		skip= self.skip;
		n = 1 + int((end-start-1)/skip);


		videoname=self.videos[self.video_index];
		videopath=self.pb.vid_dir+videoname;
		videoframes=np.empty((0,224,224,3),dtype='uint8');

		cap=cv2.VideoCapture(videopath);
		while True:
			ok,im=cap.read()
			if not ok: break;
			im=cv2.resize(im,(224,224)); #Height x Width x Channel
			videoframes=np.append(videoframes,im[None,...],axis=0);
			for i in range(skip-1):cap.grab();  # skip,beacause grab is fast
		cap.release();
		assert(videoframes.shape[0]==n);

		if (mean is not None):
			videoframes=videoframes-mean;

		# get mask
		mask=np.ones((n,1));
		# get label
		labels=np.zeros((n,self.action_num));	
		labels[:,self.labels[self.video_index]]=1;

		self.video_index+=1;
		if (self.video_index>=self.video_num):
			self.video_index=0;

		return videoframes[:,None,...],mask,labels[:,None,:];

	def GetSingleVideo(self,index=None,scale=1,mean=None):
		"""
		return frames,mask,labels of a video
		"""
		assert(self.mode=='single');
		if (index is not None):
			self.video_index=index;
		start=self.video_indices[self.video_index];
		end=start+self.fnums[self.video_index];
		skip= self.skip;
		n = 1 + int((end-start-1)/skip);

		# get video frame
		im=self.videoframe[start:end:skip,:];

		# get mask
		mask=np.ones((n,1));

		# get label
		labels=np.zeros((n,self.action_num));	
		labels[:,self.labels[self.video_index]]=1;

		im=im.reshape([n,224,224,3])*scale;

		if (mean is not None):
			im=im-mean;

		#update index
		self.video_index+=1;
		if (self.video_index>=self.video_num):
			self.video_index=0;


		return im[:,None,...],mask,labels[:,None,:];


	def GetBatchVideo(self,nframe=30,batch_size=None,scale=1,mean=None):
		"""
		A video will be extracted only one sample.
		And a sample has at most #nframe frames.
		"""
		assert(self.mode=='batch0');
		if batch_size==None:
			batch_size=self.batch_size;
		skip= self.skip;

		im=np.zeros((nframe,batch_size,self.frame_size));
		labels=np.zeros((nframe,batch_size,self.action_num));
		mask=np.zeros((nframe,batch_size));

		for j in range(batch_size):
			start=self.video_indices[self.video_index];
			length=self.fnums[self.video_index];
			if length >= nframe*skip:
				end = start + nframe*skip;
				im[:,j,:]=self.videoframe[start:end:skip,:];
				mask[:,j]=1;
			else:
				n = 1 + int((length-1)/skip);
				im[:n,j,:]=self.videoframe[start:start+length:skip,:];
				mask[:n,j]=1;
			labels[:,j,self.labels[self.video_index]]=1;

			self.video_index+=1;
			if (self.video_index>=self.video_num):
				break;

		im=im.reshape([nframe,batch_size,224,224,3])*scale;
		if (mean is not None):
			im=im-mean;

		return im,mask,labels;

	def GetSingleVideoFeature(self,index=None):
		"""
		return features,mask,labels of a video
		"""
		assert(self.mode=='single');
		if (index is not None):
			self.video_index=index;
		start=self.video_indices[self.video_index];
		end=start+self.fnums[self.video_index];
		skip= self.skip;
		n = 1 + int((end-start-1)/skip);

		# get video frame
		x=self.feature[start:end:skip,:];

		# get mask
		mask=np.ones(n);

		# get label
		labels=np.zeros((n,self.action_num));	
		labels[:,self.labels[self.video_index]]=1;

		#update index
		self.video_index+=1;
		if (self.video_index>=self.video_num):
			self.video_index=0;

		return x[:,None,:],mask[:,None],labels[:,None,:];

	def GetCurrVideoname():
		assert(self.mode=='single' or self.mode=='source');
		return self.videos[self.video_index];

	def Reset(self):
		self.video_index=0;

	def GetIntFromFile(self,filename,type=""):
		num = []
		if filename != '':
		  for line in open(filename,'r'):
		    num.append(int(line.strip()))
		return num

	def GetStringFromFile(self,filename,type=""):
		name = []
		if filename != '':
		  for line in open(filename,'r'):
		    name.append(line.strip())
		return name







class TrainProto(object):
  def __init__(self,dataset):
    if dataset=='ucf11':
      self.fps=30;
      self.actionNum=11;
      base="/home/fancy/Desktop/UCF11/";
      self.feature1024_file    = base+'feature_1024/train_feature.h5'
      self.feature1024x7x7_file    = base+'feature_1024x7x7/train_feature.h5'
      self.video_file    = base+'video/train_video.h5' # frames
      self.num_frames_file = base+'split/train_framenum.txt'
      self.labels_file     = base+'split/train_label.txt'
      self.vid_name_file   = base+'split/train_filename.txt'
      self.vid_dir   = base+'data/'
    elif dataset=='hmdb51':
      self.fps=30;
      self.actionNum=51;
      base="/home/fancy/Desktop/HMDB51/";
      self.feature1024_file    = None
      self.feature1024x7x7_file    = base+'feature_1024x7x7/train_feature.h5'
      self.video_file    = None
      self.num_frames_file = base+'split/train_framenum.txt'
      self.labels_file     = base+'split/train_label.txt'
      self.vid_name_file   = base+'split/train_filename.txt'
      self.vid_dir   = base+'data/'

class ValidProto(object):
  def __init__(self,dataset):
    if dataset=='ucf11':
      self.fps=30;
      self.actionNum=11;
      base="/home/fancy/Desktop/UCF11/";
      self.feature1024_file    = base+'feature_1024/valid_feature.h5'
      self.feature1024x7x7_file    = base+'feature_1024x7x7/valid_feature.h5'
      self.video_file    = base+'video/valid_video.h5' # frames
      self.num_frames_file = base+'split/valid_framenum.txt'
      self.labels_file     = base+'split/valid_labes.txt'
      self.vid_name_file   = base+'split/valid_filename.txt'
      self.vid_dir   = base+'data/'
    elif dataset=='hmdb51':
      self.fps=30;
      self.actionNum=51;
      base="/home/fancy/Desktop/HMDB51/";
      self.feature1024_file    = None
      self.feature1024x7x7_file    = base+'feature_1024x7x7/valid_feature.h5'
      self.video_file    = None
      self.num_frames_file = base+'split/valid_framenum.txt'
      self.labels_file     = base+'split/valid_label.txt'
      self.vid_name_file   = base+'split/valid_filename.txt'
      self.vid_dir   = base+'data/'
  

class TestProto(object):
  def __init__(self,dataset):
    if dataset=='ucf11':
      self.fps=30;
      self.actionNum=11;
      base="/home/fancy/Desktop/UCF11/";
      self.feature1024_file    = base+'feature_1024/test_feature.h5'
      self.feature1024x7x7_file    = base+'feature_1024x7x7/test_feature.h5'
      self.video_file    = base+'video/test_video.h5' # frames
      self.num_frames_file = base+'split/test_framenum.txt'
      self.labels_file     = base+'split/test_label.txt'
      self.vid_name_file   = base+'split/test_filename.txt'
      self.vid_dir   = base+'data/'
    elif dataset=='hmdb51':
      self.fps=30;
      self.actionNum=51;
      base="/home/fancy/Desktop/HMDB51/";
      self.feature1024_file    = None
      self.feature1024x7x7_file    = base+'feature_1024x7x7/test_feature.h5'
      self.video_file    = None
      self.num_frames_file = base+'split/test_framenum.txt'
      self.labels_file     = base+'split/test_label.txt'
      self.vid_name_file   = base+'split/test_filename.txt'
      self.vid_dir   = base+'data/'
  


if __name__ == '__main__':
	data_pb = TrainProto('hmdb51');
	dh = DataHandler(data_pb,featureType='1024x7x7');

	# ##############################
	# dh.SetMode('single');

	# import cv2
	# x,mask,y=dh.GetSingleVideo(scale=255);
	# for i in range(x.shape[0]):
	# 	im=x[i,0,...].astype('uint8');
	# 	#print im.shape
	# 	cv2.imshow('frame2',im);
	# 	k = cv2.waitKey(30) & 0xff;
	# 	if k == 27:
	# 		break

	#############################
	dh.SetMode('source');

	import cv2
	x,mask,y=dh.GetSingleVideoFromSource();
	for i in range(x.shape[0]):
		im=x[i,0,...].astype('uint8');
		#print im.shape
		cv2.imshow('frame2',im);
		k = cv2.waitKey(30) & 0xff;
		if k == 27:
			break