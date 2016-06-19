# -*- coding: utf-8 -*-
import sys
import h5py
import numpy as np
import time
import cv2
import random

class DataHandler(object):
	def __init__(self, dataset_name,datatype=1,fps=30.0,shuffle=True):
		ds=Dataset(dataset_name);

		self.skip = int(fps/ds.fps);   # can not be zero
		
		self.action_num=ds.actionNum;
		self.video_dir=ds.video_dir;

		# self.frame_size=3*224*224;
		# self.feature_size=49*1024;

	    # statics
		self.labels = self.GetIntFromFile(ds.label_file)
		self.fnums =self.GetIntFromFile(ds.framenum_file);
		self.videos=self.GetStringFromFile(ds.videoname_file);
		assert len(self.videos) == len(self.labels)
		assert len(self.videos) == len(self.fnums)

		
		self.video_indices=np.cumsum(self.fnums)-self.fnums;
		

		self.indexs=[];
		self.IndexFilter(datatype,ds.split_file,classChosen=[0,9,10]); # classChosen=[0,9,10]  classChosen=[32,49,50]
		if shuffle:	random.shuffle(self.indexs);
		self.sample_num=len(self.indexs);
		self.sample_index=0;

		print "sample number:",self.sample_num;
		# load feature
		self.feature = h5py.File(ds.feature_file,'r')['feature'];

		self.mode=None;


	def IndexFilter(self,datatype,file,classChosen=None):
		k=0;
		for i in open(file):
			dt=int(i);
			if (classChosen!=None) and (dt==datatype) and (self.labels[k] in classChosen):
				self.indexs.append(k);
			elif classChosen==None and (dt==datatype):
				self.indexs.append(k);
			k+=1;


	def SetMode(self,mode,batchSize=None):
		"""Set the data query mode, you need to call this before try to get any samples.
		"""
		if (mode=='single'):
			self.batch_size=1;
			self.batch_num=self.sample_num;
		elif (mode=='batch0'):  # one video will supply only one samples with rest part drop.
			self.batch_size=batchSize;
			self.batch_num=self.sample_num/self.batch_size;
			if (self.sample_num%self.batch_size!=0):
				self.batch_num+=1;
		elif (mode=='batch1'):   # one video will supply multi-samples with stride.
			pass
		elif (mode=='source'):   # each time get One video from avi/mpg file
			self.batch_size=1;
			self.batch_num=self.sample_num;
		else:
			raise Exception("mode invaild");
		self.mode=mode;

	def GetSingleVideoFromSource(self,scale=1,size=224,gray=False,mean=None,index=None):
		"""224*224*3
		"""
		assert(self.mode=='source');
		if (index is not None):
			self.sample_index=index;
		video_index=self.indexs[self.sample_index];


		start=0;
		end=start+self.fnums[video_index];
		skip= self.skip;
		n = 1 + int((end-start-1)/skip);


		videoname=self.videos[video_index];
		videopath=self.video_dir+videoname;
		
		if gray:
			videoframes=np.empty((0,size,size),dtype='uint8');
		else:
			videoframes=np.empty((0,size,size,3),dtype='uint8');

		cap=cv2.VideoCapture(videopath);
		while True:
			ok,im=cap.read()
			if not ok: break;
			im=cv2.resize(im,(size,size)); #Height x Width x Channel
			if gray:
				im=cv2.cvtColor(im, cv2.COLOR_RGB2GRAY);
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
		labels[:,self.labels[video_index]]=1;

		self.sample_index+=1;
		if (self.sample_index>=self.sample_num):
			self.sample_index=0;

		return videoframes[:,None,...],mask,labels[:,None,:];


	def GetSingleVideoFeature(self,index=None):
		"""
		return features,mask,labels of a video
		"""
		assert(self.mode=='single');
		if (index is not None):
			self.sample_index=index;
		video_index=self.indexs[self.sample_index];

		start=self.video_indices[video_index];
		end=start+self.fnums[video_index];
		skip= self.skip;
		n = 1 + int((end-start-1)/skip);

		# get video frame
		x=self.feature[start:end:skip,:];

		# get mask
		mask=np.ones(n);

		# get label
		labels=np.zeros((n,self.action_num));	
		labels[:,self.labels[video_index]]=1;

		#update index
		self.sample_index+=1;
		if (self.sample_index>=self.sample_num):
			self.sample_index=0;

		return x[:,None,:],mask[:,None],labels[:,None,:];

	def GetCurrVideoname():
		assert(self.mode=='single' or self.mode=='source');
		return self.videos[self.indexs[self.sample_index]];

	def Reset(self,shuffle=True):
		if shuffle:	random.shuffle(self.indexs);
		self.sample_index=0;
		
	def Skip(self):
		self.sample_index+=1;

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







class Dataset(object):
  def __init__(self,dataset):
  	if (dataset=='UCF11'):
  		base="/home/fancy/Desktop/UCF11/dataset/";
  		self.fps=29.97;
  		self.actionNum=11;
  		self.feature_file   = base+'feature.h5'
		self.framenum_file 	= base+'framenum.txt'
		self.label_file     = base+'label.txt'
		self.videoname_file = base+'videoname.txt'
		self.video_dir      = base+'data/'
		self.split_file			= base+'split_byGroup.txt'  # 0:excluded 1:train 2:test

	elif (dataset=='HMDB51'):
		base="/home/fancy/Desktop/HMDB51/dataset/";
		self.fps=30;
		self.actionNum=51;
		self.feature_file   = base+'feature.h5'
		self.framenum_file 	= base+'framenum.txt'
		self.label_file     = base+'label.txt'
		self.videoname_file = base+'videoname.txt'
		self.video_dir      = base+'data/'
		self.split_file			= base+'split_linear2.txt'  # 0:excluded 1:train 2:test

	else:
		raise ValueError("dataset name is valid");


def VisualVideo():
	dh = DataHandler('HMDB51',datatype=2);
	dh.SetMode('source');

	print dh.batch_num

	import cv2
	x,mask,y=dh.GetSingleVideoFromSource(size=64,gray=True);
	for i in range(x.shape[0]):
		im=x[i,0,...].astype('uint8');
		#print im.shape
		cv2.imshow('frame2',im);
		k = cv2.waitKey(0) & 0xff;
		if k == 27:  # esc
			break
		elif k==ord('s'):
			cv2.imwrite('%d.jpg'%(i),im);


def VisualizeFeature():
	dh = DataHandler('HMDB51',datatype=0);
	dh.SetMode('single');

	x,mask,y=dh.GetSingleVideoFeature(index=100);
	print x.shape;
	x=x.reshape([x.shape[0],1,1024,49]);
	x=x.mean(3);
	x=x.reshape([x.shape[0],1,32,32]);

	for i in range(x.shape[0]):
		im=x[i,0,...];
		#print im.shape
		#im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR);
		print im.min(),im.max()
		im=im/im.max();
		im=cv2.resize(im,(224,224));
		cv2.imshow('frame2',im);
		k = cv2.waitKey(0) & 0xff;
		if k == 27:  # esc
			break
		elif k==ord('s'):
			cv2.imwrite('%d.jpg'%(i),im);
			





if __name__ == '__main__':
	VisualVideo();