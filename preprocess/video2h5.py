import cv2
import numpy as np
import h5py


"""
save video frames to h5py file
the frame will be resize,and re-range to [0-1]
"""


TotalFrameNum=150162; #  150162,58485,97412  awk '{sum+=$1} END{print sum}' train_framenum.txt
videofiles="/home/fancy/Desktop/UCF11/split/train_filename.txt";
h5name="/home/fancy/Desktop/UCF11/train_video.h5";

videodir="/home/fancy/Desktop/UCF11/data/";


f=open(videofiles);
h=h5py.File(h5name,'w');
h.create_dataset("frames",(TotalFrameNum,150528),compression="gzip",chunks=True); #3*224*224=150528



i=0;j=0;
while True:
	line=f.readline();
	if (line=="" or line==None):break
	line=line.strip();

	videoFile=videodir+line;
	print videoFile
	cap=cv2.VideoCapture(videoFile);
	# get how many frames can be read
	print cap.get(cv2.CAP_PROP_FRAME_COUNT);
	videoframes=np.empty((0,150528),dtype='float32');
	k=0;
	while True:
		ok,im=cap.read()
		if not ok: break;
		im=cv2.resize(im,(224,224));
		im=im.astype('float32');
		im=im/255;
		videoframes=np.append(videoframes,im.reshape((1,-1)),axis=0);
		k+=1;
	h["frames"][i:i+k,:]=videoframes;
	i+=k;
	cap.release()
	if (j % 50==0):
		print "%d processed>>>>>>>>>>>>"% j;
	j+=1;

h.close();
f.close()


