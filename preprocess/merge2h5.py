import os
import numpy as np
import h5py

"""
Attention: numpy order is opposite to matlab matrix
if you are using matlab, the order should be opposite
"""

TotalFrameNum=150162; #  150162,58485,97412  awk '{sum+=$1} END{print sum}' train_framenum.txt
filenames="/home/fancy/Desktop/UCF11/split/train_filename.txt";
h5name="/home/fancy/Desktop/UCF11/train_features.h5";
featureDir="/home/fancy/Desktop/UCF11/feature/";


f=open(filenames);

h=h5py.File(h5name,'w');
h.create_dataset("features",(TotalFrameNum,50176),compression="gzip",chunks=True); # N x (1024*7*7)

i=0;j=0;
while (True):
	line=f.readline();
	if (line=="" or line==None):
		break;
	filename=featureDir+os.path.splitext(line.strip())[0]+".npz";
	print "process: "+filename;
	npzfile=np.load(filename);
	feat=npzfile["arr_0"].astype("float32");
	feat=feat.reshape(-1,50176);
	framenum=feat.shape[0];
	h["features"][i:i+framenum,:]=feat;
	i+=framenum;
	npzfile.close()
	if (j % 50==0):
		print "%d processed>>>>>>>>>>>>"% j;
	j+=1;

print i
f.close();
h.close();


