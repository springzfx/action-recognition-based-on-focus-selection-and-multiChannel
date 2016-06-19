import h5py
import numpy
filedir='/home/fancy/Desktop/HMDB51/feature_1024x7x7/';

h=h5py.File(filedir+'feature.h5','a');
h1=h5py.File(filedir+'train_feature.h5','r');
h2=h5py.File(filedir+'valid_feature.h5','r');
h3=h5py.File(filedir+'test_feature.h5','r');


s1=h1['features'].shape[0];
s2=h2['features'].shape[0];
s3=h3['features'].shape[0];


TotalFrameNum=s1+s2+s3;
h.create_dataset("feature",(TotalFrameNum,1024*7*7),compression="gzip",chunks=True);






def Move(i,n,f):
	t=i;
	s=i+n;
	while (i<s):
		j=i+k;
		if (j>s):
			j=s;
		print i,j
		h['feature'][i:j,:]=f[i-t:j-t,:];
		i+=k;




k=10000;
Move(0,s1,h1['features']);
Move(s1,s2,h2['features']);
Move(s1+s2,s3,h3['features']);
h.close();