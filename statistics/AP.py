import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

datatype=2;  #test
data_dir='/home/fancy/Desktop/UCF11/';
labelsfile=data_dir+'dataset/label.txt';
# splitfile=data_dir+'dataset/split_linear2.txt';
splitfile=data_dir+'dataset/split_byGroup.txt';
labelnames=['shooting','biking','diving','golf','riding','juggle','swing','tennis','jumping','spiking','walk_dog'];
# labelnames=["brush_hair","cartwheel","catch","chew","clap","climb","climb_stairs","dive","draw_sword","dribble", \
# 			"drink","eat","fall_floor","fencing","flic_flac","golf","handstand","hit","hug","jump", \
# 			"kick","kick_ball","kiss","laugh","pick","pour","pullup","punch","push","pushup", \
# 			"ride_bike","ride_horse","run","shake_hands","shoot_ball","shoot_bow","shoot_gun","sit","situp","smile", \
# 			"smoke","somersault","stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"];


resultsfile='/home/fancy/Desktop/project/params/UCF11/avg_mLSTM/model_result.txt';


testlabel=[];
i=0;
for line in open(splitfile,'r'):
	_type=int(line.strip())
	if _type==datatype:
		testlabel.append(i);
	i+=1;
testlabel=np.array(testlabel);

labels=[];
for line in open(labelsfile,'r'):
	labels.append(int(line.strip()));
labels=np.array(labels);


for i in range(testlabel.shape[0]):
	testlabel[i]=labels[testlabel[i]];



res=[];
for line in open(resultsfile,'r'):
	res.append(int(line.strip()));
res=np.array(res);


# P=precision_score(testlabel, res, average="macro");
# R=recall_score(testlabel, res,average="macro");
# F1=f1_score(testlabel, res, average="macro");

print classification_report(testlabel,res,target_names=labelnames);
# print P,R,F1;

print "acc: ",accuracy_score(testlabel, res);


#data=np.append(labels[:,None],res[:,None],axis=1);
