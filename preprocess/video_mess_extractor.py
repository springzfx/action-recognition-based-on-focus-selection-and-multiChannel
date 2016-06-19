import cv2

"""
Extract frameNum,fps,height,width from video, save in a file
Attention cap.get(cv2.CAP_PROP_FRAME_COUNT) is not exactly equals framenum read by opencv
"""

# videofiles="/home/fancy/Desktop/UCF11/filename.txt";
# videodir="/home/fancy/Desktop/UCF11/data/";

videofiles="/home/fancy/Desktop/HMDB51/split/test_filename.txt";
videodir="/home/fancy/Desktop/HMDB51/data/";

f=open(videofiles)
s=open("/home/fancy/Desktop/HMDB51/split/test_video_mess.txt",'w')


k=0;
while True:
	line=f.readline();
	if (line=="" or line==None):break
	line=line.strip();

	videoFile=videodir+line;
	print videoFile
	cap=cv2.VideoCapture(videoFile);
	# get how many frames can be read
	i=0;a=True;
	while a:
		a,b=cap.read()
		if (a):i+=1;

	strs="%s %.0f %.2f %.0f %.0f\n" %( \
			  line, \
			  i, \
		      cap.get(cv2.CAP_PROP_FPS), \
		      cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
		      cap.get(cv2.CAP_PROP_FRAME_WIDTH) \
		      );
	s.write(strs);
	cap.release()

s.close();
f.close()