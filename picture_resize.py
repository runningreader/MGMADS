#批量处理图片resize
import os
import time
import cv2

def alter(path,object):
	result=[]
	s=os.listdir(path)
	count=1
	for i in s:
		document=os.path.join(path,i)
		img = cv2.imread(document)
		img=cv2.resize(img,(235,235))
		listStr=[str(int(time.time())),str(count)]
		fileName=''.join(listStr)
		cv2.imwrite(object+os.sep+'%s.jpg'%fileName,img)
		count=count+1
alter('C:\\Users\\Admin\\Desktop\\6','C:\\Users\\Admin\\Desktop\\6')


