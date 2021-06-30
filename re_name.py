# #批量修改文件名字
# import string
# import random
# import os
# import shutil
# def rename(path,newname):
# 	filelist = os.listdir(path)
# 	#获取文件下的所有文件名   #Newdir = os.path.join(path, newname + '.jpg') % m  # 这里由于filetype是一个列表
# 	m=0
# 	for files in filelist:
# 		Olddir = path + files  #原来的文件路径
# 		filename = os.path.splitext(files)[0] #文件名
# 		#filetype = os.path.splitext(files)[1] #后缀名,是一个列表
# 		Newdir = os.path.join(path , newname + '.jpg') % m  #这里由于filetype是一个列表
# 		m += 1
# 		os.rename(Olddir , Newdir)
#
# rename('D:\\cervical\\lsil\\' , '%4d')
#
#

# -*- coding:utf8 -*-

import os
class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self):
        self.path = 'D:/cervical/train/3'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '3'+'_'+'('+str(i)+')'+'.jpg')   # 重新命名
                try:
                    os.rename(src, dst)
                    print
                    'converting %s to %s ...' % (src, dst)
                    i = i + 1
                except:
                    continue
        print
        'total %d to rename & converted %d jpgs' % (total_num, i)

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()



