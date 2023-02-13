import h5py
import numpy as np

fileName = 'train.h5'
filePath = 'F:\\OnlinePacket\\programfiles\\python\\GNNtest\\GNNbuildProject\\rebuild\\datatest\\test1\\'
h5f = h5py.File(filePath + fileName, 'r')


print([key for key in h5f.keys()])

rgb = np.array(h5f['APDs'])  # 创建以h5中rgb这一group数据为内容的numpy类型array矩阵；
print(rgb)
# rgb = np.transpose(rgb, (1, 2, 0))
# print('np.array type:', type(rgb))
# print('np.array dtype:', rgb.dtype)
rgb = np.asfarray(rgb)
print('np.asfarray type:', type(rgb))
print('np.asfarray dtype:', rgb.dtype)

with h5py.File(filePath + fileName,"r") as f:
    for key in f.keys():
    	 #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
        print(key)
        print(f[key], key, f[key].name)
