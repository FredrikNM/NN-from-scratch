#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp
import time
import matplotlib.pyplot as plt
import os

path = os.getcwd()
print(path)
# filename = 'C:/Users/fredr/Desktop/3490 Biologisk Programmering/Programmering/data/movements_day1-3.dat'
filename = path+'\data\movements_day1-3.dat'


movements = np.loadtxt(filename,delimiter='\t')
# print np.shape(movements)
# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)
# print movements[0:2,1:4]
# print type(movements)
# print movements[0:2,1:4].max(axis=0) #* np.ones((1,41))
# print np.shape(movements[0:2,:])
# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)
#print imax
# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]
# print movements
# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1
    # print x

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

# Split data into 3 sets

# Training updates the weights of the network and thus improves the network
train = movements[::2,0:40]
train_targets = target[::2]

# Validation checks how well the network is performing and when to stop
valid = movements[1::4,0:40]
valid_targets = target[1::4]

# Test data is used to evaluate how good the completely trained network is.
test = movements[3::4,0:40]
test_targets = target[3::4]

# Initialize the network:

# Run training:
for i in range(2):
	iteratio = [1000, 10000]
	hidden = [15,30]
	net = mlp.mlp(train, train_targets, hidden[i], momentum = 0.00)
	for j in range(2):
		start_time = time.time()
		rounds = iteratio[j]
		print('START')
		print('hidden nodes = %s ,iterations choosen = %s' % (hidden[i],iteratio[j]))
		terror, verror, itera = net.earlystopping(train, train_targets, valid, valid_targets, rounds, treshold = 0.03)
		time_taken = time.time() - start_time
		print("--------- %s seconds ---------\n" % time_taken)
		h, y, out = net.forward(test, test_targets)
		print( 'errorrate = %s%% after running NN %s iterations' % (y, itera))
		print(net.confusion(out, test_targets))
		print('THE END')
		print('====================================')
		line1, = plt.plot(terror, label = "train", linestyle= '--')
		line2, = plt.plot(verror, label = "val", linestyle= '-')
		plt.legend(loc='upper right')
		plt.show()



'''
python movements.py
START
hidden nodes = 2 ,iterations choosen = 10
--------- 0.00999999046326 seconds ---------

errorrate = 0.765765765766% after running NN 10 iterations
[[  0.   0.   0.   0.   0.   0.   0.   0.]
 [  6.  12.   6.   2.   0.   2.   1.  11.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [ 10.   0.   5.  12.  14.  16.  13.   1.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]]
THE END
====================================
START
hidden nodes = 2 ,iterations choosen = 100
--------- 0.0820000171661 seconds ---------

errorrate = 0.54954954955% after running NN 100 iterations
[[  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   1.   0.]
 [  2.   0.   1.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [ 12.   0.   9.   7.  12.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   2.   2.  18.  13.   0.]
 [  2.   0.   1.   5.   0.   0.   0.  12.]]
THE END
====================================
START
hidden nodes = 2 ,iterations choosen = 1000
--------- 0.714999914169 seconds ---------

errorrate = 0.495495495495% after running NN 945 iterations
[[  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   1.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   2.   7.   1.   0.   0.   0.]
 [ 14.   0.   9.   4.  13.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   1.   0.]
 [  0.   0.   0.   1.   0.  18.  12.   0.]
 [  2.   0.   0.   2.   0.   0.   0.  12.]]
THE END
====================================
START
hidden nodes = 8 ,iterations choosen = 10
--------- 0.0130000114441 seconds ---------

errorrate = 0.396396396396% after running NN 10 iterations
[[ 16.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   3.   0.   4.   5.   0.]
 [  0.   0.  11.   1.   0.   8.   5.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   8.  14.   0.   1.   1.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   6.   3.   0.]
 [  0.   0.   0.   2.   0.   0.   0.  11.]]
THE END
====================================
START
hidden nodes = 8 ,iterations choosen = 100
--------- 0.0939998626709 seconds ---------

errorrate = 0.135135135135% after running NN 100 iterations
[[ 14.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   2.   0.]
 [  0.   0.  11.   0.   0.   0.   0.   0.]
 [  0.   0.   0.  13.   1.   0.   0.   1.]
 [  0.   0.   0.   1.  13.   0.   0.   0.]
 [  0.   0.   0.   0.   0.  13.   3.   0.]
 [  0.   0.   0.   0.   0.   5.   9.   0.]
 [  2.   0.   0.   0.   0.   0.   0.  11.]]
THE END
====================================
START
hidden nodes = 8 ,iterations choosen = 1000
--------- 0.250999927521 seconds ---------

errorrate = 0.0630630630631% after running NN 271 iterations
[[ 16.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   0.   0.]
 [  0.   0.  11.   0.   0.   0.   0.   0.]
 [  0.   0.   0.  14.   1.   0.   2.   1.]
 [  0.   0.   0.   0.  13.   0.   0.   0.]
 [  0.   0.   0.   0.   0.  16.   1.   0.]
 [  0.   0.   0.   0.   0.   2.  11.   0.]
 [  0.   0.   0.   0.   0.   0.   0.  11.]]
THE END
====================================
START
hidden nodes = 30 ,iterations choosen = 10
--------- 0.0150001049042 seconds ---------

errorrate = 0.153153153153% after running NN 10 iterations
[[ 15.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   1.   0.]
 [  0.   0.  11.   1.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   4.  14.   0.   0.   0.]
 [  0.   0.   0.   0.   0.  18.   1.   0.]
 [  0.   0.   0.   4.   0.   0.  12.   0.]
 [  1.   0.   0.   5.   0.   0.   0.  12.]]
THE END
====================================
START
hidden nodes = 30 ,iterations choosen = 100
--------- 0.119999885559 seconds ---------

errorrate = 0.036036036036% after running NN 100 iterations
[[ 16.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   0.   0.]
 [  0.   0.  11.   0.   0.   0.   0.   0.]
 [  0.   0.   0.  13.   1.   0.   0.   0.]
 [  0.   0.   0.   1.  13.   0.   0.   0.]
 [  0.   0.   0.   0.   0.  17.   1.   0.]
 [  0.   0.   0.   0.   0.   1.  13.   0.]
 [  0.   0.   0.   0.   0.   0.   0.  12.]]
THE END
====================================
START
hidden nodes = 30 ,iterations choosen = 1000
--------- 0.40299987793 seconds ---------

errorrate = 0.045045045045% after running NN 375 iterations
[[ 16.   0.   0.   0.   0.   0.   0.   0.]
 [  0.  12.   0.   0.   0.   0.   0.   0.]
 [  0.   0.  11.   0.   0.   0.   0.   0.]
 [  0.   0.   0.  13.   1.   0.   1.   0.]
 [  0.   0.   0.   1.  13.   0.   0.   0.]
 [  0.   0.   0.   0.   0.  17.   1.   0.]
 [  0.   0.   0.   0.   0.   1.  12.   0.]
 [  0.   0.   0.   0.   0.   0.   0.  12.]]
THE END
====================================
'''