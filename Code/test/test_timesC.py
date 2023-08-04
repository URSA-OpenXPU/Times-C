import math
from re import M

import os


import numpy as np
import time
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft

import sys


def power_iteration_reduce(A, num_iterations):
    
    v = np.random.rand(A.shape[1])
    v = v / np.linalg.norm(v)
    for i in range(num_iterations):
        v = A.dot(v)
        if i%2 == 0:
            v = v / np.linalg.norm(v)

    lambda_ = np.dot(np.dot(A, v), v) / np.dot(v, v)
    return lambda_, v



def power_iteration(A, num_iterations):
    v = np.random.rand(A.shape[1])
    #v = np.ones(A.shape[1])
    v = v / np.linalg.norm(v)
    for i in range(num_iterations):
        v = A.dot(v)
        v = v / np.linalg.norm(v)
    lambda_ = np.dot(np.dot(A, v), v) / np.dot(v, v)
    return lambda_, v


# z-normalized
def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    # if mns.dim different with a.dim
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

# Cross-correlation 2m  - 1
def _ncc_c(x, y):
   
    #norm(x, ord = None, axis = None, keepdims = False) 
    den = np.array(norm(x) * norm(y))
    #print("den: ",den)
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den


def _ncc_c_2dim(x, y):
   
    den = np.array(norm(x, axis=1) * norm(y))
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[:,-(x_len-1):], cc[:,:x_len]), axis=1)
    return np.real(cc) / den[:, np.newaxis]


def _ncc_c_3dim(x, y):
   
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    #kkk
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):

    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _preSBD_extract_shape(x, cur_center):
    if cur_center.sum() != 0:
        x_cpy = _ncc_c_2dim(x, cur_center)
        idx = x_cpy.argmax(1)
        for i in range(len(x)):
            x[i] = roll_zeropad(x[i], (idx[i] + 1) - max(len(x[i]), len(cur_center)))
    return x


def _extract_shape(a, n):
    

    if len(a) == 0:
        return np.zeros((1, n))
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    
    
    iternums = m.shape[1]
    """ if iternums < 500:
        iternums = 500 """
    if iternums > 1000:
        iternums = 1000
    lambda_, centroid = power_iteration(m,iternums)
    #lambda_, centroid = power_iteration_acc(A,iternums)
    flag = True
    if lambda_ == 0:
        flag = False
    if flag:
        finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())
        if finddistance1 >= finddistance2:
            centroid *= -1
            
        vecright = lambda_ * centroid
        vecleft = np.dot(m, centroid)
        pos = 0
        for i in range(len(vecleft)):
            if vecleft[i] != 0 and vecright[i] != 0:
                pos = i
                break
        if (vecleft[pos] < 0 and vecright[pos] > 0) or (vecleft[pos] < 0 and vecright[pos] > 0):
            lambda_ = -1*lambda_
            vecright = -1*vecleft
    vecdiff = vecright - vecleft
    sumdiff = abs(sum(vecdiff))
    threhold = m.shape[1]*0.01
    
    #print("lambda_Power: ",lambda_)
    if lambda_ <= 0 or sumdiff > threhold:
        print("more eigh")
        lambdas, vec = eigh(m)
        centroid = vec[:, -1]
        finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

        if finddistance1 >= finddistance2:
            centroid *= -1
        lambda_ = lambdas[m.shape[1]-1]
    
    '''
    lambdas, vec = eigh(m)
    centroid = vec[:, -1]
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())
    if finddistance1 >= finddistance2:
        centroid *= -1
    lambda_ = lambdas[m.shape[1]-1]
    print("lambda_eigh: ",lambda_)
    '''
    return zscore(centroid, ddof=1)


def centerMerge(x, cur_center, n):
    x = _preSBD_extract_shape(x, cur_center)
    center = _extract_shape(x, n)
    return center


def _kshape(x, k):
    x = zscore(x,axis=1)
    m = x.shape[0]
    n = x.shape[1]
    idx = randint(0, k, size=m)
    centroids = np.zeros((k, x.shape[1]))
    distances = np.empty((m, k))
    
    iternum = 10
    print("start iter")
    for i in range(iternum):
        
        print("iter i ------------>", i)
        old_idx = idx
        
        offset = np.zeros(len(idx),dtype=int)
        counter = np.zeros(k,dtype=int)
        for i in range(len(idx)):
            offset[i] = counter[idx[i]]
            counter[idx[i]] += 1
            
        preSum = np.cumsum(counter)
        preSum = preSum*x.shape[1]
        
        dataAgg = []
        for i in range(k):
            my_2d_array = np.zeros((counter[i], x.shape[1]))
            dataAgg.append(my_2d_array)

        for i in range(m):
            dataAgg[idx[i]][offset[i]] = x[i]

        """ pool = multiprocessing.Pool(32)
        for j in range(k):
            centroids[j] = pool.apply_async(func=centerMerge,args=(dataAgg[j], centroids[j], n)).get()
            #dataAgg[j] = _preSBD_extract_shape(dataAgg[j], centroids[j])
            #centroids[j] = _extract_shape(dataAgg[j], n)
        pool.close()
        pool.join() """
        
        for j in range(k):
            dataAgg[j] = _preSBD_extract_shape(dataAgg[j], centroids[j])
            centroids[j] = _extract_shape(dataAgg[j], n)
            

        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        #print(idx)
        if np.array_equal(old_idx, idx):
            iternum = i
            break
    return idx, centroids, iternum

#@profile
def kshape(x, k):
    idx, centroids, iternum = _kshape(np.array(x), k)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return idx,clusters,iternum


def fileReader(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:
            odom = line.split(",")        
            my_list.append(odom)
    return my_list
    
def fileReader2(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:
            #odom = line.split(",")
            odom = line.split("	")         
            my_list.append(odom)
    return my_list



#@count_info

def labeled_process(filename):
    m = fileReader(filename)
    #m = fileReader2(filename)

    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    mat = np.array(m)
    data = mat[:,1:]
    labe = mat[:,:1]

    k = int(np.max(labe))
    u = int(np.min(labe))
    if u == 0:
        k += 1
    #k = para
    idx,centers,iternum = kshape(data, k)
    #print("centers",centers)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/timesC.txt", idx, delimiter=",", fmt='%.0f')


#@count_info

def unlabeled_process(filename, k):
    m = fileReader(filename)

    for i in range(len(m)):
        del m[i][len(m[i])-1]
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    mat = np.array(m)

    idx,centers,iternum = kshape(mat, k)
    #print("centers",centers)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/timesC.txt", idx, delimiter=",", fmt='%.0f')




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("please input right parameter. Usage: test_timesC.py /path/to/dataset k flag")
    else:
        filePath = sys.argv[1]
        k =  int(sys.argv[2])
        flag =  int(sys.argv[3])
        #unlabeled_process(filename27, 2)
        if(flag == 1):
            unlabeled_process(filePath, k)
        if(flag == 0):
            labeled_process(filePath)
    
    
