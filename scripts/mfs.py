import Image
import numpy as np
from math import exp, log10
import scipy.ndimage.filters as sf
import matplotlib
from matplotlib import pyplot as plt
import cv
from numpy.fft import fft2, ifft2

def gauss_kern(size, sizey):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    m = np.float32(size)
    n = np.float32(sizey)
    sigma = 1;    
    y, x = np.mgrid[-(m-1)/2:(m-1)/2+1, -(n-1)/2:(n-1)/2+1]

    b = 2*(sigma**2)
    x2 = map(lambda i: map( lambda j: j**2,i), x)
    y2 = map(lambda i: map( lambda j: j**2,i), y)
    g = np.sum([x2,y2],axis=0).astype(np.float32)
    g = np.array(map(lambda i: map( lambda j: exp(-j/b),i), g)).astype(np.float32)
    return g / g.sum()


def mfs(im,extra):
    
    #mfs Computes the MFS vector for the input measurement image  im 
    #
    # parameters: ind_num -> determines how many levels are used when computing the density 
    #                          choose 1 for using  directly the image measurement im or
    #                          >= 6 for computing the density of im (quite stable for >=5)      
    #              f_num----> determines the dimension of  MFS vector
    #              ite_num  ---> determines how many levels are used when computing MFS for each level set
    #                            (quite stable for >= 3)
    #
    #MFS = mfs(im) computes the MFS for input  im with default setting
    #                  
    #MFS = mfs(im,ind_num) computes the MFS with ind_num density levels
    #
    #MFS = mfs(im,ind_num, f_num) computes the MFS of dimension f_num for input im 
    #                             with ind_num density levels
    #
    #MFS = mfs(im, ind_num, f_num,ite_num) computes the MFS of dimension f_num for input measurement im
    #                                  using ite_num level iterations in the
    #                                  estimation of the fractal dimension and using ind_num level
    #                                  iterations in the density estimation.
    #
    #Author: Yong Xu, Hui Ji
    #Date: Apr 24, 2007
    #Code ported to python : Rodrigo Baravalle. December 2012


    im = Image.open(im)

    ind_num = 1; #density counting levels
    f_num = 26;  #the dimension of MFS
    ite_num = 3; #Box counting levels              

    if(len(extra) == 1):
        ind_num = extra[0]  #density counting levels
        f_num = 26          #the dimension of MFS
        ite_num = 3         # iteration levels in estimating fractal dimension
    if(len(extra) == 2):
        ind_num = extra[0]
        f_num = extra[1]
        ite_num = 3
    if(len(extra) == 3):
        ind_num = extra[0]
        f_num = extra[1]
        ite_num = extra[2]

    # Preprocessing: if IM is a color image convert it to a gray image 
    im = im.convert("L")
    im = np.array(im.getdata()).reshape(im.size)

    #Using [0..255] to denote the intensity profile of the image
    grayscale_box =[0, 255];

    #Preprocessing: default intensity value of image ranges from 0 to 255
    if(abs(im).max()< 1):
        im = im * grayscale_box[1];
    

    #######################

    ### Estimating density function of the image
    ### by solving least squares for D in  the equation  
    ### log10(bw) = D*log10(c) + b 
    r = 1.0/max(im.shape)
    c = np.dot(range(1,ind_num+1),r)

    c = map(lambda i: log10(i), c)
    bw = np.zeros((ind_num,im.shape[0],im.shape[1])).astype(np.float32)
    bw[0] = map(lambda i:map(lambda j: j+1,i) , im)

    import scipy.signal
    for k in range(1,ind_num):
        bw[k] = scipy.signal.convolve2d(bw[0], gauss_kern(k+1,k+1),mode="same") # FALTA ?!?!
        #bw[k] = map(lambda i: map (lambda j: log10(j*(k**2)),i), bw[k])

    
    n1 = c[0]*c[0]
    n2 = np.log10(bw[0])*c[0]

    for k in range(1,ind_num):
        n1 = n1+c[k]*c[k]
        n2 = n2 + bw[k]*c[k]

    #if(ind_num >1):    FALTA
    #    D = (n2*ind_num-sum(c)*sum(bw,3))./(n1*ind_num -sum(c)*sum(c));

    #if (ind_num > 1)
    #    max_D  = 4; min_D =1;
    #    D = grayscale_box(2)*(D-min_D)/(max_D - min_D)+grayscale_box(1);
    #else FALTA
    D = im

    #Partition the density
    # throw away the boundary
    D = D[ind_num-1:D.shape[0]-ind_num+1, ind_num-1:D.shape[1]-ind_num+1] # REVISAR
    IM = np.zeros(D.shape)
    gap = np.ceil((grayscale_box[1] - grayscale_box[0])/np.float32(f_num));
    center = np.zeros(f_num+1);
    for k in range(1,f_num+1):
        bin_min = (k-1) * gap;
        bin_max = k * gap - 1;
        center[k-1] = round((bin_min + bin_max) / 2);
        aux = np.zeros(D.shape)
        D = ((D <= bin_max) & (D >= bin_min)).choose(D,center[k-1])
        
    IM = D

    #Constructing the filter for approximating log fitting
    r = max(IM.shape)
    c = np.zeros(ite_num)
    c[0] = 1;
    for k in range(1,ite_num):
        c[k] = c[k-1]/(k+1)
    c = c / sum(c);

    #Construct level sets
    Idx_IM = np.zeros(IM.shape);
    for k in range(0,f_num):
        IM = (IM == center[k]).choose(IM,k+1)

    Idx_IM = IM
    IM = np.zeros(IM.shape)

    #Estimate MFS by box-counting
    num = np.zeros(ite_num)
    MFS = np.zeros(f_num)
    for k in range(1,f_num+1):
        #idx = find(Idx_IM == k)
        IM = np.zeros(IM.shape)
        IM = (Idx_IM==k).choose(Idx_IM,255+k)
        IM = (IM<255+k).choose(IM,0)
        IM = (IM>0).choose(IM,1)
        temp = max(IM.sum(),1)
        print r, temp
        num[0] = log10(temp)/log10(r);    
        for j in range(2,ite_num+1):
            mask = np.ones((j,j))
            bw = scipy.signal.convolve2d(IM, mask,mode="same") # FALTA ?!?!
            indx = np.arange(0,IM.shape[0],j)
            indy = np.arange(0,IM.shape[1],j)
            bw = bw[np.ix_(indx,indy)]
            idx = (bw>0).sum()
            temp = max(idx,1)
            num[j-1] = log10(temp)/log10(r/j)

        print num
        MFS[k-1] = sum(c*num) #sum(c.*num)

    print MFS
    return MFS

