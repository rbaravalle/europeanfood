from random import randrange,randint
from math import log
from scipy import ndimage
#from pylab import plot, title, show , legend
import matplotlib
from matplotlib import pyplot as plt
import Image
import numpy as np
import sys
import os
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as linsolve
import time
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def spec(filename, extra):
        t = time.clock()
        cuantas = extra[0]
        a = Image.open(filename)
        #a = filename
        Nx, Ny = a.size
        L = Nx*Ny

        points = []     # number of elements in the structure
        gray = a.convert('L') # rgb 2 gray

        alphaIm = np.zeros((Nx,Ny), dtype=np.double ) # Nx rows x Ny columns
        #measure = np.zeros(4, dtype=np.double ) # Ny rows x 4 columns

        l = 4 # (maximum window size-1) / 2
        temp = np.log((1.0,3.0,5.0,7.0))
        measure = np.zeros(l*Ny).astype(np.int32)

        b = np.vstack((temp,np.ones((1,l)))).T
        AA=coo_matrix(np.kron(np.identity(Nx), b))     

        arr = np.array(gray.getdata()).astype(np.int32)

        prg = cl.Program(ctx, """
        int maxx(__global int *img, int x1, int y1, int x2, int y2, const int Ny) {
            int i, j;
            int maxim = 0;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++)
                    if(img[i*Ny + j] > maxim) maxim = img[i*Ny + j];

            return maxim;
        }
        __kernel void measure(__global int *dest, __global int *img,
                              const int Nx, const int Ny, const int l, int i, const int d) {
             int j = get_global_id(0);
             int jim = (int)(j/l)+d;
             dest[j] = maxx(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny) + 1;

        }
        """).build()

        ms = measure[0:measure.shape[0]/2]
        img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ms.nbytes)
        sh = ms.shape


        for i in range(Nx):
            d = 0
            prg.measure(queue, sh, None, dest_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(l), np.int32(i), np.int32(d))
            cl.enqueue_read_buffer(queue, dest_buf, measure[0:Ny*2]).wait()
            d = Ny/2
            prg.measure(queue, sh, None, dest_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(l), np.int32(i), np.int32(d))
            cl.enqueue_read_buffer(queue, dest_buf, measure[Ny*2:]).wait()

        # Instead of doing polyfits, a sparse linear system is constructed and solved
            if(i==Nx-1):
                print measure

            bb=np.log(measure)
            z = linsolve.lsqr(AA,bb)[0]
            z = z.reshape(2,Ny,order = 'F')
            alphaIm[i] = z[0]

        maxim = np.max(alphaIm)
        minim = np.min(alphaIm)

        print "T: ", time.clock() - t
        t = time.clock()
        # Alpha image
        plt.imshow(alphaIm, cmap=matplotlib.cm.gray)
        plt.show()
        print measure
        #return

        paso = (maxim-minim)/cuantas
        if(paso <= 0):
            # the alpha image is monofractal
            clases = map(lambda i: i+minim,np.zeros(cuantas))
        else:
            clases = np.arange(minim,maxim,paso)


        # Window
        cant = int(np.floor(np.log(Nx)))

        # concatenate the image A as [[A,A],[A,A]]
        hs = np.hstack((alphaIm,alphaIm))
        alphaIm = np.vstack((hs,hs))

        # Multifractal dimentions
        falpha = np.zeros(cuantas)

        for c in range(cuantas):
            N = np.zeros(cant+1)
            # window sizes
            for k in range(cant+1):
                sizeBlocks = 2*k+1
                numBlocks_x = int(np.ceil(Nx/sizeBlocks))
                numBlocks_y = int(np.ceil(Ny/sizeBlocks))

                flag = np.zeros((numBlocks_x,numBlocks_y))

                for i in range(1,numBlocks_x):
                    for j in range(1,numBlocks_y):
                        xi = (i-1)*sizeBlocks
                        xf = i*sizeBlocks-1
                        yi = (j-1)*sizeBlocks
                        yf = j*sizeBlocks-1
                        if(xf == xi): xf = xf+1
                        if(yf == yi): yf = yf+1
                        block = alphaIm[xi : xf, yi : yf]

                        f = 0;
                        s1 = len(block)
                        s2 = len(block[0])

                        if(c != cuantas-1):
                            # f = 1 if any pixel in block is between clases[c] and clases[c+1]
                            for w in range(s1):
                                for t in range(s2):
                                    b = block[w,t]
                                    if (b >= clases[c] and b < clases[c+1]):
                                       f = 1
                                    if(f == 1):
                                        break
                                if(f == 1):
                                    break
                        else:
                            # f = 1 if any pixel in block is equal to classes[c]+1
                            for w in range(s1):
                                for t in range(s2):
                                    b = block[w,t]
                                    if (b == clases[c]): # !!
                                       f = 1
                                    if(f == 1):
                                        break
                                if(f == 1):
                                    break
                        
                        flag[i-1,j-1] = f

                        # number of blocks with holder exponents for this class (c)
                        # and for this window size (k)
                        N[k] = N[k] + f;

            # Haussodorf (box) dimention of the alpha distribution
            falpha[c] = -np.polyfit(map(lambda i: np.log(i*2+1),range(cant+1)),np.log(map(lambda i: i+1,N)),1)[0]
        #print "T: ", time.clock() - t
        s = np.hstack((clases,falpha))
        return s
