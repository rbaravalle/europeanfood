from random import randrange,randint
from math import log
from scipy import ndimage
import Image
import numpy as np
import sys
import os
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as linsolve
import pyopencl as cl
import pyopencl.array as cla
import mfs

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def spec(filename, extra):
        #cuantas = extra[0]
        #OPEN_IMAGE = extra[1]
        #if(OPEN_IMAGE==True):
        a = Image.open(filename)
        gray = a.convert('L') # rgb 2 gray
        Nx, Ny = a.size
        #else: # np array
        #    a = filename
            #Nx, Ny = a.shape
        #    Nx, Ny = a.size
        L = Nx*Ny

        arr = np.array(gray.getdata()).astype(np.int32)

        alphaIm = np.zeros((Nx,Ny), dtype=np.double ) # Nx rows x Ny columns

        l = 4 # (maximum window size-1) / 2
        temp = map(lambda i: 2*i+1, range(l))
        temp = np.log(temp)
        measure = np.zeros(l*Ny).astype(np.int32)

        b = np.vstack((temp,np.ones((1,l)))).T
        AA=coo_matrix(np.kron(np.identity(Ny), b))     

        # which: which measure to take
        which = 4

        prg = cl.Program(ctx, """
        int maxx(__global int *img, int x1, int y1, int x2, int y2, const int Ny) {
            int i, j;
            int maxim = 0;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++)
                    if(img[i*Ny + j] > maxim) maxim = img[i*Ny + j];

            return maxim;
        }
        int minn(__global int *img, int x1, int y1, int x2, int y2, const int Ny) {
            int i, j;
            int minim = 255;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++)
                    if(img[i*Ny + j] < minim) minim = img[i*Ny + j];

            return minim;
        }
        int summ(__global int *img, int x1, int y1, int x2, int y2, const int Ny) {
            int i, j;
            int summ = 0;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++)
                    summ += img[i*Ny + j];

            return summ;
        }
        int iso(__global int *img, int x1, int y1, int x2, int y2, const int Ny, const int x, const int y) {
            int i, j;
            int cant = 0;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++)
                    if(img[i*Ny + j] == img[x*Ny + y]) cant++;

            return cant;
        }
        int difAbsCentral(__global int *img, int x1, int y1, int x2, int y2, const int Ny, const int x, const int y) {
            int i, j;
            int maxim = 0;
            for(i = x1; i < x2; i++)
                for(j = y1; j < y2; j++) {
                    int dif = abs(img[i*Ny + j]-img[x*Ny + y]);
                    if(dif > maxim) maxim = dif;
                }

            return maxim;
        }
        __kernel void measure(__global int *dest, __global int *img, const int Nx,
                                const int Ny, const int l, int i, const int d, const int which) {
             int j = get_global_id(0);
             int jim = (int)(j/l)+d;
             if(which == 0)
                 dest[j] = maxx(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                    min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny) + 1;
             if(which == 1)
                 dest[j] = minn(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                    min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny) + 1;
             if(which == 2)
                 dest[j] = summ(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                    min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny) + 1;
             if(which == 3)
                 dest[j] = iso(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                    min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny, i, j) + 1;
             if(which == 4)
                 dest[j] = difAbsCentral(img,max(i-((j%l)+1),0),max(jim-((j%l)+1),0),
                                    min(i+(j%l)+1,Nx-1),min(jim+(j%l)+1,Ny-1), Ny, i, j) + 1;
            
        }
        """).build()

        d = measure.shape[0]/2
        ms = measure[0:l*d]
        img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ms.nbytes)
        sh = ms.shape

        for i in range(Nx):
            prg.measure(queue, sh, None, dest_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(l), np.int32(i), np.int32(0), np.int32(which))
            cl.enqueue_read_buffer(queue, dest_buf, measure[0:l*d]).wait()
            prg.measure(queue, sh, None, dest_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(l), np.int32(i), np.int32(d), np.int32(which))
            cl.enqueue_read_buffer(queue, dest_buf, measure[l*d:]).wait()

            # Instead of doing polyfits, a sparse linear system is constructed and solved

            bb=np.log(measure)
            z = linsolve.lsqr(AA,bb)[0]
            z = z.reshape(2,Ny,order = 'F')
            alphaIm[i] = z[0]

        maxim = np.max(alphaIm)
        minim = np.min(alphaIm)

        alphaIm = np.floor(255*(alphaIm-minim)/(maxim-minim))

        #import matplotlib
        #from matplotlib import pyplot as plt
        # Alpha image
        #plt.imshow(alphaIm, cmap=matplotlib.cm.gray)
        #plt.show()
        #return
        extra[3] = False
        return mfs.mfs(alphaIm,extra)
        
