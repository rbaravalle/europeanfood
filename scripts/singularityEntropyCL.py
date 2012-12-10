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

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def spec(filename, extra):
        cuantas = extra[0]
        OPEN_IMAGE = extra[1]
        if(OPEN_IMAGE==True):
            a = Image.open(filename)
            Nx, Ny = a.size
        else: # np array
            a = filename
            #Nx, Ny = a.shape
            Nx, Ny = a.size
        L = Nx*Ny

        points = []     # number of elements in the structure
        RESHAPE = extra[3]
        CONVERT = extra[2]
        if(CONVERT == True):
            #gray = a.convert('L') # rgb 2 gray
            #arr = np.array(gray.getdata()).astype(np.int32)
            arr = np.array(filename.getdata()).astype(np.int32)
        else:
            if(RESHAPE == True): # ARGHH
                arr = np.array(a).reshape(a.shape[0]*a.shape[1])
            else:
                arr = a

        alphaIm = np.zeros((Nx,Ny), dtype=np.float32 ) # Nx rows x Ny columns

        l = 4 # (maximum window size-1) / 2
        temp = map(lambda i: 2*i+1, range(l))
        temp = np.log(temp)
        measure = np.zeros(l*Ny).astype(np.int32)

        b = np.vstack((temp,np.ones((1,l)))).T
        AA=coo_matrix(np.kron(np.identity(Ny), b))     

        prg = cl.Program(ctx, """
        __kernel void measure(__global float *alphaIm, __global int *img, const int Nx,
                                const int Ny, const int size) {
             int i = get_global_id(0);
             int j = get_global_id(1);

             // make histogram of region
             int hist[256];
             int t;
             for(t = 0; t < 256; t++) hist[t] = 0;
             int xi = max(i-size,0);
             int yi = max(j-size,0);
             int xf = min(i+size,Nx-1);
             int yf = min(j+size,Ny-1);
             int u , v;
             for(int u = xi; u <= xf; u++)
                 for(int v = yi; v <= yf; v++)
                    hist[img[u*Ny+v]]++;
             float res = 0;
             int s;
             float total = (yf-yi)*(xf-xi); // size of region
             for(s = 0; s <= 255; s++) {
                 float v = hist[s]/total; // probability
                 res += v*log2(v+0.0001);
             }

             alphaIm[i*Ny+j] = res;
            
        }
        """).build()

        #d = measure.shape[0]/2
        #ms = measure[0:l*d]
        img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
        alphaIm_buf = cl.Buffer(ctx, mf.WRITE_ONLY, alphaIm.nbytes)
        sh = alphaIm.shape

        size = 8 # Window size
        prg.measure(queue, sh, None, alphaIm_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(size))
        cl.enqueue_read_buffer(queue,alphaIm_buf,alphaIm).wait()

        maxim = np.max(alphaIm)
        minim = np.min(alphaIm)
        #print maxim, minim

        import matplotlib
        from matplotlib import pyplot as plt
        # Alpha image
        #plt.imshow(alphaIm, cmap=matplotlib.cm.gray)
        #plt.show()
        #return

        paso = (maxim-minim)/cuantas
        if(paso <= 0):
            # the alpha image is monofractal
            clases = np.array(map(lambda i: i+minim,np.zeros(cuantas))).astype(np.float32)
        else:
            clases = np.arange(minim,maxim,paso).astype(np.float32)


        # Window
        cant = int(np.floor(np.log(Nx)))

        # concatenate the image A as [[A,A],[A,A]]
        hs = np.hstack((alphaIm,alphaIm))
        alphaIm = np.vstack((hs,hs))

        prg = cl.Program(ctx, """
            __kernel void krnl(__global int *flag, __global float *clases, 
                               __global float* alphaIm,const int sizeBlocks, const int Ny,
                                const int numBlocks_y, const int c, const int cuantas,
                                float minim, float maxim) {
                int i = get_global_id(0);
                int j = get_global_id(1);
                int xi = i*sizeBlocks;
                int xf = (i+1)*sizeBlocks-1;
                int yi = j*sizeBlocks;
                int yf = (j+1)*sizeBlocks-1;

                // calculate max and min for this subregion
                float maxx;
                float minn;
                int w,t;
                int first = 0;
                for(w = xi; w < xf; w++) {
                    for(t = yi; t < yf; t++) {
                        float v = alphaIm[w*Ny*2 + t];
                        if (v >= clases[c] and v <= clases[c+1]) {
                            if(!first) { first = 1; maxx = minn = v; }
                            if(v > maxx) maxx = v;
                            if(v < minn) minn = v;
                        }
                    }
                }
                float totalDif = maxim - minim;
                int nB = numBlocks_y; // num of subdivisions in the Z coordinate
                int l = floor(((maxx-minim)/totalDif)*nB)+1;
                int k = floor(((minn-minim)/totalDif)*nB)+1;
                flag[i*numBlocks_y + j] = l-k+1;
            }
        """).build()  

        # Multifractal dimentions
        falpha = np.zeros(cuantas).astype(np.float32)
        clases_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=clases.astype(np.float32))
        alphaIm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alphaIm.astype(np.float32))
        for c in range(cuantas):
            N = np.zeros(cant+1)
            # window sizes
            for k in range(1,cant+2):
                sizeBlocks = 2*k+1
                numBlocks_x = int(np.floor(Nx/sizeBlocks))
                numBlocks_y = int(np.floor(Ny/sizeBlocks))

                flag = np.zeros((numBlocks_x,numBlocks_y)).astype(np.int32)
                flag_buf = cl.Buffer(ctx, mf.WRITE_ONLY, flag.nbytes)
                sh = flag.shape            

                prg.krnl(queue, sh, None, flag_buf, clases_buf, alphaIm_buf, np.int32(sizeBlocks), np.int32(Ny), np.int32(numBlocks_y), np.int32(c), np.int32(cuantas), np.float32(minim), np.float32(maxim))
                cl.enqueue_read_buffer(queue, flag_buf, flag).wait()
                N[k-1] = cla.sum(cla.to_device(queue,flag)).get()

            #print N
            # Haussdorf (box) dimention of the alpha distribution
            falpha[c] = -np.polyfit(map(lambda i: np.log((2*i+1)),range(1,cant+2)),np.log(map(lambda i: i+1,N)),1)[0]
        s = np.hstack((clases,falpha))
        return s
        #return falpha
