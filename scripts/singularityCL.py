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
        if(extra[1]==True):
            a = Image.open(filename)
            Nx, Ny = a.size
        else:
            a = filename
            Nx, Ny = a.shape
        L = Nx*Ny

        points = []     # number of elements in the structure
        if(extra[2] == True):
            gray = a.convert('L') # rgb 2 gray
            arr = np.array(gray.getdata()).astype(np.int32)
        else:
            arr = np.array(a).reshape(a.shape[0]*a.shape[1])

        alphaIm = np.zeros((Nx,Ny), dtype=np.double ) # Nx rows x Ny columns

        l = 4 # (maximum window size-1) / 2
        temp = map(lambda i: 2*i+1, range(l))
        temp = np.log(temp)
        measure = np.zeros(l*Ny).astype(np.int32)

        b = np.vstack((temp,np.ones((1,l)))).T
        AA=coo_matrix(np.kron(np.identity(Nx), b))     


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
            cl.enqueue_read_buffer(queue, dest_buf, measure[0:l*Ny/2]).wait()
            d = Ny/2
            prg.measure(queue, sh, None, dest_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(l), np.int32(i), np.int32(d))
            cl.enqueue_read_buffer(queue, dest_buf, measure[l*Ny/2:]).wait()

            # Instead of doing polyfits, a sparse linear system is constructed and solved

            bb=np.log(measure)
            z = linsolve.lsqr(AA,bb)[0]
            z = z.reshape(2,Ny,order = 'F')
            alphaIm[i] = z[0]

        maxim = np.max(alphaIm)
        minim = np.min(alphaIm)

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
                                const int numBlocks_y, const int c, const int cuantas) {
                int i = get_global_id(0);
                int j = get_global_id(1);
                int xi = i*sizeBlocks;
                int xf = (i+1)*sizeBlocks-1;
                int yi = j*sizeBlocks;
                int yf = (j+1)*sizeBlocks-1;
                if(xf == xi) xf = xf+1;
                if(yf == yi) yf = yf+1;

                int f = 0;
                int s1 = xf-xi;
                int s2 = yf-yi;
                
                if(c != cuantas-1) {
                    // f = 1 if any pixel in block is between clases[c] and clases[c+1]
                    int w, t;
                    for(w = xi; w < xf; w++) {
                        for(t = yi; t < yf; t++) {
                            float b = alphaIm[w*Ny*2 + t];
                            if (b >= clases[c] and b < clases[c+1]) {
                               f = 1;
                               break;
                            }
                        }
                        if(f == 1) break;
                    }
                }
                else {
                    // f = 1 if any pixel in block is equal to classes[c]
                    int w, t;
                    for(w = xi; w < xf; w++) {
                        for(t = yi; t < yf; t++) {
                            float b = alphaIm[w*Ny*2 + t];
                            if (b == clases[c]) { // !!
                               f = 1;
                               break;
                            }
                        }
                        if(f == 1)
                            break;
                    }
                }
                flag[i*numBlocks_y + j] = f;
            }
        """).build()  

        # Multifractal dimentions
        falpha = np.zeros(cuantas).astype(np.float32)
        clases_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=clases.astype(np.float32))
        alphaIm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alphaIm.astype(np.float32))
        for c in range(cuantas):
            N = np.zeros(cant+1)
            # window sizes
            for k in range(cant+1):
                sizeBlocks = 2*k+1
                numBlocks_x = int(np.ceil(Nx/sizeBlocks))
                numBlocks_y = int(np.ceil(Ny/sizeBlocks))

                flag = np.zeros((numBlocks_x,numBlocks_y)).astype(np.int32)
                flag_buf = cl.Buffer(ctx, mf.WRITE_ONLY, flag.nbytes)
                sh = flag.shape            

                prg.krnl(queue, sh, None, flag_buf, clases_buf, alphaIm_buf, np.int32(sizeBlocks), np.int32(Ny), np.int32(numBlocks_y), np.int32(c), np.int32(cuantas))
                cl.enqueue_read_buffer(queue, flag_buf, flag).wait()
                N[k] = cla.sum(cla.to_device(queue,flag)).get()

            # Haussdorf (box) dimention of the alpha distribution
            falpha[c] = -np.polyfit(map(lambda i: np.log(i*2+1),range(cant+1)),np.log(map(lambda i: i+1,N)),1)[0]
        s = np.hstack((clases,falpha))
        return s
        #return falpha
