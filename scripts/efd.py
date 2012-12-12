import Image
import numpy as np
from math import exp, log10
import scipy.ndimage.filters as sf
import matplotlib
from matplotlib import pyplot as plt
import scipy.signal
import pyopencl as cl
import mfs

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def efd(im,extra):

    cantFD = extra[1]

    if(extra[3]):
        a = Image.open(im)
    else:
        a = im

    Nx, Ny = a.size
    L = Nx*Ny

    gray = a.convert('L') # rgb 2 gray
    arr = np.array(gray.getdata()).astype(np.int32)

    alphaIm = np.zeros((Nx,Ny), dtype=np.float32 ) # Nx rows x Ny columns

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
             float v = hist[s]/total+1; // probability
             res += v*log2(v);
         }

         alphaIm[i*Ny+j] = -res;
        
    }
    """).build()

    img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
    alphaIm_buf = cl.Buffer(ctx, mf.WRITE_ONLY, alphaIm.nbytes)
    sh = alphaIm.shape

    size = 16 # Window size
    prg.measure(queue, sh, None, alphaIm_buf, img_buf, np.int32(Nx), np.int32(Ny), np.int32(size))
    cl.enqueue_read_buffer(queue,alphaIm_buf,alphaIm).wait()

    min_Im = np.min(alphaIm)
    max_Im = np.max(alphaIm)
    alphaIm = 255*(alphaIm-min_Im)/(max_Im - min_Im)
   
    res = mfs.mfs(alphaIm,[1,cantFD,3,False])
    return res
