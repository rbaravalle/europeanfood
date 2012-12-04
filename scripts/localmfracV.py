import numpy as np
import singularityCL as sg
import Image
import colortransforms as ct

def inner_localMF(I, N, extra):
   if(N==0):
      return sg.spec(I,extra)

   w,h = I.shape

    # crop?
   return np.vstack(( inner_localMF(I[0:w/2,0:h/2], N-1, extra), \
                   inner_localMF(I[0:w/2,h/2:h], N-1, extra), \
                   inner_localMF(I[w/2:w,0:h/2], N-1, extra), \
                   inner_localMF(I[w/2:w,h/2:w], N-1, extra)  ))

def po2(x):
   y = 1
   while(2*y <= x):
      y=2*y
   return y

def localMF(I,N,extra):
   # calculate minimum power of 2, less or equal than w and h
   I = Image.open(I)
   w,h = I.size
   siz = min(po2(w),po2(h))

   I = I.crop((0,0,siz,siz))
   cielab = extra[5]
   channel = extra[6]
   if(cielab):
       I = ct.rgb_to_cielab_i_X(I,channel)
   else:    
       #print np.array(I.getdata()).shape
       I = np.array(I.convert('L').getdata()).reshape((siz,siz))

   return inner_localMF(I,N,extra)

#print localMF('../images/scanner/baguette/baguette1.tif',2)
