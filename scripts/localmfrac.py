import numpy as np
import singularityCL as sing
import Image

def inner_localMF(I, N):
   if(N==0):
      return sing.spec(I,[20,False,True])

   w,h = I.size
   #print w, h

    # crop?
   return np.hstack(( inner_localMF(I.crop((0,0,w/2,h/2)), N-1), \
                   inner_localMF(I.crop((0,h/2,w/2,h)), N-1), \
                   inner_localMF(I.crop((w/2,0,w,h/2)), N-1), \
                   inner_localMF(I.crop((w/2,h/2,w,h)), N-1)  ))

def po2(x):
   y = 1
   while(2*y <= x):
      y=2*y
   return y

def localMF(I,N):
   # calculate minimum power of 2, less or equal than w and h
   I = Image.open(I)
   w,h = I.size
   siz = min(po2(w),po2(h))

   return inner_localMF(I.crop((0,0,siz,siz)),N)

#print localMF('../images/scanner/baguette/baguette1.tif',2)
