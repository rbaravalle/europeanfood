import Image
import colortransforms as ct
import numpy as np
import mfs
#import singularityCL

def lab(filename,extra):

    I = Image.open(filename)
   
    L = ct.rgb_to_cielab_i_X(I,0)
    L = mfs.mfs(L,extra)
    #L = singularityCL.spec(L,extra)
    if(extra[4]):
        a = ct.rgb_to_cielab_i_X(I,1)
        b = ct.rgb_to_cielab_i_X(I,2)
        a = mfs.mfs(a,extra)
        b = mfs.mfs(b,extra)
        #a = singularityCL.spec(a,extra)
        #b = singularityCL.spec(b,extra)
        return np.hstack((L,a,b))
    return np.array(L)


# show the image
#import matplotlib
#from matplotlib import pyplot as plt

#plt.imshow(L, cmap=matplotlib.cm.gray)
#plt.show()
