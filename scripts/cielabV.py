import Image
import colortransforms as ct
import numpy as np
import singularityCL

def lab(filename,extra):

    I = Image.open(filename)
   
    L = ct.rgb_to_cielab_i_X(I,0)
    L = singularityCL.spec(L,extra)
    if(extra[3]):
        a = ct.rgb_to_cielab_i_X(I,1)
        b = ct.rgb_to_cielab_i_X(I,2)
        a = singularityCL.spec(a,extra)
        b = singularityCL.spec(b,extra)
        return np.vstack((L,a,b))
    return np.array(L)


# show the image
#import matplotlib
#from matplotlib import pyplot as plt

#plt.imshow(L, cmap=matplotlib.cm.gray)
#plt.show()
