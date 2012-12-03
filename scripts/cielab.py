import Image
import colortransforms as ct
import numpy as np
import singularityCL

def lab(filename):

    I = Image.open(filename)

    # ?, num of FDs , convert to grayscale?
    extra = [40,False, False]

    L = ct.rgb_to_cielab_i_X(I,0)
    #a = ct.rgb_to_cielab_i_X(I,1)
    #b = ct.rgb_to_cielab_i_X(I,2)
    L = singularityCL.spec(L,extra)
    #a = singularityCL.spec(a,extra)
    #b = singularityCL.spec(b,extra)
    #return np.hstack((L,a,b))
    return np.array(L)


# show the image
#import matplotlib
#from matplotlib import pyplot as plt

#plt.imshow(L, cmap=matplotlib.cm.gray)
#plt.show()
