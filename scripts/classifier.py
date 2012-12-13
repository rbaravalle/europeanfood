import csv
import sys
import os
from subprocess import *
from gch import colorHistogram
#import pyccv
import colortransforms
import numpy as np
from featureTexture import *
import singularityCL as sg
import mfs
import localmfrac
import time
import cielab
import laplacian
import mfs
import singularityMFS

cantDF = 30

def callF(filename,which,extra):
    return features(filename,which,3,False,extra)

def features(filename,i,j,combine,extra):
    #farr = [colorHistogram,ccv,haralick,lbp,tas,zernike, singularity.spec]
    #farr = [localmfrac.localMF]
    #farr = [cielab.lab]
    #farr = [mfs.mfs]
    #farr = [sg.spec]
    #farr = [mfs.mfs, laplacian.laplacian]
    farr = [singularityMFS.spec]
    if(combine==True):
        return hstack((farr[1](filename),farr[2](filename),farr[3](filename),farr[4](filename),farr[5](filename)))
    t =  time.clock()
    # num of FDs , Open Image?,  convert to grayscale?, (cielab) use L,a,b?
    extra = [1,cantDF*2,3,True]
    res = farr[0](filename,extra)
    #res2 = farr[1](filename,extra)
    t =  time.clock() - t
    #print "Time: ", t
    return res
    #return np.hstack([res,res2])


def ccv(filename):
    import Image
    I = Image.open(filename)
    I.putdata(colortransforms.rgb_to_cielab_i(I))
    size = I.size[0] # Normalize image size
    threshold = 500
    return pyccv.calc_ccv(I,size,threshold)

def conf_mat(test, classes):
    m = [ [0 for i in range(max(classes))] for j in range(max(classes))]
    for i in range(len(classes)):
        m[test[i]-1][classes[i]-1] = m[test[i]-1][classes[i]-1] + 1
    return m

# get cross validation of executing ./easy.py ../exps/gchS.txt
def getCross(fileStxt):

    easy = './easy.py'

    cmd = '{0} {1}'.format(easy, fileStxt)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout

    line = 1
    while True:
        last_line = line
        line = f.readline()
        if str(line).find("rate") != -1:        
            cross = float(line.split()[-1][5:8])
            break
        if not line: break
    return cross


def csvm(dtrain,dtest,labels,fileStxt, fileCtxt, base):
    arch = base+'.predict'

    easy = './easy.py'

    cmd = '{0} {1} {2}'.format(easy, fileStxt, fileCtxt)
    print cmd
    f = Popen(cmd, shell = True).communicate()
   
    cmd = 'cat "{0}"'.format(arch)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout
    g = f

    c = 0
    line = 1
    while True:
        last_line = line
        line = f.readline()
        c = c+1
        if not line: break
    testL = [0 for i in range(c)]

    cmd = 'cat "{0}"'.format(arch)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout
    c = 0
    line = 1
    while True:
        last_line = line
        line = f.readline()
        testL[c] = int(last_line)
        c = c+1
        if not line: break

    return testL

from sklearn.neighbors import KNeighborsClassifier
def cnearestneighbors(dtrain,dtest,labels,fileStxt, fileCtxt, base):
    cnn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=True)

    cnn.fit(dtrain,labels)
    print "Nearest Neighbors: ", cnn.score(dtest,labels)
    return np.array(cnn.predict(dtest)).astype(np.int32)


def crandomforest(dtrain,dtest,labels,fileStxt, fileCtxt, base):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=60)
    clf.fit(dtrain,labels)
    print "Random Forest: ", clf.score(dtest,labels)
    return np.array(clf.predict(dtest)).astype(np.int32)


def classifierPredict(dtrain,dtest,labels,fileStxt, fileCtxt, base, i):
    pred = [csvm, crandomforest,cnearestneighbors]
    return pred[i](dtrain,dtest,labels,fileStxt, fileCtxt, base)


def test(dtrain,dtest,labels,fileStxt, fileCtxt, base):

    testL = classifierPredict(dtrain,dtest,labels,fileStxt, fileCtxt, base,0)

    print "Test 0: ", testL

    b = conf_mat(testL,labels)
    for row in b:
        print row

    testL = classifierPredict(dtrain,dtest,labels,fileStxt, fileCtxt, base,1)

    print "Test 1: ", testL

    b = conf_mat(testL,labels)
    for row in b:
        print row

    testL = classifierPredict(dtrain,dtest,labels,fileStxt, fileCtxt, base,2)

    print "Test 2: ", testL

    b = conf_mat(testL,labels)
    for row in b:
        print row


    return testL



def main(subname,which,local):
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    fileCtxt = '../exps/'+base
    fileCcsv = '../exps/'+subname+'C.csv'
    cant = 20+1
    dDFs  = 64
    baguette = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvado   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactal   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwich = [['Df' for j in range(dDFs)] for i in range(cant)]

    baguetteC = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvadoC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactalC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwichC = [['Df' for j in range(dDFs)] for i in range(cant)]

    path = '../images/nonbread/res/'
    dirList=os.listdir(path)
    #print len(dirList)
    
    nonbread = [['Df' for j in range(dDFs)] for i in range(len(dirList))]


    if(local == True):
        trainingData, testingData = localFeatures(subname,which)
        with open(fileScsv, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(trainingData)

        with open(fileCcsv, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(testingData)

        prog = './a.out' # convert.c
        cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        cmd = '{0} "{1}" > "{2}"'.format(prog, fileCcsv, fileCtxt)
        Popen(cmd, shell = True, stdout = PIPE).communicate()
        labels = [i for i in range(len(testingData))]
        labels = map(lambda i: i/20+1, labels)
        test(trainingData, testingData, labels, fileStxt, fileCtxt, base)
        return

    else:    # else global
        cantDF = 40
        import Image
        j = 0
        extra = [cantDF,40,1.15]
        for i in range(len(dirList)):
            filename = path+dirList[i]
            I = Image.open(filename)
            if(I.mode == 'RGB'):
                print filename
                nonbread[j] = callF(filename,which,extra)
                j = j+1
            if (j > cant*2+1):
                break


        for i in range(1,cant):
            extra = [cantDF,40,1.2]
            filename = '../images/scanner/baguette/baguette{}.tif'.format(i)
            print filename
            baguette[i] = callF(filename,which,extra)
            filename = '../images/scanner/lactal/lactal{}.tif'.format(i)
            print filename
            lactal[i] = callF(filename,which,extra)
            filename = '../images/scanner/salvado/salvado{}.tif'.format(i)
            print filename
            salvado[i] = callF(filename,which,extra)
            filename = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
            print filename
            sandwich[i] = callF(filename,which,extra)


            extra = [cantDF,50,1.05]
            filename = '../images/camera/baguette/slicer/b{}.tif'.format(i)
            print filename
            baguetteC[i] = callF(filename,which,extra)
            filename = '../images/camera/lactal/l{}.tif'.format(i)
            print filename
            lactalC[i] = callF(filename,which,extra)
            filename = '../images/camera/salvado/s{}.tif'.format(i)
            print filename
            salvadoC[i] = callF(filename,which,extra)
            filename = '../images/camera/sandwich/s{}.tif'.format(i)
            print filename
            sandwichC[i] = callF(filename,which,extra)
   

    trainingData = baguette[1:]+lactal[1:]+salvado[1:]+sandwich[1:]+nonbread[0:cant-1]
    testingData = baguetteC[1:]+lactalC[1:]+salvadoC[1:]+sandwichC[1:]+nonbread[cant-1:2*(cant-1)]
    with open(fileScsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(trainingData)

    with open(fileCcsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(testingData)


    prog = './a.out' # convert.c
    cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
    Popen(cmd, shell = True, stdout = PIPE).communicate()	
    cmd = '{0} "{1}" > "{2}"'.format(prog, fileCcsv, fileCtxt)
    Popen(cmd, shell = True, stdout = PIPE).communicate()
    labels = [i for i in range(len(testingData))]
    labels = map(lambda i: i/20+1, labels)
    test(trainingData, testingData, labels,fileStxt, fileCtxt, base)


# SIFT BoW
from os.path import exists, isdir, basename, join, splitext
import sift
#from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
PRE_ALLOCATION_BUFFER = 1000  # for sift
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'
DATASETPATH = 'dataset/'
SIZE_LOCAL_FEATURE = 128 # 64: SURF, 128: SIFT, DF*2: MF
from cPickle import dump, HIGHEST_PROTOCOL, load
from mahotas.features import surf
from singularityBoF import featuresMF

# only one file
def extractSift(filename):
    features_fname = filename + '.sift'
    sift.process_image(filename, features_fname)
    locs, descriptors = sift.read_features_from_file(features_fname)
    return descriptors

# perform a multifractal spectrum in the first M points detected by sift
def extractMF(filename):
    features_fname = filename + '.sift'
    sift.process_image(filename, features_fname)
    locs, descriptors = sift.read_features_from_file(features_fname)
    sh = locs.shape[0]
    res = np.zeros((sh,SIZE_LOCAL_FEATURE)).astype(np.float32)
    extra = [20,False,True,False,0,0,0]
    WIN = 5
    for i in range(sh):
        x = np.int32(round(locs[i][0]))
        y = np.int32(round(locs[i][1]))
        I = Image.open(filename)
        Nx,Ny = I.size
        a = sg.spec(I.crop((max(x-WIN,0),max(y-WIN,0),min(x+WIN,Nx-1),min(y+WIN,Ny-1))),extra)
        res[i] = a
    print res.shape
    return res

from mahotas import surf
def extractSURF(filename):
    import Image
    I = Image.open(filename)
    f = np.array(I.convert('L').getdata()).reshape(I.size[0],I.size[1])
    f = f.astype(np.uint8)
    return surf.surf(f,descriptor_only=True)

def extractSurf(filename):
    features_fname = filename
    descriptors = surf.surf(np.array(Image.open(filename).convert('L')),descriptor_only=True)
    print "Shape: ", descriptors.shape
    return descriptors

def extractLocal(filename,i):
    f = [extractSift, extractSURF, extractMF] #featuresMF]
    return f[i](filename)

def dict2numpy(dict):
    nkeys = len(dict)
    print "100 no? : ", nkeys
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, SIZE_LOCAL_FEATURE))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0] # num of elements of "value"
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, SIZE_LOCAL_FEATURE)) # eliminate 0's rows in "array"
    print "Tamanio del array devuelto por dict2numpy?: ", array.shape
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


def localFeatures(subname,alg):
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    fileCtxt = '../exps/'+base
    fileCcsv = '../exps/'+subname+'C.csv'
    cant = 20+1
    dDFs  = 64
    baguette = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvado   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactal   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwich = [['Df' for j in range(dDFs)] for i in range(cant)]

    baguetteC = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvadoC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactalC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwichC = [['Df' for j in range(dDFs)] for i in range(cant)]

    path = '../images/nonbread/res/'
    dirList=os.listdir(path)
    print len(dirList)
    nonbread = [['Df' for j in range(dDFs)] for i in range(len(dirList))]
    import Image
    j = 0
    for i in range(len(dirList)):
        filename = path+dirList[i]
        I = Image.open(filename)
        if(I.mode == 'RGB'):
            print filename
            nonbread[j] = extractLocal(filename,alg)
            j = j+1
        if (j > (cant-1)*2+1):
            break

    for i in range(1,cant):
        filename = '../images/scanner/baguette/baguette{}.tif'.format(i)
        print filename
        baguette[i] = extractLocal(filename,alg)
        filename = '../images/scanner/lactal/lactal{}.tif'.format(i)
        print filename
        lactal[i] = extractLocal(filename,alg)
        filename = '../images/scanner/salvado/salvado{}.tif'.format(i)
        print filename
        salvado[i] = extractLocal(filename,alg)
        filename = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
        print filename
        sandwich[i] = extractLocal(filename,alg)

        filename = '../images/camera/baguette/slicer/b{}.tif'.format(i)
        print filename
        baguetteC[i] = extractLocal(filename,alg)
        filename = '../images/camera/lactal/l{}.tif'.format(i)
        print filename
        lactalC[i] = extractLocal(filename,alg)
        filename = '../images/camera/salvado/s{}.tif'.format(i)
        print filename
        salvadoC[i] = extractLocal(filename,alg)
        filename = '../images/camera/sandwich/s{}.tif'.format(i)
        print filename
        sandwichC[i] = extractLocal(filename,alg)


    # array of SIFT features
    all_featuresS = {}
    all_featuresC = {}

    arrS = baguette[1:]+lactal[1:]+salvado[1:]+sandwich[1:]+nonbread[0:cant-1]
    arrC = baguetteC[1:]+lactalC[1:]+salvadoC[1:]+sandwichC[1:]+nonbread[cant-1:(cant-1)*2]

    # to dict
    for d in range(len(arrS)):
        all_featuresS[d] = arrS[d]

    for d in range(len(arrS)):
        all_featuresC[d] = arrC[d]

    all_features_arrayS = dict2numpy(all_featuresS)
    all_features_arrayC = dict2numpy(all_featuresC)

    print "Kmeans"
    nfeatures = all_features_arrayS.shape[0]
    nclusters = int(sqrt(nfeatures))
    print "nfeatures: ", nfeatures, " , nclusters: ", nclusters

    # the codebook is made with the training SIFT descriptors
    codebook, distortion = vq.kmeans(all_features_arrayS,
                                             nclusters,
                                             thresh=K_THRESH)


    #with open(CODEBOOK_FILE, 'wb') as f:
    #    dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    #print "---------------------"
    #print "## loading codebook from " + CODEBOOK_FILE
    #with open(CODEBOOK_FILE, 'rb') as f:
    #    codebook = load(f)

    print "Codebook OK"
    all_word_histgramsS = [[0 for j in range(20)] for i in range(len(all_featuresS))]
    for imagefname in all_featuresS:
        word_histgram = computeHistograms(codebook, all_featuresS[imagefname])
        all_word_histgramsS[imagefname] = word_histgram

    all_word_histgramsC = [[0 for j in range(20)] for i in range(len(all_featuresC))]
    for imagefname in all_featuresC:
        word_histgram = computeHistograms(codebook, all_featuresC[imagefname])
        all_word_histgramsC[imagefname] = word_histgram
        #print "W", word_histgram

    return all_word_histgramsS, all_word_histgramsC


#main('gch',0,0)
#main('ccv',1,0)
#main('haralick',2,0)
#main('lbp',3,0)
#main('tas',4,0)
#main('zernike',5,0)
main('sift',0,True)
#main('singularity',6,False,SVM)
#main('xxx3lbp',6,False,RANDOM_FOREST)
# Multi Fractal Bag of Features
#main('difAbsCentralmfsRobustez60',2,False)

