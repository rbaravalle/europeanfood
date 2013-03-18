import csv
import sys
import os
from subprocess import *
from gch import colorHistogram
#import pyccv
import colortransforms
import numpy as np
from featureTexture import *
#import singularityEntropyCL as sg
#import localmfrac
import time
import cielab
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import mfs
#import efd
import gradient
import laplacian
#import singularityMFS as smfs
from matplotlib import pyplot as plt
from vf import VF
from mca import mca

SVM = 0
RANDOM_FOREST = 1
cantDF = 10

def callF(filename,which,extra):
    return features(filename,which,3,False,extra)

def features(filename,i,j,combine,extra):
    #farr = [colorHistogram,ccv,haralick,lbp,tas,zernike, singularity.spec]
    #farr = [localmfrac.localMF]
    farr = [cielab.lab]
    #farr = [lbp]
    #farr = [sg.spec]
    #farr = [smfs.spec]
    #farr = [mfs.mfs]
    #farr = [mfs.mfs, gradient.main]
    #farr = [mfs.mfs, gradient.main, laplacian.laplacian]
    #farr = [mfs.mfs, laplacian.laplacian]
    #farr = [mfs.mfs]
    #farr = [gradient.main]
    #farr = [efd.efd]
    #farr = [haralick]
    #farr = [lbp]
    #if(combine==True):
    #    return hstack((farr[1](filename),farr[2](filename),farr[3](filename),farr[4](filename),farr[5](filename)))
    t =  time.clock()
    # num of FDs , Open Image?,  convert to grayscale?, (cielab) use L,a,b?
    cantDF = 10
    extra = [1,cantDF*2,3,False,True]
    res = farr[0](filename, extra)
    #res = farr[0](filename,extra)
    #res2 = farr[1](filename,extra)
    #res3 = farr[2](filename,extra)
    t =  time.clock() - t
    #print "Time: ", t
    return res
    #return np.hstack([res,res2])
    #return np.hstack([res,res2,res3])


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


def csvm(dtrain,labels,fileStxt, base):
    arch = base+'.predict'

    easy = './easy.py'

    cmd = '{0} {1}'.format(easy, fileStxt)
    print cmd
    f = Popen(cmd, shell = True).communicate()
   
    
    #return testL

from sklearn.neighbors import KNeighborsClassifier
def cnearestneighbors(data,labels,fileStxt,base):
    cnn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=True)

    scores = cross_validation.cross_val_score(cnn, data, labels, cv=4)
    print scores
    print "Nearest Neighbors: " + str( np.array(scores).mean() )

def crandomforest(data,labels,fileStxt,base):
    #read in data, parse into training and target sets
    #dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    #In this case we'll use a random forest, but this could be any classifier
    cfr = RandomForestClassifier(n_estimators=100)

    #Simple K-Fold cross validation. 5 folds.
    #cv = cross_validation.KFold(len(data), k=4, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    #data = np.array(data)
    #labels=np.array(labels)
    scores = cross_validation.cross_val_score(cfr, data, labels, cv=4)
    print scores
#    for traincv, testcv in cv:
#        print "T1, T2", traincv, testcv
#        probas = cfr.fit(data[traincv], labels[traincv]).predict_proba(data[testcv])
#        results.append( logloss.llfun(labels[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Random Forest: " + str( np.array(scores).mean() )


def classifierPredict(dtrain,labels,fileStxt, base, i):
    pred = [csvm, crandomforest, cnearestneighbors]
    return pred[i](dtrain,labels,fileStxt, base)


def test(dtrain,labels,fileStxt, base, classifier):
    testL = classifierPredict(dtrain,labels,fileStxt, base,0)
    testL = classifierPredict(dtrain,labels,fileStxt, base,1)
    testL = classifierPredict(dtrain,labels,fileStxt, base,2)



def main(subname,which,local,classifier):
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    #fileCtxt = '../exps/'+base
    #fileCcsv = '../exps/'+subname+'C.csv'
    cant = 20+1
    dDFs  = 64
    baguette = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    salvado   = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    lactal   = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    sandwich = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]

    baguetteC = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    salvadoC   = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    lactalC   = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]
    sandwichC = [[np.float32(0) for j in range(dDFs)] for i in range(cant)]

    vfBaguette = [np.float32(0) for i in range(cant)]
    vfBaguetteC = [np.float32(0) for i in range(cant)]
    mcaBaguette = [np.float32(0) for i in range(cant)]
    mcaBaguetteC = [np.float32(0) for i in range(cant)]
    mcadvBaguette = [np.float32(0) for i in range(cant)]
    mcadvBaguetteC = [np.float32(0) for i in range(cant)]

    vfLactal = [np.float32(0) for i in range(cant)]
    vfLactalC = [np.float32(0) for i in range(cant)]
    mcaLactal = [np.float32(0) for i in range(cant)]
    mcaLactalC = [np.float32(0) for i in range(cant)]
    mcadvLactal = [np.float32(0) for i in range(cant)]
    mcadvLactalC = [np.float32(0) for i in range(cant)]

    vfSalvado = [np.float32(0) for i in range(cant)]
    vfSalvadoC = [np.float32(0) for i in range(cant)]
    mcaSalvado = [np.float32(0) for i in range(cant)]
    mcaSalvadoC = [np.float32(0) for i in range(cant)]
    mcadvSalvado = [np.float32(0) for i in range(cant)]
    mcadvSalvadoC = [np.float32(0) for i in range(cant)]

    vfSandwich = [np.float32(0) for i in range(cant)]
    vfSandwichC = [np.float32(0) for i in range(cant)]
    mcaSandwich = [np.float32(0) for i in range(cant)]
    mcaSandwichC = [np.float32(0) for i in range(cant)]
    mcadvSandwich = [np.float32(0) for i in range(cant)]
    mcadvSandwichC = [np.float32(0) for i in range(cant)]

    path = '../images/nonbread/res/'
    dirList=os.listdir(path)
    #print len(dirList)
    
    nonbread = [['Df' for j in range(dDFs)] for i in range(len(dirList))]

    if(local == True):
        data = localFeatures(subname,which)
        with open(fileScsv, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        prog = './clas2' # convert.c
        cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        labels = [i for i in range(len(data))]
        labels = map(lambda i: i/(2*(cant-1))+1, labels)
        print labels
        test(data, labels, fileStxt, base,classifier)
        return

    else:    # else global
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
            extra = [cantDF,40,1.15]
            filename = '../images/scanner/baguette/baguette{}.tif'.format(i)
            print filename
            baguette[i] = callF(filename,which,extra)
            vfBaguette[i] = VF(filename,extra)
            mcaBaguette[i] = mca(filename,extra)[0]
            mcadvBaguette[i] = mca(filename,extra)[1]
            filename = '../images/scanner/lactal/lactal{}.tif'.format(i)
            print filename
            lactal[i] = callF(filename,which,extra)
            vfLactal[i] = VF(filename,extra)
            mcaLactal[i] = mca(filename,extra)[0]
            mcadvLactal[i] = mca(filename,extra)[1]
            filename = '../images/scanner/salvado/salvado{}.tif'.format(i)
            print filename
            salvado[i] = callF(filename,which,extra)
            vfSalvado[i] = VF(filename,extra)
            mcaSalvado[i] = mca(filename,extra)[0]
            mcadvSalvado[i] = mca(filename,extra)[1]
            filename = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
            print filename
            sandwich[i] = callF(filename,which,extra)
            vfSandwich[i] = VF(filename,extra)
            mcaSandwich[i] = mca(filename,extra)[0]
            mcadvSandwich[i] = mca(filename,extra)[1]

            extra = [cantDF,50,1]
            filename = '../images/camera/baguette/slicer/b{}.tif'.format(i)
            print filename
            baguetteC[i] = callF(filename,which,extra)
            vfBaguetteC[i] = VF(filename,extra)
            mcaBaguetteC[i] = mca(filename,extra)[0]
            mcadvBaguetteC[i] = mca(filename,extra)[1]
            filename = '../images/camera/lactal/l{}.tif'.format(i)
            print filename
            lactalC[i] = callF(filename,which,extra)
            vfLactalC[i] = VF(filename,extra)
            mcaLactalC[i] = mca(filename,extra)[0]
            mcadvLactalC[i] = mca(filename,extra)[1]
            filename = '../images/camera/salvado/s{}.tif'.format(i)
            print filename
            salvadoC[i] = callF(filename,which,extra)
            vfSalvadoC[i] = VF(filename,extra)
            mcaSalvadoC[i] = mca(filename,extra)[0]
            mcadvSalvadoC[i] = mca(filename,extra)[1]
            filename = '../images/camera/sandwich/s{}.tif'.format(i)
            print filename
            sandwichC[i] = callF(filename,which,extra)
            vfSandwichC[i] = VF(filename,extra)
            mcaSandwichC[i] = mca(filename,extra)[0]
            mcadvSandwichC[i] = mca(filename,extra)[1]

    data = baguette[1:]+ baguetteC[1:]+lactal[1:]+lactalC[1:]+salvado[1:]+salvadoC[1:]+sandwich[1:]+sandwichC[1:]+nonbread[0:(2*(cant-1))]


    dataB = np.array(baguette[1:]+ baguetteC[1:])
    dataL = np.array(lactal[1:]+lactalC[1:])
    dataSal = np.array(salvado[1:]+salvadoC[1:])
    dataSan = np.array(sandwich[1:]+sandwichC[1:])

    # Void Fraction
    arrB = np.array(vfBaguette[1:]+vfBaguetteC[1:])
    arrL = np.array(vfLactal[1:]+vfLactalC[1:])
    arrSal = np.array(vfSalvado[1:]+vfSalvadoC[1:])
    arrSan = np.array(vfSandwich[1:]+vfSandwichC[1:])

    # Mean cell area
    mcaB = np.array(mcaBaguette[1:]+mcaBaguetteC[1:])
    mcaL = np.array(mcaLactal[1:]+mcaLactalC[1:])
    mcaSal = np.array(mcaSalvado[1:]+mcaSalvadoC[1:])
    mcaSan = np.array(mcaSandwich[1:]+mcaSandwichC[1:])

    # Mean cell area
    mcadvB = np.array(mcadvBaguette[1:]+mcadvBaguetteC[1:])
    mcadvL = np.array(mcadvLactal[1:]+mcadvLactalC[1:])
    mcadvSal = np.array(mcadvSalvado[1:]+mcadvSalvadoC[1:])
    mcadvSan = np.array(mcadvSandwich[1:]+mcadvSandwichC[1:])

    # correlation coefficients
    cfeat = len(dataB[0])
    print cfeat
    cB = np.zeros(cfeat)
    cL = np.zeros(cfeat)
    cSal = np.zeros(cfeat)
    cSan = np.zeros(cfeat)
    cmcaB = np.zeros(cfeat)
    cmcaL = np.zeros(cfeat)
    cmcaSal = np.zeros(cfeat)
    cmcaSan = np.zeros(cfeat)
    cmcadvB = np.zeros(cfeat)
    cmcadvL = np.zeros(cfeat)
    cmcadvSal = np.zeros(cfeat)
    cmcadvSan = np.zeros(cfeat)
    print "Shapes:"
    print cB.shape
    print dataB.shape
    for i in range(cfeat):
        print i
        cB[i] = np.corrcoef(dataB[:,i],arrB)[0,1]
        cmcaB[i] = np.corrcoef(dataB[:,i],mcaB)[0,1]
        cmcadvB[i] = np.corrcoef(dataB[:,i],mcadvB)[0,1]
        cL[i] = np.corrcoef(dataL[:,i],arrL)[0,1]
        cmcaL[i] = np.corrcoef(dataL[:,i],mcaL)[0,1]
        cmcadvL[i] = np.corrcoef(dataL[:,i],mcadvL)[0,1]
        cSal[i] = np.corrcoef(dataSal[:,i],arrSal)[0,1]
        cmcaSal[i] = np.corrcoef(dataSal[:,i],mcaSal)[0,1]
        cmcadvSal[i] = np.corrcoef(dataSal[:,i],mcadvSal)[0,1]
        cSan[i] = np.corrcoef(dataSan[:,i],arrSan)[0,1]
        cmcaSan[i] = np.corrcoef(dataSan[:,i],mcaSan)[0,1]
        cmcadvSan[i] = np.corrcoef(dataSan[:,i],mcadvSan)[0,1]


    #print "LO IMPORTANTE"
    #print "FD 0: Baguette"
    #print dataB[:,0]
    #print "Void Fraction Baguette"
    #print arrB
    #x = np.arange(arrB.shape[0])
    #plt.plot(x, dataB[:,0],'r',x,arrB)
    #plt.show()

    print "Coefficients"
    print "VF Baguette"
    print cB
    print "MCA Baguette"
    print cmcaB
    print "stdev MCA Baguette"
    print cmcadvB
    print "VF Lactal"
    print cL
    print "MCA Lactal"
    print cmcaL
    print "stdev MCA Lactal"
    print cmcadvL
    print "VF Salvado"
    print cSal
    print "MCA Salvado"
    print cmcaSal
    print "stdev MCA Salvado"
    print cmcadvSal
    print "VF Sandwich"
    print cSan
    print "MCA Sandwich"
    print cmcaSan
    print "stdev MCA Sandwich"
    print cmcadvSan

    x = np.arange(cfeat)
    plt.plot(x, cB,'b',x, cL,'r',x,cSal,'g',x,cSan)
    plt.show()

    mean = np.zeros((4,cfeat))
    std = np.zeros((4,cfeat))

    dataB = np.array(baguette[1:]+ baguetteC[1:])
    mean[0] = dataB.mean(axis=0)
    std[0] = dataB.std(axis=0)

    dataL = np.array(lactal[1:]+lactalC[1:])
    mean[1] = dataL.mean(axis=0)
    std[1] = dataL.std(axis=0)

    dataSal = np.array(salvado[1:]+salvadoC[1:])
    mean[2] = dataSal.mean(axis=0)
    std[2] = dataSal.std(axis=0)

    dataSan = np.array(sandwich[1:]+sandwichC[1:])
    mean[3] = dataSan.mean(axis=0)
    std[3] = dataSan.std(axis=0)

    x = np.arange(cfeat)
    #plt.plot(x, mean[0],'b',x, mean[1],'r',x,mean[2],'g',x,mean[3])
    #plt.show()

    print "Mean : ", mean[0]
    print "Std : ", std[0]
    print "Mean : ", mean[1]
    print "Std : ", std[1]
    print "Mean : ", mean[2]
    print "Std : ", std[2]
    print "Mean : ", mean[3]
    print "Std : ", std[3]
    with open('means.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(mean)

    with open('stds.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(std)


    print "200?: ", len(data)
    with open(fileScsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    prog = './clas2' # convert.c
    cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
    Popen(cmd, shell = True, stdout = PIPE).communicate()
    labels = [i for i in range(len(data))]
    labels = map(lambda i: i/(2*(cant-1))+1, labels)
    test(data, labels, fileStxt, base,classifier)


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
#from singularityBoF import featuresMF

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
    sh = min(locs.shape[0], 1000)
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

    arrS = baguette[1:]+ baguetteC[1:]+lactal[1:]+lactalC[1:]+salvado[1:]+salvadoC[1:]+sandwich[1:]+sandwichC[1:]+nonbread[0:(2*(cant-1))]
    # to dict
    for d in range(len(arrS)):
        all_featuresS[d] = arrS[d]

    #for d in range(len(arrS)):
    #    all_featuresC[d] = arrC[d]

    all_features_arrayS = dict2numpy(all_featuresS)
    #all_features_arrayC = dict2numpy(all_featuresC)

    print "Kmeans"
    nfeatures = all_features_arrayS.shape[0]
    nclusters = int(sqrt(nfeatures))
    print "nfeatures: ", nfeatures, " , nclusters: ", nclusters

    # the codebook is made with the training SIFT descriptors
    codebook, distortion = vq.kmeans(all_features_arrayS,
                                             nclusters,
                                             thresh=K_THRESH)

    print "Codebook OK"
    all_word_histgramsS = [[0 for j in range(20)] for i in range(len(all_featuresS))]
    for imagefname in all_featuresS:
        word_histgram = computeHistograms(codebook, all_featuresS[imagefname])
        all_word_histgramsS[imagefname] = word_histgram

    return all_word_histgramsS


#def coefCorr():
    #x = columna
    #y = void fraction
    

#main('gch',0,0)
#main('ccv',1,0)
#main('haralick',2,0)
#main('lbp',3,0)
#main('tas',4,0)
#main('zernike',5,0)
#main('surf',1,True, RANDOM_FOREST)
#main('singularity',6,False,SVM)
#main('sift2',0,True,RANDOM_FOREST)
# Multi Fractal Bag of Features
#main('singularityBoF1000_5',2,True,RANDOM_FOREST)

main('efd_x_60_2',6,False,SVM)
