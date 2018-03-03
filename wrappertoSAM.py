#Main has to change to a function defintion such that one can call SAM from a
#script directly

#Original Usage to compute saliency maps using our pre-trained model:
#python main.py test path/to/images/folder/

from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_image, preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
import matplotlib.image as mpimg
import pdb


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

def generator_test_singleimage(b_s, image):
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    while True:
        yield [preprocess_image(image, shape_r, shape_c), gaussian]


def saliencyattentivemodel(inputimage):

    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    # version (0 for SAM-VGG and 1 for SAM-ResNet)
    if version == 0:
        m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
        print("Compiling SAM-VGG")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    elif version == 1:
        m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
        print("Compiling SAM-ResNet")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    else:
        raise NotImplementedError
    # Output Folder Path
    output_folder = 'predictions/'
    nb_imgs_test = 1 #len(file_names)

    if nb_imgs_test % b_s != 0:
        print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    if version == 0:
        print("Loading SAM-VGG weights")
        m.load_weights('weights/sam-vgg_salicon_weights.pkl')
    elif version == 1:
        print("Loading SAM-ResNet weights")
        m.load_weights('weights/sam-resnet_salicon_weights.pkl')

    predictions = m.predict_generator(generator_test_singleimage(b_s=b_s, image=inputimage), nb_imgs_test)[0]
    print("predictions:",predictions[0])
    outname = 'SalmapFromWrapper.jpg'
    original_image = image
    res = postprocess_predictions(predictions[0], original_image.shape[0], original_image.shape[1])
    #mport pdb; pdb.set_trace()
    cv2.imwrite(output_folder + '%s' % outname, res.astype(int))
    cv2.imshow('salmap', res.astype('uint8'))
    cv2.waitKey(0)
    return res.astype('uint8')

def saliency_on_frame(saliency, frame, fudge_factor, channel=2, sigma=3):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    #S = cv2.resize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = saliency if sigma == 0 else gaussian_filter(saliency, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    I = frame.astype('uint16')
    I[:,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

def mysaliency_on_frame(saliency, frame):
    pmax = saliency.max()
    #pdb.set_trace()
    saliency_normalized = saliency/pmax
    saliency_normalized_new = np.broadcast_to(np.expand_dims(saliency_normalized, 2), (saliency.shape[0],saliency.shape[1],3))
    frame = frame * saliency_normalized_new
    return frame.astype('uint8')


## Works like a charm --- Use this for all the attention overlays
def mysaliency_on_frame_colormap(saliency, frame):
    height, width, _ = frame.shape
    import pdb;pdb.set_trace()
    saliency_new = np.broadcast_to(np.expand_dims(saliency, 2), (saliency.shape[0],saliency.shape[1],3))
    heatmap = cv2.applyColorMap(saliency_new, cv2.COLORMAP_JET)
    result = heatmap * 0.4 + frame * 0.5
    return result.astype('uint8')



# GBVS Matlab HeatMaponImage Here
def overlaysaliency_on_frame(saliency, frame):
    smax = saliency.max()
    fmax = frame.max()
    heatmap = saliency/smax
    termtemp = np.power(heatmap, 0.8)
    term1 = np.broadcast_to(np.expand_dims(termtemp, 2), (heatmap.shape[0],heatmap.shape[1],3))
    arr1 = np.arange(1,4)
    arr2 = np.arange(1,51)
    temp = np.zeros(shape=(50,4))
    for i in arr2-1:
    	temp[i][:] = cm.jet(0.5*i)
    	colorfunc = temp[:,[0,1,2]] 																#  Dimensions:  50*3 
   	
    t = interpolate.interp2d(arr1,arr2,colorfunc)
    ip = t(arr1,1+49*(np.reshape(heatmap, (np.prod(heatmap.shape)))))								#  Dimensions: 172800 * 3
    ipp = np.transpose(ip)   																		#  Dimensions: 3*172800
    term2 = np.moveaxis(np.reshape(ipp,(3,heatmap.shape[0],heatmap.shape[1])),0,-1)
    framewithsaliencyoverlay = ((0.8*(1-term1))*(frame/fmax)) + (term1*term2)
    return framewithsaliencyoverlay



#Test Code calling SAM
imgs_test_path = '/home/ml/kkheta2/sam/sample_images/test/'
name = 'frame1.jpg'
fullfilename = imgs_test_path + name
image = cv2.imread(fullfilename)
cv2.imshow('video',image)
cv2.waitKey(0)
print("Predicting saliency maps for " + fullfilename)
salmap = saliencyattentivemodel(image)
originalimagesize = image.shape
salmapsize = salmap.shape
print "Image size:",originalimagesize, "Salmap size:", salmapsize

#Simple Display Overlay Plots
#i1 = plt.imshow(image)
#i2 = plt.imshow(salmap, cmap='jet', alpha=0.5)
#plt.savefig(("frame1" + "overlay" + ".png"), orientation='portrait', papertype=None, format=None,
#            transparent=True, bbox_inches='tight', pad_inches=0,
#            frameon=False)
#plt.show()


#Graph Based Visual Saliency Overlay Maps
#overlay = overlaysaliency_on_frame(salmap, image)
#cv2.imshow('Overlay',overlay)
#cv2.waitKey(0)

#My Saliency Function: Gives Regions Which are salient and rest all is black 
overlay = mysaliency_on_frame(salmap, image)
cv2.imshow('Overlay',overlay)
cv2.waitKey(0)

#My Saliency Function: Gives Regions Which are salient and rest all is black 
import pdb;pdb.set_trace()
overlay = mysaliency_on_frame_colormap(salmap, image)
cv2.imshow('Attention On Frame',overlay)
cv2.waitKey(0)

print("Done ")