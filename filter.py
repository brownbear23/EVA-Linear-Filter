'''
Description: This script applies low-vision filters to images using Horizontal and Vertical Shift of Contrast Sensitivity Function
Default settings of imgs is set to '/images/...' then export degraded images to '/export/' and filters are set to [2,3, 4, 6, 10, 11, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
'''

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

## PARAMETERS
input_folder = 'images'
IMG = 'groceries.jpg'
filters = [2,3, 4, 6, 10, 11, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]


HShiftList = [1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
    0.250, 0.134, 0.072, 0.039, 0.022,
    0.267, 0.144, 0.078, 0.043, 0.024,
    0.314, 0.172, 0.096, 0.055, 0.032,
    0.345, 0.193, 0.110, 0.064, 0.038,
    0.439, 0.256, 0.154, 0.033, 0.018,
    0.125, 0.063, 0.031, 0.016, 1.000,
    1.000, 1.000, 1.000, 1.000, 1.000]

VShiftList = [1.000, 0.288, 0.157, 0.086, 0.048, 0.027,
    1.000, 0.534, 0.288, 0.157, 0.086,
    0.534, 0.288, 0.157, 0.086, 0.048,
    0.157, 0.086, 0.048, 0.027, 0.016,
    0.086, 0.048, 0.027, 0.016, 0.010,
    0.027, 0.016, 0.010, 0.534, 0.288,
    1.000, 1.000, 1.000, 1.000, 0.355,
    0.178, 0.089, 0.045, 0.022, 0.011]

def add_filter(img,HShift,VShift):
    """ Add low-vision filter to images using Horizontal and Vertical Shift of Contrast Sensitivity Function
    Refer to Xiong et al., 2021 Fontiers in Neuroscience Table for corresponding acuity (logMAR) and contrast sensitivity (logCS) levels

    Args:
        img (np.array): original image
        HShift (float): horizontal shift. The smaller the ratio is, the more blurry the image would be.
        VShift (float): vertical shift. The smaller the ratio is, the more low contrast the image would be. 

    Returns:
        np.array: degraded image
    """
    charIm = img.copy()
    thisHShift = HShift
    thisVShift = VShift
    # Calculate Viewing Angle
    PsysicalWidth = 20.5 # physical width/height of the image on the screen (cm)
    PsysicalHeight = 27 # physical width/height of the image on the screen (cm)
    distance=40 # Viewing distance in cm
    vh = 2*math.atan((PsysicalWidth)/(2*distance))*(180/math.pi) #horizontal visual angle of the image at the specified viewing distance
    vv = 2*math.atan((PsysicalHeight)/(2*distance))*(180/math.pi); #vertival visual angle of the image at the specified viewing distance
    imgSize = vh*vv #visual angle of the entire image at the specified viewing distance
    h=charIm.shape[1] # horizontal pixel number of the image
    v=charIm.shape[0] # vertical pixel number of the image
    #% hsize=PsysicalWidth/h; % height of a pixel in cm (cm/pixel)
    #% vsize=PsysicalHeight/v; % width of a pixel in cm (cm/pixel)

    fx = np.arange(start=-h/2, stop=h/2, step=1)
    fx = fx/vh
    fy = np.arange(start=-v/2, stop=v/2, step=1)
    fy = fy/vv
    [ux,uy] = np.meshgrid(fx,fy)
    finalImg = charIm
    for j in range(3): # three color channels or only luminance channel
        
        thisimage = charIm[:,:,j]
        meanLum = np.mean(thisimage)
        ## Generate blur
        
        # Vertical Shift
        for ii in range(thisimage.shape[0]):
            for jj in range(thisimage.shape[1]):
                if thisimage[ii,jj] !=255:
                    thisimage[ii,jj] = 255-(255-thisimage[ii,jj])*thisVShift

        ## Horizontal shift
        sSF0 = np.sqrt(ux**2+uy**2+.0001)
        CSF0 = (5200*np.exp(-.0016* (100/meanLum+1)**.08 * sSF0**2))/np.sqrt((0.64*sSF0**2+144/imgSize+1) * (1./(1-np.exp(-.02*sSF0**2))+63/(meanLum**.83)))
        
        sSF = (thisHShift)*np.sqrt(ux**2+uy**2+.0001)
        CSF = (5200*np.exp(-.0016*(100/meanLum+1)**.08*sSF**2))/np.sqrt((0.64*sSF**2+144/imgSize+1) * (1./(1-np.exp(-.02*sSF**2))+63/(meanLum**.83)))
        
        nCSF = CSF/CSF0
        nCSF = np.fft.fftshift(nCSF)
        
        maxValue = 1
        nCSF = nCSF*(1-(nCSF>maxValue))+maxValue*(nCSF>maxValue) #replace maximun to 1
        nCSF[0,0]=1
        
        Y = np.fft.fft2(thisimage)

        spectrum = np.abs(Y)

        filtImg = np.fft.ifft2(nCSF*Y)
        
        ## put the three channels together
        finalImg[:,:,j] = filtImg.astype(np.uint8)
        
    
    return finalImg


script_dir = os.path.dirname(os.path.abspath(__file__))

target_folder = os.path.join(script_dir,"export",str.split(IMG,'.')[0])

if not os.path.exists(target_folder):
    os.mkdir(target_folder)
    print(f"Folder '{target_folder}' created successfully.")
else:
    print(f"Folder '{target_folder}' already exists.")

imgFolder = os.path.join(script_dir,input_folder)
exportFolder = target_folder
inputImg = os.path.join(imgFolder,IMG)

# load image
charIm = cv2.imread(inputImg)
charIm = cv2.cvtColor(charIm, cv2.COLOR_BGR2RGB)
for filter in tqdm(filters):
    outputImg = os.path.join(exportFolder,str.split(IMG,'.')[0] + '_blur_' + str(filter) + '.png')
    thisHShift = 1/HShiftList[filter-1]
    thisVShift = VShiftList[filter-1]
    finalImg = add_filter(charIm,thisHShift,thisVShift)
    plt.imsave(outputImg,finalImg)
    

