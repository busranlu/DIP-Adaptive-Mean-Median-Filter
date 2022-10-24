#Busra_Unlu_211711008_HW5

import numpy as np
import cv2 as cv
import math

img_filename1 ='noisyImage_Gaussian.jpg'
image1= cv.imread(img_filename1, 0)

img_filename = 'noisyImage_SaltPepper.jpg'
image = cv.imread(img_filename, 0)

img_filename2 = 'lena_grayscale_hq.jpg'
groundTruth = cv.imread(img_filename2, 0)


#normalise image [0-1]
normalised_image= cv.normalize(image1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

#QUESTION 1 
"""
Sxy=kernel size (5x5)
g(x,y)= noisy imput image
noise_variance=0.004
local_average
local_variance
f(x,y)=filter output
ref:https://www.youtube.com/watch?v=oUyuwvZDz7s&list=LLY_nxtCyTSevVUpLczqk0gQ
"""
def adaptiveMeanFilter(normalised_image,Sxy,noise_variance):   
    (h, w) = normalised_image.shape[:2]
    local_average=np.zeros((h,w))
    mean_squared_image=np.zeros((h,w))
    f=np.zeros((h,w))
    #padding
    top=left=right=bottom=(Sxy-1)//2
    padded_image = cv.copyMakeBorder( normalised_image,top, bottom, left, right, cv.BORDER_REPLICATE, None, value = 0 )
    #finding mean image
    for i in range(w):
        for j in range(h):
            temp = padded_image[i:i + Sxy, j:j + Sxy]
            local_average[i,j]=np.mean(temp)   
    
    #square of mean image
    square_mean_image=np.square(local_average)

    #square of image
    squared_image=np.square(padded_image)

    #mean of squared image
    for i in range(w):
        for j in range(h):
            temp2 = squared_image[i:i + Sxy, j:j + Sxy]
            mean_squared_image[i,j]=np.mean(temp2)

    #local variance
    local_variance= mean_squared_image - square_mean_image

    #formule implamentation
    f=normalised_image - ((noise_variance / local_variance) * (normalised_image - local_average ) )
    
    #get back[0-255]
    f = cv.normalize(f, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return f.astype(np.uint8)

#my adaptive median filter 
output_1_1=adaptiveMeanFilter(normalised_image,5,0.004)
#OpenCV 5x5 box filter
output_1_2=cv.blur(image1, (5,5), borderType = cv.BORDER_CONSTANT)
#OpenCV 5x5 Gaussian filter (with auto var  σ = 0).
output_1_3=cv.GaussianBlur(image1, (5, 5), 0, borderType = cv.BORDER_CONSTANT)


#QUESTION 2
def adaptiveMedianFilter(img, windowSize): 
    #padding
    top=bottom=left=right=windowSize
    padImg = cv.copyMakeBorder( image, top, bottom, left, right, cv.BORDER_REPLICATE, value = 0 )

    row = padImg.shape[0]
    col = padImg.shape[1]

    sMax = 7
    a=sMax//2
    filter_output = np.zeros(padImg.shape)
    for y in range(a, row - 1):
        for x in range(a, col - 1):
            filter_output[y, x] = stageA(padImg, y, x, windowSize, sMax)

    return filter_output[a:-a,a:-a].astype(np.uint8)
def stageA(img, y, x, windowSize, sMax):
    img_part = img[y - (windowSize//2):y + (windowSize//2) + 1, x - (windowSize//2):x + (windowSize//2) + 1]
    
    zmin = np.min(img_part)
    zmed = np.median(img_part)
    zmax = np.max(img_part)
    
    A1 = zmed - zmin
    A2 = zmed - zmax

    if A1 > 0 and A2 < 0:                       #go to level B
        return stageB(img_part, zmin, zmed, zmax)
    else:
        windowSize = windowSize + 2             #increase window size (must be odd so add 2)
        if windowSize <= sMax:                  #window size is lower than Smax, repeat level A 
            return stageA(img,y,x,windowSize,sMax)
        else:                                   # return zmed
            return zmed
def stageB(img, zmin, zmed, zmax):
    h,w = img.shape
    zxy = img[h // 2, w //2]

    B1 = zxy - zmin
    B2 = zxy - zmax

    if B1 > 0 and B2 < 0:   #return Zxy
        return zxy
    else:
        return zmed         #return zmed

#From HW3
#center weighted median filter
def weightedMedianFilter(image,kernelSize):
    h,w = image.shape
    #padding
    top  = bottom = int((kernelSize-1)/2)        #rows
    left = right  = int((kernelSize-1)/2)        #cols
    padImg = cv.copyMakeBorder( image, top, bottom, left, right, cv.BORDER_REPLICATE)
       
    imageW, imageH = image.shape               # get image dimensions
    filter_out = np.zeros((imageW, imageH))    # intialize output image
    #convolution
    for i in range(w):
        for j in range(h):
            temp = padImg[i:i + kernelSize, j:j + kernelSize]
            centerValue=temp[kernelSize//2,kernelSize//2]
            flattenedImg=temp.flatten()
            flattenedImg=np.append(flattenedImg,centerValue)
            flattenedImg=np.append(flattenedImg,centerValue)
            median=np.median(flattenedImg)
            filter_out[i,j] = median
    filter_out=filter_out.astype(np.uint8)
    return filter_out

output_2_1=adaptiveMedianFilter(image,3)
output_2_2=cv.medianBlur(image,3)
output_2_3=cv.medianBlur(image,5)
output_2_4=cv.medianBlur(image,7)
output_2_5=weightedMedianFilter(image,3)
output_2_6=weightedMedianFilter(image,5)
output_2_7=weightedMedianFilter(image,7)

#PSNR
psnr_1_1=cv.PSNR(groundTruth,output_1_1) 
psnr_1_2=cv.PSNR(groundTruth,output_1_2) 
psnr_1_3=cv.PSNR(groundTruth,output_1_3) 

psnr_2_1=cv.PSNR(groundTruth,output_2_1) 
psnr_2_2=cv.PSNR(groundTruth,output_2_2)  
psnr_2_3=cv.PSNR(groundTruth,output_2_3)  
psnr_2_4=cv.PSNR(groundTruth,output_2_4)  
psnr_2_5=cv.PSNR(groundTruth,output_2_5)  
psnr_2_6=cv.PSNR(groundTruth,output_2_6)  
psnr_2_7=cv.PSNR(groundTruth,output_2_7) 

#TERMINAL OUTPUTS
print("-----------Question1-------------------------------")
print("My Adaptive mean filter(5x5), PSNR= " + str(psnr_1_1) )
print("cv2 Box filter(5x5), PSNR= "          + str(psnr_1_2) )
print("cv2 Gaussian filter(5x5), PSNR= "     + str(psnr_1_3) )
print("-----------Question2-------------------------------")
print("my adaptive median filter(3x3), PSNR="+ str(psnr_2_1) )
print("cv2 median filter(3x3), PSNR= "       + str(psnr_2_2) )
print("cv2 median filter(5x5), PSNR= "       + str(psnr_2_3) )
print("cv2 median filter(7x7), PSNR= "       + str(psnr_2_4) )
print("weighted Median Filter(3x3) , PSNR= " + str(psnr_2_5) )
print("weighted Median Filter(5x5), PSNR= "  + str(psnr_2_6) )
print("weighted Median Filter(7x7), PSNR= "  + str(psnr_2_7) )

#OUTPUTS
cv.imshow("salt paper noise image", image)
cv.imshow("ground truth", groundTruth )
cv.imshow("Gaussian noise image", image1 )

cv.imshow("My Adaptive mean filter(5x5), PSNR= " + str(psnr_1_1) , output_1_1 )
cv.imshow("cv2 Box filter(5x5), PSNR= "          + str(psnr_1_2) , output_1_2 )
cv.imshow("cv2 Gaussian filter(5x5), PSNR= "     + str(psnr_1_3) , output_1_3 )

cv.imshow("my adaptive median filter(3x3), PSNR="+ str(psnr_2_1) , output_2_1 )
cv.imshow("cv2 median filter(3x3), PSNR= "       + str(psnr_2_2) , output_2_2 )
cv.imshow("cv2 median filter(5x5), PSNR= "       + str(psnr_2_3) , output_2_3 )
cv.imshow("cv2 median filter(7x7), PSNR= "       + str(psnr_2_4) , output_2_4 )
cv.imshow("weighted Median Filter(3x3) , PSNR= " + str(psnr_2_5) , output_2_5 )
cv.imshow("weighted Median Filter(5x5), PSNR= "  + str(psnr_2_6) , output_2_6 )
cv.imshow("weighted Median Filter(7x7), PSNR= "  + str(psnr_2_7) , output_2_7 )


cv.waitKey(0)
cv.destroyAllWindows()
