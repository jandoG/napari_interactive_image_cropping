# All image processing functions are moved here to clear up the notebooks
# Notebooks and this file have to be shipped together!
# Partly adapted from Noreen Walker (Scientific Computing Facility, MPI-CBG)'s work
#
# Author: Gayathri Nadar (Scientific Computing Facility, MPI-CBG)

# 2021-08

import numpy as np
from matplotlib import pyplot as plt
import tifffile
from tifffile import imread, imwrite, TiffFile, TiffSequence
from csbdeep.io import save_tiff_imagej_compatible
import glob
import os, time, re

import napari
from skimage import transform, img_as_float, exposure

def loadImagesFromPath(folderpath):
    """
    Load images from a folder with extension. 
    
    params: folderpath - path to images 
    extension: default is .tif 
    
    returns: list of sorted filenames
    """
    filenames = naturalSort(glob.glob(folderpath))
    
    return filenames 
    
def naturalSort(l): 
	# from here: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def prepareMovieForCrop(files_array):
    """
    Prepare movie from individual files - concatenate the individual files (TCYX) into a complete time-lapse movie. 
    Each file in the files_array is read and added to a list. 
    This list is concatenated along the first axis (axis=0) to make a time sequence. 
    
    params:
    files_array: list of filenames extracted from the folder 
    
    returns: 
    concatenated image sequence TCYX where T = t1 + t2 + t3 + ... + tn (tx = no time points in each file)
    """
    # 20211027
    # == updated to read individual files and concatenate 
    # == works even if some images have different no of time points 
    # == works even if only one image in files_array
    
    images = [] 
    for f in files_array:
        im = tifffile.imread(f) 
        images.append(im)

    image_sequence_movie = np.concatenate(images, axis = 0)
    shape_movie = image_sequence_movie.shape
    
    # == outdated part, only here for reference
    """
    == 20211027: issues if one file in array has different number of time points ==
    == ignore this section ==
    image_sequence = imread(files_array) 
    shape_full = image_sequence.shape #NofilesTCYX
    ndims = image_sequence.ndim       #5
    assert ndims == 5, "Invalid number of dimensions, should be (Numberoffiles,T,C,Y,X)"
    # flatten first 2 dimensions = n images x t timepoints = nt time points, 2 channels 
    image_sequence_movie = image_sequence.reshape(-1, *image_sequence.shape[-3:]) # TCYX 
    shape_movie = image_sequence_movie.shape
    """
    
    return image_sequence_movie, shape_movie 

def displayForCrop(movie, contrasts):
    """
    Sets up interactive viewer to crop image. 
    
    params: 
    movie: T,C,Y,X
    contrasts: contrast limits (computed outside only once: faster)
    
    returns: the shapes layer from which the roi can be extracted
    """
    dispmovie = movie 
    
    nframes, nchannels, nrows, ncols = dispmovie.shape 
    h=nrows
    w=ncols
    
#     print(nframes, nchannels, nrows, ncols)
    
    offset=10
    roi_init=np.array([[offset,offset],[h-offset, offset], [h-offset, w-offset],[offset,w-offset]])
    shapes = [roi_init]

    # add the image
    viewer = napari.view_image(dispmovie, channel_axis=1, name='image',title="Crop image", contrast_limits=contrasts)

    # add the polygons
    shapes_layer = viewer.add_shapes(shapes, name='Roi',shape_type='rectangle', edge_width=2, # orange
                              edge_color='coral', face_color='#ffb17d48')#,opacity=0.7)
    shapes_layer.mode="SELECT"
    
    return shapes_layer


def extractAndPrepRoi(shapes_layer, max_r, max_c):
    """
    Processes data obtained from a napari 'shapes layer'.
    Extracts the rectangle roi coordinates of the shapes_layer. Clips to valid range.
    If no roi exists in the shape layer the full image size is returned as roi: 0,max_r,0,max_c
    
    params:
    shapes_layer: napari shapes layer
    max_r: maximum possible value for rows (from image shape)
    max_c: similar, for columns
    returns: [rmin,rmax,cmin,cmax]
    """
    shapes=shapes_layer.data
    if len(shapes)>=1:
        roi=shapes[0]

        # make sure it's a rectangle & convert to int
        rmin=int(min(roi[:,0]))
        rmax=int(np.round(max(roi[:,0])))
        cmin=int(min(roi[:,1]))
        cmax=int(np.round(max(roi[:,1])))

        # clip to image dims
        rmin=max(0,rmin)
        rmax=min(max_r,rmax)
        cmin=max(0,cmin)
        cmax=min(max_c,cmax)
        print("Roi coordinates (px): top-left:(",rmin,",",cmin,"),  bottom-right: (",rmax,",",cmax,
              "). Limits from image dim: :(",max_r,",",max_c,")" )       
    else:
        rmin=0
        rmax=max_r
        cmin=0
        cmax=max_c
        print("No cropping Roi selected. Using full range: top-left:(",rmin,",",cmin,"), bottom-right: (",rmax,",",cmax, ")")
        
    return [rmin,rmax,cmin,cmax]

def cropImage(movie, cropcoords):
    """
    Crop the movie based on the crop coordinates extracted. 
    
    params:
    movie: one complete image sequence TCYX 
    cropcoords: dict specifying cropping rectangle coordinates and time range 

    returns: cropped movie TCYX 
    """
    # --Crop--
    if cropcoords is not None:
        cropped=movie[cropcoords["tmin"]:cropcoords["tmax"],:,
                    cropcoords["ymin"]:cropcoords["ymax"],cropcoords["xmin"]:cropcoords["xmax"]]
    else:
        cropped=movie 
    
    print("Cropping done. Shape cropped image (TCYX): ",cropped.shape)
    
    return cropped 

def getNormalizedSequence(imgsequence, method = "clahe", kernelsize = None):
    """
    Apply normalization on the image sequence. Histogram of the entire sequence is used. 
    
    params:
    imgsequence: image to normalize (numpy array)
    method: string - 'clahe' for adaptive normalization or 'percentile' for percentile based 
    kernel size: for clahe, defines the shape of contextual regions used in the algorithm. If integer, it is broadcasted to each image dimension.
    
    returns: normalized image 
    """
     # Adaptive Equalization
    if method == "clahe":
        if kernelsize is not None:
            img_norm = exposure.equalize_adapthist(imgsequence, kernelsize, clip_limit=0.01)
        else:
            img_norm = exposure.equalize_adapthist(imgsequence, clip_limit=0.01)
        img_norm = exposure.rescale_intensity(img_norm, out_range=(0, 65535)).astype(np.uint16)
    
    # Contrast stretching
    elif method == "percentile":
        p2, p98 = np.percentile(imgsequence, (2, 98))
        img_norm = exposure.rescale_intensity(imgsequence, in_range=(p2, p98))
        
    else:
        print("ERROR: Unknown normalization method") 
    
    print("Histogram normalization done. Shape cropped-normalized image (TYX): ", img_norm.shape)
    
    return img_norm

def save_image(img, calib, frameinterval, outdir, savename):
    """Saves an image to imagej tiff hyperstack format. Adds spatial calibration (pixelsize)
    to meta-data.
    The axis order of the input img is important!!! Img of type (z,y,x) are NOT implemented here!
    IJ hyperstack requires a 6 dimensional array in axes order TZCYXS, fix the axes before saving
    
    img: numpy array of shape (T,C,Y,X) or (T,Y,X)
    calib: spatial calibration in (x,y), as list. 
    outdir: will be automatically created
    savename: image file name (incl ".tif")
    is3D: set to False for single channel sequence
    """
    # create outputdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    fullsavename = os.path.join(outdir,savename)
    
    # IJ hyperstack requires a 6 dimensional array in axes order TZCYXS
    # prepare axis order
    if img.ndim==4:
        outimg= img[:,np.newaxis,:]    # -> TCYX, add an axis for Z
        is3d=True
        
    elif img.ndim==3:
        outimg=img[:, np.newaxis, np.newaxis, :] # -> TYX, add an axis for Z, C
        is3d=False
    
    else:
        print("Wrong image dimensions (not implemented). Failed to save image.")
        return 

    # for tifffile saving: help(tifffile) and https://forum.image.sc/t/python-copy-all-metadata-from-one-multipage-tif-to-another/26597/8
    if is3d:
        tifffile.imsave(fullsavename,outimg, imagej=True, resolution=(1/calib[0], 1/calib[1]),
                metadata={'axes': 'TCYX', 'unit': 'um', 'finterval': frameinterval})
    else:
        tifffile.imsave(fullsavename,outimg, imagej=True, resolution=(1/calib[0], 1/calib[1]),
                metadata={'axes': 'TYX', 'unit': 'um', 'finterval': frameinterval})
        
def saveImageIJCompatible(img, outdir, savename, calib, frameinterval, axes='TCYX'):
    """Saves an image to imagej tiff hyperstack format Adds spatial calibration (pixelsize)
    to meta-data.
    
    img: numpy array of shape (T,C,Y,X) or (T,Y,X)
    calib: spatial calibration in (x,y), as list. 
    outdir: will be automatically created
    savename: image file name (incl ".tif")
    frameinterval: in seconds 
    axes: sequence such as TCYX or TYX (specify according to IJ order TZCYXS)
    """
    # create outputdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    fullsavename = os.path.join(outdir,savename)
    
    tifffile.imwrite(fullsavename, img, imagej=True, resolution=(1/calib[0], 1/calib[1]), 
                     metadata={'unit': 'um', 'finterval': frameinterval, 'axes': axes})
    