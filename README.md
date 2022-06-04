# README 

This repository contains jupyter notebooks which interactively crop fluorescence microscopy 2D time-lapse images with 2 channels. 

### Analysis goal:
- Crop interactively different regions in the 2D files, normalize and save them
- ***Expected input images***: 2D, 2 channels, time-lapse, time image files saved in folder (need to concatenate)
- ***Expected output***: cropped image 2D , single channel time-lapse, normalized to combat variabilities in intensities. Additionally, drift corrected.


### Workflow using Napari:

- Read images from folder - concatenate and display (XYCT).
- Display a rectangular Region of Interest (ROI).
- Ask user to adjust ROI (Region of Interest) to crop.
- Crop image to the ROI coordinates.
- Ask user to specify channel to normalize.
- Normalize the histogram of the cropped single channel time sequence image - options between percentile/ clahe methods. 
- Drift correct the image using image registration.
- Option to normalize other channel. 
- Option to apply same transformation as previous channel (drift correction).
- Save images: merge channels, save images with proper calibration and imagej compatible. Ask user to specify calibration and frame interval at the start and apply this to images before saving.


### Python libraries used
- `tifffile`: reading and writing images
- `napari`: display and cropping 
- `scikit image`: histogram normalization
- `pystackreg`: image registration to correct for drifting nuclei.

### Installation
- Use the environment file unser `set_up_instructions` to install napari and other required libraries.
- Keep the script `imgprocessing_functions.py` next to the jupyter notebook while running.


*This project was original done in collaboration with Ksenia Kuznetsova, PhD at Vastenhouw lab (Oct 2021, MPI-CBG, Dresden). The code is published with permission.*
