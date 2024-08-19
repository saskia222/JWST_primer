import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LogStretch
# Define the path to your weight image file
wht_file = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_COSMOS_EAST\primer-cosmos-east-grizli-v7.0-f444w-clear_drc_wht.fits'


# Open the weight image data
with fits.open(wht_file) as hdul:
    wht_data = hdul[0].data
    header = hdul[0].header

# Calculate the RMS image
rms_data = 1 / np.sqrt(wht_data)

# Resize the RMS image to the desired shape
#resized_rms_data = resize_image_to_shape(rms_data, desired_shape)

# Update the header with the new dimensions
#header['NAXIS1'] = desired_shape[1]
#header['NAXIS2'] = desired_shape[0]

# Define the path for the output scaled RMS image file
rms_file = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_COSMOS_EAST\primer-cosmos-east-grizli-v7.0-f444w-clear_rms.fits'

# Save the scaled RMS image to a new FITS file
hdu = fits.PrimaryHDU(data=rms_data, header=header)
hdu.writeto(rms_file, overwrite=True)
print(f"Scaled RMS image saved to {rms_file}")

m, s = np.mean(rms_data), np.std(rms_data)
plt.imshow(rms_data, interpolation='nearest', cmap='gnuplot2', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()
plt.show() 