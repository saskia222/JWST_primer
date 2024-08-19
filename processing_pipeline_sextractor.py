"""
Code for full data processing pipeline of level 3 calibrated JWST images.

Stages:

	• Rescale PSF'S to 0.031
	• Rescale (dimension + pixel scale) of long wavelength images 
	• Psf match to F444W filter 
	• Get convolution kernel 
	• Convolve the image 

   Output both scaled 0.031 image + psf matched image

"""
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import zoom
from astropy.convolution import convolve_fft
from reproject import reproject_interp
import pypher.pypher as ph
from photutils.psf.matching import resize_psf

def rescale_psf_image(input_file, output_file, desired_pixel_scale):
    """
    Rescales a FITS image to a desired pixel scale.
    """
    hdul = fits.open(input_file)
    data_hdu = hdul[1]
    data = data_hdu.data
    header = data_hdu.header
    wcs = WCS(header)

    # Current pixel scale from the FITS header
    current_pixel_scale = header.get('PIXELSCL', 0.063)  # Default value if PIXELSCL not found
    print(f"Current pixel scale: {current_pixel_scale}")

    # Resize PSF using photutils' resize_psf method
    ny, nx = data.shape
    ny_resize = ny * current_pixel_scale / desired_pixel_scale
    ny_resize = np.round(ny_resize)
    ny_resize = int((ny_resize // 2) * 2 + 1)  # Make it an odd number of pixels to ensure PSF is centered
    PSF_pixel_scale = ny_resize / ny * desired_pixel_scale
    resized_psf_data = resize_psf(data, PSF_pixel_scale, desired_pixel_scale)
    #removed trim since want the psf to be the same size as the SW PSF'S 
    
    # Update the header with the desired pixel scale
    header['PIXELSCL'] = desired_pixel_scale
    # Create a new HDU with the resampled data and updated header
    hdu = fits.PrimaryHDU(data=resized_psf_data, header=header)
    
    # Save the resampled image to a new FITS file
    hdu.writeto(output_file, overwrite=True)
    print(f"Resampled image saved to {output_file}")

def run_pypher(psf_source, psf_target, kernel_output, reg_fact=1e-5):
    """
    Computes the homogenization kernel using Pypher.
    """
    # Load images
    psf_source_data = fits.getdata(psf_source)
    psf_target_data = fits.getdata(psf_target)

    # Get pixel scale 
    pixscale_source = fits.getheader(psf_source)['PIXELSCL']
    pixscale_target = fits.getheader(psf_target)['PIXELSCL']

    # Normalize the PSFs
    psf_source_data /= psf_source_data.sum()
    psf_target_data /= psf_target_data.sum()

    # Resample high resolution image to the low one
    if pixscale_source != pixscale_target:
        try:
            psf_source_data = ph.imresample(psf_source_data, pixscale_source, pixscale_target)
        except MemoryError:
            raise Exception('The size of the resampled PSF would have exceeded 10K x 10K. Please resize your image and try again.')
    
    print(psf_source_data.shape, psf_target_data.shape)
    # Check the new size of the source vs. the target
    if psf_source_data.shape > psf_target_data.shape:
        psf_source_data = ph.trim(psf_source_data, psf_target_data.shape)
    else:
        psf_source_data = ph.zero_pad(psf_source_data, psf_target_data.shape, position='center')

    kernel, _ = ph.homogenization_kernel(psf_target_data, psf_source_data, reg_fact=reg_fact)

    # Check if kernel is valid
    if kernel is None or kernel.size == 0:
        raise ValueError("Computed kernel is invalid. Check the PSF data and parameters.")

    # Write kernel to FITS file
    fits.writeto(kernel_output, data=kernel, overwrite=True)
    print(f"Pypher: Output kernel saved to {kernel_output}")

def reproject_fits(input_file, reference_file, output_file):
    """
    Reproject a FITS image to match the WCS and shape of a reference FITS image.
    Good for resizing the long wavelength JWST images.
    
    Parameters:
    - input_file: str, path to the input FITS file to be reprojected
    - reference_file: str, path to the reference FITS file defining the target WCS and shape
    - output_file: str, path to save the reprojected FITS file
    """
    # Open the input FITS file
    with fits.open(input_file) as hdul_input:
        input_data = hdul_input[1].data
        input_header = hdul_input[1].header
        input_wcs = WCS(input_header)

    # Open the reference FITS file
    with fits.open(reference_file) as hdul_ref:
        ref_data = hdul_ref[1].data
        ref_header = hdul_ref[1].header
        ref_wcs = WCS(ref_header)
        target_shape = ref_data.shape

    # Reproject the input image to match the reference WCS and shape
    reprojected_data, footprint = reproject_interp((input_data, input_wcs), ref_wcs, shape_out=target_shape)

    # Update the header with the reference WCS information
    output_header = ref_header.copy()

    # Update the header with the new dimensions
    output_header['NAXIS1'] = reprojected_data.shape[1]
    output_header['NAXIS2'] = reprojected_data.shape[0]

    # Create a new HDU with the reprojected data and updated header
    hdu = fits.PrimaryHDU(data=reprojected_data, header=output_header)

    # Save the reprojected image to a new FITS file
    hdu.writeto(output_file, overwrite=True)
    print(f"Reprojected image saved to {output_file}")
    
    # Print the pixel dimensions of the reprojected image
    print(f"Pixel dimensions of the reprojected image: {reprojected_data.shape}")

def psf_matching_workflow(input_psf, output_psf, ref_psf, desired_pixel_scale, kernel_output, input_image, output_image, reference_image, reprojected_output):
    # Step 1: Rescale the PSF image
    rescale_psf_image(input_psf, output_psf, desired_pixel_scale)
    
    # Step 2: Reproject the target image to match the reference image
    reproject_fits(input_image, reference_image, reprojected_output)
    # Step 3: Generate the convolution kernel
    run_pypher(output_psf, ref_psf, kernel_output)

    # Load the reprojected target image and the kernel
    with fits.open(reprojected_output) as hdul:
        target_data = hdul[0].data
        target_header = hdul[0].header  

    with fits.open(kernel_output) as hdul:
        kernel_data = hdul[0].data

    # Perform the convolution using convolve_fft from astropy
    convolved_data = convolve_fft(target_data, kernel_data, allow_huge=True)

    # Save the convolved image
    fits.writeto(output_image, convolved_data, header=target_header, overwrite=True)
    print(f"PSF matched image saved to {output_image}")

    print(convolved_data.shape)

    

if __name__ == '__main__':
    filters = ['f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m','f444w']

    # Define the base directories for images and PSFs
    base_psf_dir = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\JWST_psf'
    base_image_dir = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_UDS_SOUTH'

    # Want to PSF match to the f444w image
    ref_psf = rf'{base_psf_dir}\PSF_0.031_NIRCam_filter_F444W.fits'

    desired_pixel_scale = 0.031  # Adjust this as needed
    kernel_output = 'psf_kernel.fits'

# List all filenames in the specified base_image_dir directory
filenames = os.listdir(base_image_dir)
# Extract the first part of each filename before the first '_'
unique_filename_parts = set('_'.join(filename.split('_')[:3]) for filename in filenames if filename.endswith('.fits'))

for part in unique_filename_parts:
    for filter_name in filters:
        input_psf = rf'{base_psf_dir}\PSF_NIRCam_in_flight_opd_filter_{filter_name.upper()}.fits'
        output_psf = rf'{base_psf_dir}\PSF_NIRCam_in_flight_opd_filter_{filter_name.upper()}_0.031.fits'
        input_image = rf'{base_image_dir}\{part}_clear-{filter_name}_i2d.fits'
        output_image = rf'{base_image_dir}\psf_matched_{part}_{filter_name}.fits'
        reprojected_output = rf'{base_image_dir}\reproject_{part}_{filter_name}.fits'
        reference_image = rf'{base_image_dir}\{part}_clear-f090w_i2d.fits'

        # Run the PSF matching workflow
        psf_matching_workflow(input_psf, output_psf, ref_psf, desired_pixel_scale, kernel_output, input_image, output_image, reference_image, reprojected_output)
