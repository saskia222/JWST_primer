import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
import pypher.pypher as ph
from photutils.psf.matching import resize_psf
from scipy.signal import fftconvolve

def rescale_psf_image(input_file, output_file, desired_pixel_scale):
    """
    Rescales a FITS image to a desired pixel scale.
    """
    with fits.open(input_file) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        current_pixel_scale = header.get('PIXELSCL') 

    print(f"Current pixel scale: {current_pixel_scale}")

    # Resize PSF using photutils' resize_psf method
    ny, nx = data.shape
    ny_resize = int(np.round(ny * current_pixel_scale / desired_pixel_scale))
    ny_resize = int((ny_resize // 2) * 2 + 1)  # Ensure odd number of pixels
    PSF_pixel_scale = ny_resize / ny * desired_pixel_scale
    resized_psf_data = resize_psf(data, PSF_pixel_scale, desired_pixel_scale)
    
    # Update the header with the desired pixel scale
    header['PIXELSCL'] = desired_pixel_scale
    
    # Create a new HDU with the resampled data and updated header
    hdu = fits.PrimaryHDU(data=resized_psf_data, header=header)
    
    # Save the resampled image to a new FITS file
    hdu.writeto(output_file, overwrite=True)
    print(f"Rescaled PSF image saved to {output_file}")

def run_pypher(psf_source, psf_target, kernel_output, reg_fact=1e-5):
    """
    Computes the homogenization kernel using Pypher.
    """
    psf_source_data = fits.getdata(psf_source)
    psf_target_data = fits.getdata(psf_target)

    psf_source_data = psf_source_data.astype(np.float32)
    psf_target_data = psf_target_data.astype(np.float32)

    pixscale_source = fits.getheader(psf_source)['PIXELSCL']
    pixscale_target = fits.getheader(psf_target)['PIXELSCL']

    psf_source_data /= psf_source_data.sum()
    psf_target_data /= psf_target_data.sum()

    if pixscale_source != pixscale_target:
        try:
            psf_source_data = ph.imresample(psf_source_data, pixscale_source, pixscale_target)
        except MemoryError:
            raise MemoryError('The size of the resampled PSF would exceed 10K x 10K. Please resize your image and try again.')

    if psf_source_data.shape > psf_target_data.shape:
        psf_source_data = ph.trim(psf_source_data, psf_target_data.shape)
    else:
        psf_source_data = ph.zero_pad(psf_source_data, psf_target_data.shape, position='center')

    kernel, _ = ph.homogenization_kernel(psf_target_data, psf_source_data, reg_fact=reg_fact)

    if kernel is None or kernel.size == 0:
        raise ValueError("Computed kernel is invalid. Check the PSF data and parameters.")

    fits.writeto(kernel_output, data=kernel, overwrite=True)
    print(f"Homogenization kernel saved to {kernel_output}")

def convolve_with_fftconvolve(data, kernel):
    """
    Convolve an image with a kernel using fftconvolve from scipy.
    """
    return fftconvolve(data, kernel, mode='same')

def psf_matching_tile_workflow(input_psf, output_psF, ref_psf, desired_pixel_scale, kernel_output, input_image, output_image, tile_size=(1024, 1024), overlap=50):
    """
    Executes the PSF matching workflow on image tiles with overlap to avoid memory errors and boundary artifacts.
    """
    rescale_psf_image(input_psf, output_psF, desired_pixel_scale)
    run_pypher(output_psF, ref_psf, kernel_output)

    # Load the target image and its WCS
    with fits.open(input_image) as hdul:
        target_data = hdul[0].data
        target_header = hdul[0].header  
        target_wcs = WCS(target_header)
    
    # Load the kernel
    with fits.open(kernel_output) as hdul:
        kernel_data = hdul[0].data
    
    ny, nx = target_data.shape
    tile_ny, tile_nx = tile_size

    convolved_data = np.zeros_like(target_data)

    # Process each tile with overlap
    for y in range(0, ny, tile_ny - overlap):
        for x in range(0, nx, tile_nx - overlap):
            # Define the tile boundaries with overlap
            y1 = max(y - overlap // 2, 0)
            y2 = min(y + tile_ny + overlap // 2, ny)
            x1 = max(x - overlap // 2, 0)
            x2 = min(x + tile_nx + overlap // 2, nx)
            
            # Extract the tile
            tile = target_data[y1:y2, x1:x2]
            
            # Convolve the tile using fftconvolve
            convolved_tile = convolve_with_fftconvolve(tile, kernel_data)
            
            # Place the convolved tile back into the convolved data array (account for overlap)
            convolved_data[y1:y2, x1:x2] = convolved_tile

    # Save the convolved image
    fits.writeto(output_image, convolved_data, header=target_header, overwrite=True)
    print(f"PSF matched image saved to {output_image}")
    print(f"Convolved image dimensions: {convolved_data.shape}")

if __name__ == '__main__':
    filters = ['f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w']

    base_psf_dir = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\JWST_psf'
    base_image_dir = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_COSMOS_EAST'

    ref_psf = rf'{base_psf_dir}\PSF_NIRCam_in_flight_opd_filter_f444w_0.0399.fits'
    kernel_output = 'psf_kernel.fits'

    desired_pixel_scale = 0.03999
    filenames = os.listdir(base_image_dir)

    for filename in filenames:
        if filename.endswith('.fits'):
            # Extract the filter name from the filename assuming it matches the filter convention
            parts = filename.split('-')
            if len(parts) > 3:
                file_filter_name = parts[-2].lower()  # Adjust index based on your filename structure
                
                # Check if the extracted filter name matches one of the filters in the list
                if file_filter_name in filters:
                    unique_filename_part = '-'.join(parts[:5])
                    input_psf = rf'{base_psf_dir}\PSF_NIRCam_in_flight_opd_filter_{file_filter_name.upper()}.fits'
                    output_psf = rf'{base_psf_dir}\PSF_NIRCam_in_flight_opd_filter_{file_filter_name.upper()}_0.0399.fits'
                    input_image = rf'{base_image_dir}\{filename}'
                    output_image = rf'{base_image_dir}\psf_matched_{unique_filename_part}_{file_filter_name}.fits'

                    psf_matching_tile_workflow(input_psf, output_psf, ref_psf, desired_pixel_scale, kernel_output, input_image, output_image)
