import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, LogStretch
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

# Convert HMS and DMS to degrees
def hms_to_deg(h, m, s):
    return 15 * (h + m / 60 + s / 3600)

def dms_to_deg(d, m, s):
    sign = 1 if d >= 0 else -1
    return d + sign * (m / 60 + s / 3600)

def parse_coords(coords):
    parsed_coords = []
    for coord in coords:
        ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = map(float, coord.replace(':', ' ').replace(',', ' ').split())
        ra_deg = hms_to_deg(ra_h, ra_m, ra_s)
        dec_deg = dms_to_deg(dec_d, dec_m, dec_s)
        parsed_coords.append((ra_deg, dec_deg))
    return parsed_coords

# Calculate FWHM and recenter the stamp
def calculate_fwhm_and_recenter(data, x, y, pixel_scale, stamp_size):
    half_size = stamp_size // 2
    sub_image = data[y-half_size:y+half_size+1, x-half_size:x+half_size+1]

    # Ensure sub_image is not empty
    if sub_image.size == 0:
        raise ValueError("Sub-image is empty. Check coordinates and stamp size.")

    mean, median, stddev = sigma_clipped_stats(sub_image)
    sub_image -= median

    y_grid, x_grid = np.mgrid[:sub_image.shape[0], :sub_image.shape[1]]
    gaussian_init = models.Gaussian2D(amplitude=sub_image.max(), x_mean=half_size, y_mean=half_size)
    fit_p = fitting.LevMarLSQFitter()
    gaussian_fit = fit_p(gaussian_init, x_grid, y_grid, sub_image)

    # Recenter based on Gaussian fit
    x_center = int(x + gaussian_fit.x_mean.value - half_size)
    y_center = int(y + gaussian_fit.y_mean.value - half_size)

    # Re-extract the sub-image centered on the new peak
    sub_image_centered = data[y_center-half_size:y_center+half_size+1, x_center-half_size:x_center+half_size+1]

    # Ensure sub_image_centered is not empty
    if sub_image_centered.size == 0:
        raise ValueError("Centered sub-image is empty. Check new coordinates and stamp size.")

    # Refit Gaussian to the recentered sub-image
    gaussian_fit_centered = fit_p(gaussian_init, x_grid, y_grid, sub_image_centered)

    fwhm_x = 2.355 * gaussian_fit_centered.x_stddev.value
    fwhm_y = 2.355 * gaussian_fit_centered.y_stddev.value
    fwhm = np.mean([fwhm_x, fwhm_y])

    fwhm_arcsec = fwhm * pixel_scale

    return fwhm, fwhm_arcsec, sub_image_centered

# Extract the stamp and display
def extract_stamp(data, x, y, stamp_size):
    half_size = stamp_size // 2
    return data[y-half_size:y+half_size+1, x-half_size:x+half_size+1]

def display_stamps(stamps_before, stamps_after, titles):
    num_stamps = len(stamps_before)
    
    # Create the figure with 2 subplots side by side for each stamp
    fig, axes = plt.subplots(num_stamps, 2, figsize=(12, 6 * num_stamps))

    # Ensure axes is always a 2D array, even if there's only one stamp
    if num_stamps == 1:
        axes = np.array([axes])

    norm = ImageNormalize(stretch=LogStretch())

    for i, (stamp_before, stamp_after, title) in enumerate(zip(stamps_before, stamps_after, titles)):
        # Display the "Before PSF Scaling" stamp
        axes[i, 0].imshow(stamp_before, origin='lower', norm=norm, cmap='viridis')
        axes[i, 0].set_title(f"{title} - Before PSF Scaling")
        axes[i, 0].axis('off')

        # Display the "After PSF Scaling" stamp
        axes[i, 1].imshow(stamp_after, origin='lower', norm=norm, cmap='viridis')
        axes[i, 1].set_title(f"After PSF Scaling")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def get_fwhm_and_stamps(fits_file, coords, pixel_scale, stamp_size=31):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
        if not wcs.has_celestial:
            raise ValueError("WCS should contain celestial component")

    sky_coords = SkyCoord(ra=[coord[0] for coord in coords], dec=[coord[1] for coord in coords], unit='deg')
    pixel_coords = sky_coords.to_pixel(wcs)

    results = []
    stamps = []
    for i, ((ra, dec), (x, y)) in enumerate(zip(coords, zip(*pixel_coords))):
        x, y = int(x), int(y)
        try:
            fwhm_pix, fwhm_arcsec, centered_stamp = calculate_fwhm_and_recenter(data, x, y, pixel_scale, stamp_size)
            results.append((ra, dec, fwhm_pix, fwhm_arcsec))
            stamps.append(centered_stamp)
        except ValueError as e:
            print(f"Error processing coordinates {ra}, {dec}: {e}")
            results.append((ra, dec, None, None))
            stamps.append(None)

    return results, stamps

if __name__ == '__main__':
    # Paths to the FITS files
    fits_file_before = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_UDS_NORTH\psf_matched_primer-uds-north-grizli-v7.2_f410m.fits'
    fits_file_after = r'c:\Users\Saskia.Hagan-Fellow\OneDrive - ESA\Documents\JWST_DATA_PRIMER\PRIMER_UDS_NORTH\psf_matched_primer-uds-north-grizli-v7.2_f090w.fits'
    
    coords = [
        "02:17:33.1217, -05:07:31.416"
    ]

    parsed_coords = parse_coords(coords)
    pixel_scale = 0.04  # Adjust this as needed
    stamp_size = 31  # Size of the stamp, 31x31 pixels

    # Extract stamps from both images
    results_before, stamps_before = get_fwhm_and_stamps(fits_file_before, parsed_coords, pixel_scale, stamp_size)
    results_after, stamps_after = get_fwhm_and_stamps(fits_file_after, parsed_coords, pixel_scale, stamp_size)

    # Print FWHM results
    for (ra, dec, fwhm_pix, fwhm_arcsec) in results_before:
        if fwhm_pix is not None:
            print(f"Before PSF Matching: RA={ra}, Dec={dec}, FWHM (pixels)={fwhm_pix:.2f}, FWHM (arcsec)={fwhm_arcsec:.2f}")
    
    for (ra, dec, fwhm_pix, fwhm_arcsec) in results_after:
        if fwhm_pix is not None:
            print(f"After PSF Matching: RA={ra}, Dec={dec}, FWHM (pixels)={fwhm_pix:.2f}, FWHM (arcsec)={fwhm_arcsec:.2f}")

    # Display stamps side by side with labels
    titles = [f"Star at RA: {ra}, Dec: {dec}" for (ra, dec) in parsed_coords]
    display_stamps(stamps_before, stamps_after, titles)
