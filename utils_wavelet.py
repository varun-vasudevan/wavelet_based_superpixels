import cv2
import numpy as np
import pywt
from skimage.segmentation import relabel_sequential

###############################################################################
################################# Notes #######################################
# See Azzalini et al. (2005) for conventions
# X discrete signal with N values (=N_x*N_y) with mean = 0
# X[k] = sum_{i,j,d}(coeffs[i,j,d]*wav_{i,j,d}[k])
# Wavelets are orthonormal: sum_{k} wav_{i,j,d}[k] wav_{i',j',d'}[k]
# = 1 when (i,j,d)=(i',j',d'), 0 otherwise
# It follows: sum(X[k]*X[k]) = sum_{i,j,d}(coeff[i,j,d]^2)

# (also see Donoho & Johnstone (1994) for thresholding theory)
# X = S + W, with W Gaussian white noise where S is estimated as:
# coeffs_S = 0 if abs(coeffs_X) <= T, coeffs_X otherwise
# T = std(W)*sqrt(2*ln(N))
# where std(W) = 1/N * sum_{k} W[k]^2 = 1/N * sum_{i,j,d}(coeffs_w[i,j,d]^2)

# Iterative procedure of Azzalini et al. (2005)
# Initialize std(W) = std(X), then std(W) = std(estimate of W)
###############################################################################


def cantor_pairing(a, b):
  ''' Cantor pairing function (unique NxN->N mapping):
    Source: https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers
  '''
  return (a+b)*(a+b+1)//2 + b


def one_time_filter(img_var, coeffs, thresh_val, verbose=False):
    """ Performs thresholding on the wavelet coefficients using the specified
    threshold value.
    Args:
        img_var: (Scalar) Variance of the input (original) image
        coeffs: (2D numpy array of shape(Nx, Ny))
            wavelet coefficients from the multiresolution analysis
        thresh_val: (Scalar) threshold value
        verbose: (Boolean) If True then prints, else doesn't print
    Returns:
        coeffs_thresh: say something here
        fimg_nnz: (Scalar) number of non-zeros in coeffs_thresh
        fimg_var: (Scalar) variance of coeffs_thresh
    """

    if verbose:
        print('> one_time_filter()')

    coeffs_thresh = pywt.threshold(coeffs, thresh_val, mode='hard', substitute=0)
    fimg_nnz = np.count_nonzero(coeffs_thresh)
    fimg_var = (coeffs_thresh**2).mean()

    noise = coeffs - coeffs_thresh
    noise_var = (noise**2).mean()

    if verbose:
        N = coeffs.shape[0] * coeffs.shape[1]
        print('> one_time_filter()')
        print('\nUpdated!\n\
          thresh: {:.4f}\n\
          img_var: {:.4f}, fimg_var: {:.4f}, noise_var: {:.4f}\n\
          img_var: {:.4f}, fimg_var: {:.4f}, noise_var: {:.4f}\n\
          fimg_nnz: {:d}, fimg_nnz: {:.4f}\
          '.format(thresh_val,\
              img_var, fimg_var, noise_var,\
              img_var/img_var, fimg_var/img_var, noise_var/img_var,\
              fimg_nnz, fimg_nnz/(N-1.)))

    return coeffs_thresh, fimg_nnz, fimg_var


def recursive_filter(img_var, coeffs, verbose=False):
    """ Performs thresholding on the wavelet coefficients using the specified
    threshold value.
    Args:
        img_var: (Scalar) Variance of the input (original) image
        coeffs: (2D numpy array of shape(Nx, Ny))
            wavelet coefficients from the multiresolution analysis
        verbose: (Boolean) If True then prints, else doesn't print
    Returns:
        coeffs_thresh: say something here
        thresh_old: (Scalar) the threshold calculated from the recursive algorithm
        fimg_nnz: (Scalar) number of non-zeros in coeffs_thresh
        fimg_var: (Scalar) variance of coeffs_thresh
    """

    if verbose:
        print('> recursive_filter()')

    if verbose:
        fimg_nnz_original = np.count_nonzero(coeffs)
        print('\nfimg_nnz_original: {:d}\n'.format(fimg_nnz_original));

    noise_var = img_var
    N = coeffs.shape[0]*coeffs.shape[1]
    rel_tol = 0.1e-2

    thresh_old = 0.0
    thresh_new = np.sqrt(2.0 * np.log(N) * noise_var)
    niter = 0
    while (np.abs(thresh_new-thresh_old) > rel_tol*thresh_old):
        thresh_old = thresh_new
        coeffs_thresh = pywt.threshold(coeffs, thresh_old, mode='hard', substitute=0)
        noise = coeffs - coeffs_thresh
        noise_var = (noise**2).mean()
        thresh_new = np.sqrt(2.0 * np.log(N) * noise_var)
        niter += 1

        if verbose:
            print('iter: {:2d}, thresh_old: {:.10f}, thresh_new: {:.10f}, noise_var : {:.10f}'.
              format(niter, thresh_old, thresh_new, noise_var))

    fimg_nnz = np.count_nonzero(coeffs_thresh)
    fimg_var = (coeffs_thresh**2).mean()

    if verbose:
        print('\nConverged!\n\
          thresh: {:.4f}\n\
          img_var: {:.4f}, fimg_var: {:.4f}, noise_var: {:.4f}\n\
          img_var: {:.4f}, fimg_var: {:.4f}, noise_var: {:.4f}\n\
          fimg_nnz: {:d}, fimg_nnz: {:.4f}\
          '.format(thresh_old,\
              img_var, fimg_var, noise_var,\
              img_var/img_var, fimg_var/img_var, noise_var/img_var,\
              fimg_nnz, fimg_nnz/(N-1.)))

    return coeffs_thresh, thresh_old, fimg_nnz, fimg_var


def create_mesh(Nx, rcoeffs, thresh_val, max_scale, verbose=False):
    """ Create mesh corresponding to superpixels up to max_scale
    Args:
        rcoeffs: [0 1 ... len(rcoeffs)-1] where:
            0: scaling function coefficient (mean of signal)
            1 ... len(rcoeffs)-1: all wavelet coefficients from coarse to fine
    """

    assert(max_scale >= 1), 'At least one level'
    assert(max_scale <= len(rcoeffs)-1), 'No more levels than available scales'

    if verbose:
        print('> create_mesh()')
        print('Creating mesh up to scale: ' + str(max_scale))

    mask = list()
    mask.append(np.empty((1))) # (!) mask is indexed starting at 1 as well: 1 -> max_scale
    for s in range(1, max_scale+1): # looping over scale 1 (coarsest) -> log(Nx,2) (finest)
        mask.append(np.logical_or(np.logical_or(abs(rcoeffs[s][0])>thresh_val,
                                                abs(rcoeffs[s][1])>thresh_val),
                                  abs(rcoeffs[s][2])>thresh_val))

    # To avoid L shape, it is necessary to propagate mask=1 from small to next larger scale, and so on
    #for s in range(max_scale, 1+1, -1): # from finest to coarsest
    for s in range(max_scale, 1, -1): # from finest to coarsest
        for i in range(0, mask[s].shape[0]): # looping through positions
            for j in range(0, mask[s].shape[1]):
                if mask[s][i,j]:
                    mask[s-1][i//2,j//2] = True

    sp = np.zeros((Nx,Nx), dtype=np.int64)
    tmp = 0
    for s in range(1, max_scale+1): # looping over scale 1 (coarsest) -> log(Nx,2) (finest)
        delta = int(Nx/2**(s-1))
        for i in range(0, mask[s].shape[0]): # looping through positions
            for j in range(0, mask[s].shape[1]):
                if mask[s][i,j]:
                    lft = np.array([delta*i, delta*j])
                    ctr_lft = lft + int(delta/2) - 1
                    ctr_rgt = ctr_lft + 1
                    rgt_exc = lft + delta

                    sp[lft[0]:1+ctr_lft[0],lft[1]:1+ctr_lft[1]] = tmp
                    sp[lft[0]:1+ctr_lft[0],ctr_rgt[1]:rgt_exc[1]] = tmp + 1
                    sp[ctr_rgt[0]:rgt_exc[0],lft[1]:1+ctr_lft[1]] = tmp + 2
                    sp[ctr_rgt[0]:rgt_exc[0],ctr_rgt[1]:rgt_exc[1]] = tmp + 3
                    tmp += 4

    if verbose:
        num_sp = len(np.unique(sp))
        print('\nnumber of ''superpixels'': {:d}'.format(num_sp))
        print('num_sp/(Nx**2): {:.4f}'.format(num_sp/(Nx**2)))

    sp, *__ = relabel_sequential(sp+1)
    return sp


def is_power_of_two(n):
    ''' Returns True if n is a power of 2
        Source: https://www.sanfoundry.com/python-program-find-whether-number-power-two/
    '''
    assert(n>0)
    return n & (n - 1) == 0


def check_img(img):
    ''' Check image is single channel, square, power of 2
        Source: https://www.sanfoundry.com/python-program-find-whether-number-power-two/
    '''
    assert(len(img.shape)==2) # should be single channel
    assert(img.shape[0]==img.shape[1]) # should be square
    assert(is_power_of_two(img.shape[0])) # size should be a power of two
    return


def wavelet_superpixel_singlechannel(image, params):
    ''' Compute and return wavelet-based superpixels for the input image.
    Args:
        img: image of Shape of H x W.
        params: dict containing the following keys:
            target_size:
            wname: name of the wavelet to be used.
            thresh_mult:
            verbose: if True print information during computation.

    Returns:
        sp: Dictionary of meshes (1-based label indicating which superpixel each
        pixel belongs to)
    '''

    wname = params['wname']
    num_sp = params['number']
    thresh_mult = params['thresh_mult']
    verbose = params['verbose']

    img = image
    check_img(img)
    Nx, Ny = img.shape

    img_mean = np.mean(img)
    img -= img_mean
    coeffs = pywt.wavedec2(img, wname)
    coeffs_arr_2d, coeff_slices = pywt.coeffs_to_array(coeffs)

    # perform recursive filtering
    img_var = ((img-np.mean(img))**2).mean()

    if (num_sp == 0): # use automatic algorithm
        # iterative algorithm determines and uses theoretical threshold
        coeffs_arr_2d_thresh, thresh_val, fimg_nnz, fimg_var = recursive_filter(
            img_var, coeffs_arr_2d, verbose=verbose)

        # if want different threshold, re-do one filtering with desired value
        if (thresh_mult != 1):
            #assert(thresh_val>0) # to alert user that thresh would not change
            if not thresh_val>0: # to alert user that thresh would not change
                print('*** wavelet threshold is 0! ***')
            thresh_val *= thresh_mult
            coeffs_arr_2d_thresh, fimg_nnz, fimg_var = one_time_filter(
                img_var, coeffs_arr_2d, thresh_val, verbose=verbose)

    else: # use (approximate) predefined number of super-pixels
        coeffs_arr_2d_raveled = coeffs_arr_2d.ravel()
        coeffs_arr_2d_raveled_sorted = np.sort(coeffs_arr_2d_raveled)
        thresh_val = coeffs_arr_2d_raveled_sorted[-num_sp//4-1] # this is where the approximation comes in
        coeffs_arr_2d_thresh, fimg_nnz, fimg_var = one_time_filter(
            img_var, coeffs_arr_2d, thresh_val, verbose=verbose)

    # perform reconstruction
    rcoeffs = pywt.array_to_coeffs(coeffs_arr_2d_thresh, coeff_slices, 'wavedec2')

    sp_dict = {}
    for scale in range(1, len(rcoeffs)): # 1 is coarsest
        sp_dict[scale] = create_mesh(Nx, rcoeffs, 0., scale, verbose=verbose)

    return sp_dict


def wavelet_superpixel(image, sp_params):
    ''' Compute and return wavelet-based superpixels for the input image.
    Args:
        img: image of Shape of H x W.
        params: dict containing the following keys:
            target_size:
            wname: name of the wavelet to be used.
            thresh_mult:
            verbose: if True print information during computation.

    Returns:
        sp: Dictionary of meshes (1-based label indicating which superpixel each
        pixel belongs to)
    '''
    if not sp_params['multichannel']:
        sp_dict = wavelet_superpixel_singlechannel(image, sp_params)
    else:
        sp_ch_dict = {}
        sp_dict = {}
        for channel in range(0, image.shape[-1]):
            sp_ch_dict[channel] = wavelet_superpixel_singlechannel(image[:,:,channel], sp_params)

            for scale in sp_ch_dict[0].keys():
                if channel==0:
                    sp_dict[scale] = sp_ch_dict[0][scale]
                else:
                    sp_dict[scale] = cantor_pairing(sp_dict[scale], sp_ch_dict[channel][scale])
                    sp_dict[scale], *__ = relabel_sequential(sp_dict[scale]+1)

    return sp_dict
