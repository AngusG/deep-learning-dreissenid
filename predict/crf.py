# for post-processing model predictions by conditional random field
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


def run_crf(rgb, pred_np):
    """
    Takes an input image and corresponding model predictions, then
    runs a conditional random field (CRF) on the predictions.

    @param rgb: is an rgb image in uint8 format (pixels 0-255)
    @param pred_np: are model predictions in greyscale format as float [0,1]
    """
    MAX_ITER = 20
    labels = np.stack([pred_np, 1 - pred_np])
    c, h, w = labels.shape[0], labels.shape[1], labels.shape[2]
    labels = labels.astype('float') / labels.max()
    U = utils.unary_from_softmax(labels)
    U = np.ascontiguousarray(U)
    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    """
    @param compat=3, Potts model - it introduces a penalty for nearby similar
    pixels that are assigned different labels.
    """
    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=3, compat=6)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb, compat=10)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))

    # binarize output
    Q[0][Q[0] >= 0.5] = 1
    Q[0][Q[0] < 0.5] = 0

    return Q[0, :, :]