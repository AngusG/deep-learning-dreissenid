===============
Quadrat Extraction
===============

### To successfully extract a quadrat the algorithm assumes:

- All four corners of the quadrat are contained in the image, i.e. the side
lengths or corners are not cropped from the scene.

- The quadrat side lengths are not occluded, e.g., by diver equipment, mesh bags, or vegetation.

- The image is sufficiently clear, i.e.~not turbulent or cloudy from disrupted sediment.

- The camera angle is within 65-90 degrees with respect to the top surface of
the quadrat. Note that this is separate from quadrat rotation in the camera plane,
which can be arbitrary.

- The camera is not too far from the quadrat such that the side lengths are less
than 400 pixels for 1080x1440 resolution, 500 pixels for 1080x1920 HD resolution,
or 400 for portrait mode in HD res.


The algorithm still works reasonably well in some cases even when the
assumptions are violated, e.g., input 7 with the mesh bag covering one of the
corners, as missing corner coordinates can sometimes be inferred if enough
complementary lines are detected. Conversely, even when the assumptions are
satisfied, a best effort is made to extract the *interior* of the quadrat, but
this occasionally won't be possible due to missing or misleading lines and part
of the quadrat may be included in the output.

.. autofunction:: utils.crop_still_image
