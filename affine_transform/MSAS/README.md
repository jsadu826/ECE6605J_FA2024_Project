Code structure
--------------

All classes can be informally divided into three groups: algorithms, structures and utilities.

**Algorithms:**

* `AffinePatchDistance` - encapsulates computation of affine invariant patch distance 
between two given points on two images.
* `StructureTensor` - encapsulates iterative scheme for computation of affine covariant 
structure tensors and affine covariant regions (shape-adaptive patches).
* `EllipseNormalization` - encapsulates logic related to normalization of an elliptical
region to a disc and subsequent interpolation of it to a regular grid.

**Structures:**

* `StructureTensorBundle` - represents a field of structure tensors (and other data) computed on an image.
* `DistanceInfo` - contains value of affine invariant patch distance together with
the transformations that gave this value and points at which it was computed.
* `GridInfo` - represents regular grid used for interpolation of normalized patches.
* `NormalizedPatch` - contains normalized patch interpolated to a regular grid together with
the normalizing transformation and the additional orthogonal transformation.

**Utilities:**

* `ArrayDeleter2d` - generic deleter for a 2D dynamic array. It is used for automatic management 
of memory allocated for normalized patches.

Navigating the code
-------------------

`AffinePatchDistance` should be considered the main entry point. It does not access raw images and 
instead operates on fields of structure tensors (`StructureTensorBundle`) which in turn encapsulate
the images. Computed distance is returned together with the corresponding transformations of patches
(`DistanceInfo`).

For performance reason elliptical patches are normalized (`EllipseNormalization`) to a regular grid 
(`GridInfo`). This normalized form (`NormalizedPatch`) is used for comparison of patches and
can also be cached in memory. Caching is implemented transparently in `StructureTensorBundle`.

`StructureTensorBundle` represents several related kinds of data computed from an underlying image.
Depending on the type, this data can be accessed either as a whole (e.g. image, gradients, etc.) 
or in a point-wise manner (e.g. tensor(x), region(x), etc.). Fields of structure tensors and regions 
are not computed all at once, instead values at every point are computed on-demand upon access. 
Computation of individual structure tensor and region is delegated to `StructureTensor`.

Usage
-----

Given: one or two input images, two points of interest.

1. Create and configure an instance of `StructureTensor`.
2. Create and configure one or two instances of `StructureTensorBundle` encapsulating the images.
3. Create and configure an instance of `AffinePatchDistance`.
4. Compute affine invariant patch distance between two given points.

Optionally all normalized patches can be pre-computed between steps 3 and 4 using the instance 
of `AffinePatchDistance`. This might be beneficial for use cases in which all points should be 
compared with each other and thus the deferred on-demand computation is meaningless.