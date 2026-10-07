# Camera Geometry And Pose Estimation

The SLAM branch so far: the pinhole camera model, the two pose problems
every visual SLAM system is built on, robust estimation with RANSAC, bundle
adjustment, and a full "two images in, camera motion out" pipeline. It is
all pure Python, and every demo has exact ground truth to grade against.

```bash
python3 pose_demo.py                    # synthetic matches, known 3D points
python3 pose_demo.py --outliers 0.3     # 30% wrong matches
python3 image_pair_demo.py              # rendered images -> corners -> pose
python3 image_pair_demo.py --output-dir /tmp/pair   # also writes PGM images
python3 -m unittest test_linalg test_geometry test_ransac test_bundle_adjustment test_features
```

## Files

| File | What it holds |
| --- | --- |
| `linalg.py` | list-of-lists matrices: Gaussian elimination, inverse, Jacobi eigen solver, 3x3 SVD, nearest rotation |
| `geometry.py` | rotations (Rodrigues), `Pose`, `PinholeCamera`, triangulation, linear PnP, eight-point epipolar/essential matrices, LM pose refinement |
| `ransac.py` | generic LO-RANSAC, robust PnP, robust two-view geometry with the Sampson distance |
| `bundle_adjustment.py` | two-view bundle adjustment: Schur complement, gauge fixing, Huber kernel, outlier rejection |
| `features.py` | Harris corners (built on `conv2d_valid` / `avg_pool2d`), NCC patch descriptors, mutual + ratio-test matching, PGM files |
| `pose_demo.py` | synthetic two-camera scene with optional wrong matches |
| `image_pair_demo.py` | ray-cast renderer for a textured scene, plus the full image pipeline |
| `engine.py` | scalar autograd, with `sin`, `cos`, `sqrt`, and `gradients()` for Jacobians |

## Conventions

- `Pose` maps world to camera: `X_cam = R X_world + t`. The camera center in
  world coordinates is `C = -R^T t`.
- Camera looks down `+z`; image `x` goes right and `y` goes down.
- Normalized coordinates remove the intrinsics: `(x, y) = (X/Z, Y/Z)`.

```text
pixel = K [X/Z, Y/Z, 1]^T

    [fx  0  cx]
K = [ 0 fy  cy]
    [ 0  0   1]
```

## The Estimators

**Rotation vectors.** A rotation is stored as a 3x3 matrix and optimized as
a 3-vector `w` (axis times angle) through Rodrigues' formula
`R = I + A [w]x + B [w]x^2`. Near `w = 0`, `A` and `B` switch to their Taylor
series in `theta^2`, which keeps autograd finite at exactly the point where
every refinement step starts.

**Triangulation (DLT).** Each view of a point gives two equations linear in
the homogeneous point: `x P_3 - P_1 = 0` and `y P_3 - P_2 = 0`. Stack them for
both views and take the null vector. `triangulate_matches` keeps only points
in front of both cameras.

**Absolute pose, PnP (DLT).** The same two equations per point, now linear
in the 12 entries of `P = [R | t]`. The null vector's left 3x3 block is
`s R`; snap it to the nearest rotation with an SVD. With `R` fixed, the
projection equations are *linear* in `t`, so `t` is re-solved by least
squares rather than by dividing by a guessed `s`. That step matters: with
minimal 6-point samples, the guessed-scale translation was so far off that
RANSAC found no inliers at all.

**Relative pose (eight-point algorithm).** Matches satisfy `x2^T E x1 = 0`
with `E = [t]x R`. Eight or more matches give a least-squares null vector,
which `epipolar_matrix` makes rank 2. `essential_matrix` then also forces two
equal singular values, and `relative_pose_from_essential` picks the one of
four `(R, +/-t)` factorizations that puts the points in front of both
cameras. The translation only has a direction: two views cannot tell the
scale of the scene.

**Refinement (Levenberg-Marquardt).** Linear estimates minimize an algebraic
error, not pixel error. `refine_pose` minimizes pixel reprojection error
directly:

1. parameterize a small update `delta = (w, dt)` around the current pose
2. build every pixel residual from `Value` scalars
3. read the Jacobian with `engine.gradients`, one residual at a time
4. solve `(J^T J + lambda diag(J^T J)) step = -J^T r` and adapt `lambda`

This is motion-only bundle adjustment, the problem ORB-SLAM3 solves for
every tracked frame. Here autograd replaces hand-derived Jacobians.

**RANSAC.** Fit a model to a random minimal sample (6 matches for PnP, 8
for two-view), count inliers, and keep the best. The number of tries adapts
to the inlier ratio `w`: `log(1 - p) / log(1 - w^s)`. Two details made it
work reliably:

- *Local optimization* (LO-RANSAC): every new best model is refit on all of
  its inliers until the inlier set stops growing. Minimal samples are
  noisy, so without this step the best model misses real inliers.
- *Score with the rank-2 epipolar matrix.* Two-view inliers are scored with
  the Sampson distance, a first-order estimate of a match's distance from
  satisfying `x2^T E x1 = 0`. Projecting each sample's estimate onto a true
  essential matrix (equal singular values) before scoring moved
  low-parallax models enough to lose a third of the real inliers. The
  essential constraint is enforced once, at the end.

**Two-view bundle adjustment.** Refines the second pose *and* every 3D point
to minimize pixel error in both images. Three ideas:

- *Schur complement.* Each residual touches the pose and one point, so the
  normal equations have an arrow shape: a 6x6 pose block, a block-diagonal
  3x3-per-point block, and the coupling between them. Eliminating the points
  leaves a 6x6 system, and then each point update is its own 3x3 solve. A
  test checks this against the dense `(6 + 3N)`-sized solve.
- *Gauge.* Scaling the whole scene changes nothing in the images, so after
  every step the scene is rescaled to keep the baseline length fixed.
- *Huber kernel and rejection.* Errors above `delta` count linearly, not
  quadratically, solved by reweighting each observation by
  `min(1, delta / error)`. Points still above a threshold after a round are
  dropped and the rest are adjusted again.

**Corners and matching.** Harris builds the structure tensor from Sobel
gradients (`conv2d_valid`) summed over a 5x5 window (`avg_pool2d`) and scores
`det - k trace^2`. Peaks are refined to sub-pixel accuracy with a parabola
fit. Patches are made zero-mean and unit-length, so their dot product is the
normalized cross-correlation. Matches must be mutual best matches and pass
Lowe's ratio test.

**Linear algebra.** `symmetric_eigen` is the cyclic Jacobi method: sweep
every off-diagonal pair, rotating each to zero, until the matrix is
diagonal. Null vectors are the smallest eigenvector of `A^T A`, and the 3x3
SVD is built from the eigen decomposition of `M^T M`.

## Results: Synthetic Matches (`pose_demo.py`)

40 points 4 to 8 units deep, a 640x480 camera with `f = 500`, 0.5 px
Gaussian pixel noise, second camera moved about 0.8 units sideways and
turned about 9 degrees.

| Step (default seed) | Rotation error | Translation | Reprojection RMSE |
| --- | --- | --- | --- |
| PnP, RANSAC + linear DLT | 0.075 deg | 0.0082 error | 0.648 px |
| PnP, after LM | 0.074 deg | 0.0077 error | 0.616 px |
| ground-truth pose, same noisy pixels | 0 | 0 | 0.647 px |
| Two-view, RANSAC eight-point | 0.417 deg | 3.8 deg direction | 1.03 px (both images) |
| Two-view, bundle adjusted | 0.609 deg | 0.70 deg direction | 0.298 px |

LM ends slightly *below* the ground-truth reprojection error, which is
exactly what least squares does: it fits this particular noise draw. With
`--noise 0`, every estimator is exact to about 1e-9.

**With 30% wrong matches** (`--outliers 0.3`), the plain linear methods fail
outright: PnP is 171 degrees off and eight-point is 14 degrees off. RANSAC
rejects every planted wrong match on the default seed, and after LM, PnP is
0.058 degrees off.

Single draws of the two-view problem vary a lot, because the baseline (0.8)
is short compared to the depth (about 6). Medians over 20 seeds with 30%
wrong matches:

| pixel noise | stage | rotation | translation direction | point error |
| --- | --- | ---: | ---: | ---: |
| 0.5 px | RANSAC eight-point | 0.70 deg | 3.6 deg | 0.55 |
| 0.5 px | + bundle adjustment | 0.54 deg | 0.88 deg | 0.41 |
| 1.0 px | RANSAC eight-point | 2.6 deg | 13.5 deg | 1.57 |
| 1.0 px | + bundle adjustment | 1.2 deg | 2.8 deg | 1.06 |

Wrong matches rejected: 98% at 0.5 px, 92% at 1.0 px. The rest happen to
lie on their epipolar lines (see below).

Bundle adjustment helps in the median, but not on every draw: its rotation
ends up *worse* than the eight-point estimate on 4 of 20 seeds at 0.5 px
(5 of 20 at 1.0 px), and the default seed above is one of them. It fits the
noise in both images, and with a short baseline rotation and translation
can trade off against each other (the "bas-relief" ambiguity), so a
lower reprojection error does not always mean a lower rotation error.

## Results: Rendered Image Pair (`image_pair_demo.py`)

Two 320x240 images (`f = 220`) are ray-cast from a back wall, a floor, and a
side wall tiled with random gray tiles. The second camera has moved 0.6
sideways and turned about 7 degrees. Ground truth grades each stage:

```text
Harris corners:      250 / 250
NCC matches:         114 (93 correct within 2 px)
RANSAC inliers:      95 (92 correct, 94 in front of both cameras), 27 iterations
RANSAC pose:         rotation err 1.722 deg, translation direction err 47.22 deg
BA outlier rejection: kept 92 (91 correct)
bundle adjusted:     rotation err 0.054 deg, translation direction err 0.37 deg, reprojection 0.106 px
reconstructed depth: median error 1.1% (true scale)
```

The eight-point start is poor here (47 degrees off in translation
direction), so bundle adjustment needs its full iteration budget (it stops
early once converged). Over render seeds 0 to 3 it lands at 0.37, 0.07,
4.5, and 0.12 degrees of translation-direction error, and 0.8 to 1.3% median
depth error. Correctly matched corners agree with ground truth to 0.26 px
(median over the four seeds).

Three lessons from getting this to work:

1. **Repetitive texture breaks two-view matching.** On an axis-aligned
   tile grid, rows of tile corners lined up with the (nearly horizontal)
   epipolar lines. A corner matched to its neighbour in the same row passed
   every epipolar test. Rotating the grid and staggering the rows removed
   most of these; larger patches (radius 7) removed more.
2. **A few epipolar-consistent wrong matches still bend the solution.** A
   wrong match that lies on its epipolar line triangulates to a
   self-consistent but wrong depth and reprojects almost perfectly. No
   two-view test can catch it. With a short baseline, the translation
   direction is so weakly constrained that four such matches moved it by 24
   degrees. The Huber kernel limits their pull (back to 0.3 degrees in that
   case). The 4.5-degree seed above is one remaining wrong match. Real
   systems settle this with a third view.
3. **Harris corners sit slightly inside sharp corners** (about 1.4 px for
   an L corner). The bias is the same in both views, so matches still agree.

## How This Maps To The Reference Points

These references stay distant: no real camera data, no keyframes, no map,
no loop closure. But the pieces above are their core.

| Building block | ORB-SLAM3 | DROID-SLAM |
| --- | --- | --- |
| pinhole projection, `Pose` | camera models, `Sophus::SE3` poses | same, as tensors |
| features and matching | ORB corners + binary descriptors | none: dense learned correlation |
| two-view initialization | essential/homography + RANSAC + cheirality | learned flow, no explicit E |
| frame tracking | PnP + motion-only BA (`refine_pose`) | dense BA layer on flow |
| robust cost | Huber kernel + chi-square outlier rejection in g2o | learned per-pixel confidence weights |
| Jacobians | hand-derived in g2o | autograd through a differentiable BA |
| bundle adjustment | local/global BA with Schur complement | differentiable dense BA with Schur complement |

## Next Gaps

- A third view: track points into a third image and estimate its pose with
  PnP. This is the natural way to reject epipolar-consistent wrong matches
  and fix the scale between frame pairs.
- Homography model for planar scenes. The eight-point algorithm is
  degenerate when every match lies on one plane, which is why the demo
  scene has three.
- Real images: `load_pgm` reads 8-bit PGM files, but there is no
  calibration step or real dataset yet.
- Lens distortion in `PinholeCamera`.
