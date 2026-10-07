# Camera Geometry And Pose Estimation

First steps on the SLAM branch: the pinhole camera model and the two pose
problems every visual SLAM system is built on. Everything is pure Python and
runs on a synthetic scene, so results can be checked against exact ground
truth.

```bash
python3 pose_demo.py
python3 pose_demo.py --noise 2.0 --points 60 --seed 3
python3 -m unittest test_linalg test_geometry
```

## Files

| File | What it holds |
| --- | --- |
| `linalg.py` | list-of-lists matrices, Gaussian elimination, Jacobi eigen solver, 3x3 SVD, nearest rotation |
| `geometry.py` | rotations (Rodrigues), `Pose`, `PinholeCamera`, triangulation, PnP, essential matrix, LM refinement |
| `pose_demo.py` | synthetic two-camera scene and reports against ground truth |
| `engine.py` | scalar autograd, now with `sin`, `cos`, `sqrt`, and `gradients()` for Jacobians |

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

**Rotation vectors.** A rotation is stored as a 3x3 matrix and
optimized as a 3-vector `w` (axis times angle) through Rodrigues' formula
`R = I + A [w]x + B [w]x^2`. Near `w = 0`, `A` and `B` switch to their
Taylor series in `theta^2`, which keeps autograd finite at exactly the point
where every refinement step starts.

**Triangulation (DLT).** Each view of a point gives two equations linear
in the homogeneous point: `x P_3 - P_1 = 0` and `y P_3 - P_2 = 0`. Stack them
for both views and take the null vector.

**Absolute pose, PnP (DLT).** The same two equations per point, now
linear in the 12 entries of `P = [R | t]`. The null vector gives `P` up to
scale; the left 3x3 block is `s R`, so divide out `s` (mean singular value)
and snap the block to the nearest rotation with an SVD.

**Relative pose (eight-point algorithm).** Matches satisfy
`x2^T E x1 = 0` with `E = [t]x R`. Eight or more matches give `E` up to
scale; project it to two equal singular values and one zero, then pick the
one of four `(R, +/-t)` factorizations that puts triangulated points in front
of both cameras. The translation only has a direction: two views cannot
tell the scale of the scene.

**Refinement (Levenberg-Marquardt).** Linear estimates minimize an
algebraic error, not pixel error. `refine_pose` minimizes pixel reprojection
error directly:

1. parameterize a small update `delta = (w, dt)` around the current pose
2. build every pixel residual from `Value` scalars
3. read the Jacobian with `engine.gradients`, one residual at a time
4. solve `(J^T J + lambda diag(J^T J)) step = -J^T r` and adapt `lambda`

This is motion-only bundle adjustment, the same problem ORB-SLAM3 solves
for every tracked frame. Here autograd replaces hand-derived Jacobians.

**Linear algebra.** `symmetric_eigen` is the Jacobi method: rotate away
the off-diagonal entries one plane at a time until the matrix is diagonal.
Null vectors are the smallest eigenvector of `A^T A`, and the 3x3 SVD is
built from the eigen decomposition of `M^T M`.

## Results On The Default Scene

40 points 4 to 8 units deep, a 640x480 camera with `f = 500`, 0.5 px
Gaussian pixel noise, second camera moved about 0.8 units sideways and
turned about 9 degrees (`python3 pose_demo.py`):

| Step | Rotation error | Translation error | Reprojection RMSE |
| --- | --- | --- | --- |
| PnP, linear DLT | 0.111 deg | 0.0093 | 0.766 px |
| PnP, after LM (converges in 1-2 iterations) | 0.074 deg | 0.0077 | 0.616 px |
| ground-truth pose, same noisy pixels | 0 | 0 | 0.647 px |
| Two-view, eight-point + cheirality | 0.417 deg | 3.8 deg (direction) | - |

LM ends slightly *below* the ground-truth reprojection error. That is
expected: the refined pose fits this particular noise draw, which is exactly
what least squares does. With `--noise 0`, every estimator is exact to about
1e-9.

Over 8 seeds, the median eight-point rotation error grows linearly with
noise: 0.26, 0.53, and 1.13 degrees at 0.25, 0.5, and 1.0 px. The baseline
(0.8) is small compared to the scene depth (about 6), so two-view geometry
is weakly constrained. That is the low-parallax regime monocular SLAM
initializers try to avoid.

## How This Maps To The Reference Points

These references stay distant: no feature detection, no real images, no
mapping. But the math above is their core.

| Building block | ORB-SLAM3 | DROID-SLAM |
| --- | --- | --- |
| pinhole projection, `Pose` | camera models, `Sophus::SE3` poses | same, as tensors |
| two-view initialization | essential/homography + cheirality | learned flow, no explicit E |
| frame tracking | PnP + motion-only BA (this file's LM) | dense BA layer on flow |
| Jacobians | hand-derived in g2o | autograd through a differentiable BA |
| bundle adjustment | local/global BA over points and poses | differentiable dense BA |

## Next Gaps

- Two-view bundle adjustment: refine both the relative pose and the
  triangulated points (the same LM loop with more parameters).
- RANSAC around eight-point and PnP to survive outlier matches.
- Real images: corner detection plus patch matching on a small image pair,
  reusing `conv2d_valid` for gradients.
- Lens distortion in `PinholeCamera`.
