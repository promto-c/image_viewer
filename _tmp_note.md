Here's a list of some of the popular ones with their initialization and key arguments:

1. **SIFT (Scale-Invariant Feature Transform)**:
   ```python
   sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
   ```

2. **SURF (Speeded-Up Robust Features)**:
   ```python
   surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False)
   ```

3. **ORB (Oriented FAST and Rotated BRIEF)**:
   ```python
   orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20)
   ```

4. **FAST (Features from Accelerated Segment Test)**:
   ```python
   fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv2.FastFeatureDetector.TYPE_9_16)
   ```

5. **BRIEF (Binary Robust Independent Elementary Features)**:
   ```python
   brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
   ```

6. **AKAZE (Accelerated-KAZE)**:
   ```python
   akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
   ```

7. **BRISK (Binary Robust Invariant Scalable Keypoints)**:
   ```python
   brisk = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
   ```

8. **KAZE**:
   ```python
   kaze = cv2.KAZE_create(extended=False, upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
   ```

9. **MSER (Maximally Stable Extremal Regions)**:
   ```python
   mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=14400, _max_variation=0.25, _min_diversity=.2, _max_evolution=200, _area_threshold=1.01, _min_margin=0.003, _edge_blur_size=5)
   ```

10. **AGAST (Accelerated AGAST)**:
    ```python
    agast = cv2.AgastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv2.AgastFeatureDetector.OAST_9_16)
    ```

11. **GFTT (Good Features to Track)**:
    ```python
    gftt = cv2.GFTTDetector_create(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
    ```

12. **SimpleBlob**:
    ```python
    simple_blob = cv2.SimpleBlobDetector_create()
    ```

Note that some algorithms, such as SIFT and SURF, were patented and are now available in the `xfeatures2d` module of OpenCV (requiring the OpenCV contrib modules). Depending on the version of OpenCV you're using and how it was compiled, some of these algorithms might not be available.

To use any of these algorithms, you typically follow the pattern:

```python
keypoints = algorithm.detect(image, None)
keypoints, descriptors = algorithm.compute(image, keypoints)
```

Or use the combined method:

```python
keypoints, descriptors = algorithm.detectAndCompute(image, None)
```

Each algorithm has its strengths and weaknesses and is suited for different types of images and requirements. You might want to experiment with each one to see which is best for your needs.