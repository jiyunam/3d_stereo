# 3d_stereo

The task of 3D reconstruction from 2D images requires an estimation of the depth lost during the 
transition from 3D object to 2D image representation. One approach is to use feature matching among 
stereo images. We compare the Scale Invariant Feature Transform (SIFT), and the Speeded-Up Robust 
Feature (SURF). These techniques are applied to two phases of the 3D depth extraction task. 
The first is in approximating the fundamental matrix used to rectify general stereo images to parallel 
stereo images. The second phase is in evaluating the feature extraction method against the 3D point cloud
generation from parallel stereo data. From this comparison, we deduce that SURF is oftentimes preferable 
to SIFT in its robustness in both the general to parallel stereo task, and the 3D point cloud reconstruction. 
Further significant observations, such as the importance of outlier detection in the approximation of the 
fundamental matrix are also noted.

## Important Notebooks
`normalized_8_points.ipynb`: Performs the normalized 8 point algorithm using different combinations of feature and outlier extraction methods.
To run, open notebook and set the image filepaths for the image to apply the algorithm to. By running the cells
that specify the featuer extraction and outlier removing cells, we can apply different methods. Also includes 
plotting functions for the epilines and the rectification.

`parallel_stereo.ipynb`: Performs the stereo image 3D point cloud extraction (phase 2)
Open in colab and include the Imgs.zip file as well. Then run all. To rerun for different images, change the 
parameter in run_folder(0) to be the index matching the file in pic_list that you wish to run. You can also alter 
the match_method in the run() command inside run_folder() to be sift or surf to check each feature selection method.