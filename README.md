### Emotion Recognition of fer2013

## Dataset Augmentation

Input file: <code>fer2013.csv</code>
Output file: <code>fer2013_augmented.csv</code>

From analysis of the dataset emerges that it is small, only 35887 images, and not uniformly distributed: happiness has 8989 samples meanwhile disgust only 547.<br/>
The analysis and the augmentation is done in the <code>fer2013_augmenter.ipynb</code> file.<br/>
The transformation class is implemented inside the <code>fer2013_augmenter.py</code> and is invoked by the notebook.

We decided to enlarge the database using some transformations on the images.<br/>
The size of original dataset is almost 290 MB.<br/>
Each transformation increase the size of 290 MB more or less.<br/>
The transformation applied are the following, defined in the class <code>Filters(Enum)</code>:
 1. <code>SOBEL</code>
 2. <code>VERTICAL</code>
 3. <code>HORIZONTAL</code>
 4. <code>CONTRAST_LOW</code>
 5. <code>CONTRAST_HIGH</code>
 6. <code>CONTRAST_VERY_HIGH</code>
 7. <code>FLIP_HORIZONTAL</code>
 8. <code>ROT_LEFT_60_DEGREES</code>
 9. <code>ROT_LEFT_40_DEGREES</code>
 10. <code>ROT_LEFT_20_DEGREES</code>
 11. <code>ROT_RIGHT_20_DEGREES</code>
 12. <code>ROT_RIGHT_40_DEGREES</code>
 13. <code>ROT_RIGHT_60_DEGREES</code>

The final size is 14 * 290 MB = 4 GB (original images included).
Editing the notebook and selecting only the desidered, it is possible to create a dataset with only some of these filters.

The filters now are implemented in three possible ways:
 * using a lambda function: (x, y, pixel) -> pixel
    * used for altering the contrast
 * using a filter matrix in a function: (image, filter) -> image
    * used for vertical and horizontal filters
 * using custom function: (image, custom_parameter) -> image
    * used for sobel filter, flipping, rotations

Before applying the transformation filter-based, the image is padded to avoid images with lower dimensions.<br/>
Currently it is not supported the application of a stride.

If you want to add any type of transformation you just need to:
 1. add the transformation name to Filters(Enum)
 2. (for lambda/filter) : add a lambda/filter to the list of lambdas/filters 
 2. (for custom functions) : add the function and edit the also the <code>generate_all_filters</code> function adding the function call
 3. initialize and execute the class

## Emotion Recognition

