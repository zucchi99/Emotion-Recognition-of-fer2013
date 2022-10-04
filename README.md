### Emotion Recognition of fer2013

## Dataset Augmentation

From analysis of the dataset emerges that it is small, only 35887 images, and not uniformly distributed, happiness has 8989 samples meanwhile disgust only 547.
This data is retrieved from the first blocks of the <code>fer2013_augmenter.ipynb</code> file.

So, we decided to enlarge the database using some transformation on the dataset.
The size of original dataset is almost 290 MB.
Each transformation increase the size of 290 MB more or less.
The transformation applied are the following:
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

So the final size is 14 * 290 MB = 4 GB (original images included).
It is possible to create a dataset with only some of these filters selecting only the desidered.
The transformation class is inside the <code>fer2013_augmenter.py</code> and is invoked in the notebook.

The filters now are implemented in three possible ways:
 * using a lambda function: (x, y, pixel) -> pixel
  - used for altering the contrast
 * using a filter matrix 

