### Emotion Recognition of fer2013

## Dataset Augmentation

From analysis of the dataset emerges that it is small, only 35887 images, and not uniformly distributed, happiness has 8989 samples meanwhile disgust only 547.
This data is retrieved from the first blocks of the <code>fer2013_augmenter.ipynb</code> file.

So, we decided to enlarge the database using some transformation on the dataset.
The size of original dataset is almost 290 MB.
Each transformation increase the size of 290 MB more or less.
The transformation applied are the following:
 1. <code>SOBEL</code>
 * <code>VERTICAL</code>
 * <code>HORIZONTAL</code>
 * <code>CONTRAST_LOW</code>
 * <code>CONTRAST_HIGH</code>
 * <code>CONTRAST_VERY_HIGH</code>
 * <code>FLIP_HORIZONTAL</code>
 * <code>ROT_LEFT_60_DEGREES</code>
 * <code>ROT_LEFT_40_DEGREES</code>
 * <code>ROT_LEFT_20_DEGREES</code>
 * <code>ROT_RIGHT_20_DEGREES</code>
 * <code>ROT_RIGHT_40_DEGREES</code>
 * <code>ROT_RIGHT_60_DEGREES</code>


