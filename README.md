# Emotion Recognition of fer2013

## Dataset Augmentation

Input file: <code>fer2013.csv</code><br/>
Output file: <code>fer2013_augmented.csv</code>

From analysis of the dataset emerges that it is:
 * **small**: has only 35887 images
 * **not uniformly distributed**: happiness has 8989 samples meanwhile disgust only 547.

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

The final size will be almost 14 * 290 MB = 4 GB (original images included).<br/>
Editing the notebook and selecting only the desidered, it is possible to create a dataset with only some of these filters.

The filters now are implemented in three possible ways:
 * using a **lambda function**: (x, y, pixel) -> pixel
    * used for altering the contrast
 * using a **filter matrix** in a function: (image, filter) -> image
    * used for vertical and horizontal filters
 * using **custom function**: (image, custom_parameter) -> image
    * used for sobel filter, flipping, rotations

Before applying the transformation filter-based, the image is padded to avoid images with lower dimensions.<br/>
Currently it is not supported the application of a stride.

If you want to add any type of transformation you just need to:
 1. add the transformation name to Filters(Enum)
 2. 
    * (for lambda/matrix) : add a lambda/matrix to the list of lambdas/matrixes 
    * (for custom functions) : add the function and edit the also the <code>generate_all_filters</code> function adding the function call
 3. initialize and execute the class

## Emotion Recognition

The image recognition is done by a Convolutional Neural Network.
The creation of the CNN is done by the classes <code>DynamicNetBasic</code> and <code>DynamicNetInceptions</code>.
Both the classes allow to create dynamic nets (with a variable number of layers).
The class constructor allows to try many different nets by simply changing few parameters.

Structure of the CNNs:
<table border="0">
 <tr>
    <td><b style="font-size:30px">DynamicNetBasic</b></td>
    <td><b style="font-size:30px">DynamicNetInceptions</b></td>
 </tr>
 <tr>
    <td>
        <ol>
            <li> List( List(<i>Conv-Drop-Block</i>), <i>MaxPool2D</i> )
            <li> <i>DropOut</i>
            <li> List( <i>Linear</i> )
            <li> <i>SoftMax</i>
        </ol>
        where a <i>Conv-Drop-Block</i> is formed by:
        <img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/ConvDrop-Block.png?raw=true" alt="Conv-Drop-Block">
    </td>
    <td>
        <ol>
            <li> List( List(<i>Conv-Basic-Block</i>), <i>MaxPool2D</i> )
            <li> <i>DropOut</i>
            <li> List( <i>Inception-Block</i> )
            <li> <i>DropOut</i>
            <li> List( <i>Linear</i> )
            <li> <i>SoftMax</i>
        </ol>
        where a <i>Conv-Basic-Block</i> is formed by:
        <img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/Conv-Block.png?raw=true" alt="Conv-Basic-Block">
    </td>
 </tr>
</table>

So, for both the classes, the full view of the first point of the structure is the following:

![alt text](https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/SequenceOfC-Block.png?raw=true "SequenceOfC-Block")

Tipically we repeated the whole structure (the outer list) $3 \le m \le 5$ times, every time increasing the number of channels $N_0 \rightarrow N_1 \rightarrow ... \rightarrow N_m$. Obviously $N_0$ is the number of filters applied to the dataset, $N_0 = 1$ if only original images are used. <br/>

For each step, the i-th element of the List(<i>Conv-Block</i>), three parameters are needed: 
 * $in{\textunderscore}channel=N_i$, 
 * $out{\textunderscore}channel=N_{i+1}$, 
 * $layer{\textunderscore}repetition$.


As a reminder, an *Inception-Block* is the following (developed by Google):

![alt text](https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/Inception-Block.png?raw=true "Inception-Block")

