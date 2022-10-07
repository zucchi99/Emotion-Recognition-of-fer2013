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
If the size is too much large, it is very easy to create a smaller dataset, by editing the notebook and selecting only the some of the filters.

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
    * (for lambda/matrix) : add the lambda/matrix to the list of lambdas/matrixes 
    * (for custom functions) : add the function and edit the also the <code>generate_all_filters</code> function adding the function call
 3. initialize and execute the class

## Emotion Recognition

The image recognition is done by a Convolutional Neural Network, using <code>PyTorch</code>.
The creation of the CNN is done by the classes <code>DynamicNetBasic</code> and <code>DynamicNetInceptions</code>, which are subclasses of the class <code>torch.nn.Module</code>.
Both the classes allow to create dynamic nets (with a variable number of layers).
The class constructor allows to try many different nets by simply changing few parameters.

Structure of the CNNs:
<table>
 <tr>
    <td><b style="font-size:30px">DynamicNetBasic</b></td>
    <td><b style="font-size:30px">DynamicNetInceptions</b></td>
 </tr>
 <tr>
    <td>
        <ol>
            <li> List( List(<i>C-Block</i>), <i>MaxPool2D</i> )
            <li> <i>DropOut</i>
            <li> List( <i>Linear</i> )
            <li> <i>SoftMax</i>
        </ol>
        <br/><br/>
    </td>
    <td>
        <ol>
            <li> <i>List( List( C-Block ), MaxPool2D )</i>
            <li> <i>DropOut</i>
            <li> <i>List( Inception-Block )</i>
            <li> <i>DropOut</i>
            <li> <i>List( Linear )</i>
            <li> <i>SoftMax</i>
        </ol>
    </td>
 </tr>
</table>

The C-Block (convolutional-block) is formed by a <i>Conv2D</i>, a <i>DropOut</i> (optionally), and a <i>ReLU</i>:
<table>
   <tr>
      <td><b style="font-size:30px">Conv-Drop-ReLU</b></td>
      <td><b style="font-size:30px">Conv-ReLU</b></td>
   </tr>
   <tr>
      <td>
         <img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/ConvDrop-Block.png" height="200" alt="Conv-Drop-Block">
      </td>
      <td>
         <img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/Conv-Block.png" height="200" alt="Conv-Block">
      </td>
   </tr>
</table>

So, for both the classes, the full view of the first point of the structure is the following:

<img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/SequenceOfC-Block.png" height="250" alt="SequenceOfC-Block">

### Class DynamicNetBasic

The class <i>DynamicNetBasic</i> has a linear structure and has the following parameters (divided by which step are used):

 1. <i>List( List( Conv-Drop-Block ), MaxPool2D )</i>:
   * <i>integer</i> <code>conv__in_channels</code>: number of channels in input (the number of filters used).
   * <i>tuple of integer</i> <code>conv__out_channels</code>: each element represents the number of channels in output for all che Conv2d inside the inner list. 
      * NB: Tipically you want always to increase the number of channels in the convolutional part
   * <i>tuple of integer</i> <code>conv__layer_repetitions</code>: each element represents the number of times each inner list must be repeated before the <i>MaxPool2D</i>. 
      * NB1: From the second Conv2D the number of $in{\textunderscore}channel$ will be obviously same as $out{\textunderscore}channels$.
      * NB2: since the class is dynamic the two tuples can have any length, but must be same for both.
 2. <i>DropOut</i>:
   * <i>double</i> <code>dropout_prob</code>: percentage of dropout probability
 3. <i>List( Linear )</i>: 
   * <i>tuple of integer</i> <code>lin__out_dimension</code>: each element represents the number of features in output. The last element must have same value $7 = len(emotions)$, so that each value of the final array will be the probability of the i-th emotion.
      * NB: Tipically you want always to decrease the number of channels in the linear part
 4. <i>SoftMax</i>: no parameters

So, for example, this would be produce well performing -but huge- model:<br/>
$dropout{\textunderscore}prob = 0.35$<br/>
$conv{\textunderscore}{\textunderscore}in{\textunderscore}channels = len(filters{\textunderscore}used)$<br/>
$conv{\textunderscore}{\textunderscore}out{\textunderscore}channels =      (288, 566, 1122, 2244)$<br/>
$conv{\textunderscore}{\textunderscore}layer{\textunderscore}repetitions = (  4,   3,    2,    1)$<br/>
$lin{\textunderscore}{\textunderscore}out{\textunderscore}dimension = (1024, 356, 158, 64, len(emotions))$

### Class DynamicNetInceptions

As a reminder, an *Inception-Block* is the following (developed by Google):

<img src="https://github.com/zucchi99/Emotion-Recognition-of-fer2013/blob/master/Images/Inception-Block.png" height="300" alt="Inception-Block">

The class <i>DynamicNetBasic</i> has a linear structure and has the following parameters (divided by which step are used):

