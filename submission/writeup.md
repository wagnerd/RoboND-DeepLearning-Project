# Writeup for Udacity Project "Follow Me"

## What to include in the submission

The submission must include these items:

1. The ```model_training.ipynb``` notebook that you have filled out.
2. A HTML version of the ```model_training.ipynb``` notebook.
3. This writeup report (as md or pdf file)
4. The model and weights file in the ```.h5``` file format

The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.

## My submission

1. Filled out  ```model_training.ipynb``` [here](run/model_training.ipynb)
2. A HTML version of the ```model_training.ipynb``` notebook [here](run/model_training.html)
3. This writeup, see below
4. The model and weights file in the ```.h5``` file format [here](run/data/weights/model_weights.h5)

The neural network had a final score of __0.412932498048__ as you can see in the notebook.

## Network architecture

The architecture of the network is based on a Fully Convolutional Network (FCN).
One of the main advantages of the FCN is the ability to retain spatial information.
The FCN is made up of atleast three blocks:

1. Encoder Block
2. 1x1 Convolution Block
3. Decoder Block

In this solution, three encoder/decoder blocks were used. This leads to the following architecture:

![architecture](fcn_architecture.png)

### Encoder Block

The encoder block is used to extract the main features from an image.
Each layer of the encoder used in this project consists of a separable convolutional layer and a batch normalization with an activation function included.

Each layer of the encoder has a clear function. The separable convolutional layer reduces the number of parameters of the network and identifies the main features (extracting spatial information).
The batch normalization layer, normalizes the inputs to each layer within the network to have a well conditioned problem.
The used RELU activation function adds non-linearities to the network for it to better fit the model.

The implemented code for the encoder looks like this:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

### 1x1 Convolution Block

Between the decoder block and the encoder block, there is a 1x1 convolution layer which increases the depth of the network while preserving spatial information.
This has also the advantage that during inference we can feed images of any given size into the trained network. The alternative would be a fully-connected layer, that keeps the number of features the same but needs a fixed image size.

In code the 1x1 convolutional block is implemtened as below:

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```

### Decoder Block

The decoder consists of:

* A bilinear upsampling layer  
 This layer expands the dimensions of the encoded information (the objective is to reach the size of the original image)
* A concatenation layer  
 This layer is similar to skipping the connections, which is a technique that allows one layer to use information from different resolution scales (from layers prior to the preceding one)
* A separable convolutional layer  

The code for the decoder looks like this:

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # Upsample the small input layer using the bilinear_upsample() function.
    small_ip_layer_upsampled = bilinear_upsample(small_ip_layer)
    # Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([small_ip_layer_upsampled, large_ip_layer])
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    return output_layer
```

### Model

Finally the complete models as implemented code:

```python
def fcn_model(inputs, num_classes):
    # Encoder Blocks. 
    encoded_1 = encoder_block(inputs, filters=32, strides=2)
    encoded_2 = encoder_block(encoded_1, filters=64, strides=2)
    encoded_3 = encoder_block(encoded_2, filters=128, strides=2)
    
    # 1x1 Convolution layer using conv2d_batchnorm().
    one_by_one = conv2d_batchnorm(encoded_3, filters=256, kernel_size=1, strides=1)
    
    # Same number of Decoder Blocks as the number of Encoder Blocks
    decoded_1 = decoder_block(small_ip_layer=one_by_one, large_ip_layer=encoded_2, filters=128)
    decoded_2 = decoder_block(small_ip_layer=decoded_1, large_ip_layer=encoded_1, filters=64)
    decoded_3 = decoder_block(small_ip_layer=decoded_2, large_ip_layer=inputs, filters=32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, kernel_size=1, activation='softmax', padding='same')(decoded_3)
```

The values of the filters were chosen based on the input image shape of 160x160x3.

## Hyper Parameters

The hyper parameters for the FCN are defined as follows:

* __learning_rate:__  
 controls how much the weights of our network are adjusted with respect to the loss gradient
* __batch_size:__  
  number of training samples/images that get propagated through the network in a single pass
* __num_epochs:__  
  number of times the entire training dataset gets propagated through the network
* __steps_per_epoch:__  
  number of batches of training images that go through the network in 1 epoch
* __validation_steps:__  
  number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset.
* __workers:__  
  maximum number of processes to spin up. This can affect the training speed and is dependent on the hardware.
* __stride:__  
  The amount by which the filter slides is referred to as the 'stride'. Increasing the stride reduces the size of your model by reducing the number of total patches each layer observes. This usually comes with a reduction in accuracy.

The final parameters for the used model are:

```python
learning_rate = 0.0015
batch_size = 25
num_epochs = 50
steps_per_epoch = 4131/batch_size
validation_steps = 50
workers = 2
```

The values for ```validation_steps``` and ```workers``` have not
been changed and are still the default values from the notebook.
The value for ```steps_per_epoch``` was set to the hint given in the comments where 4131 is the number of images provided.

For the other parameters, I have chosen the parameters based on my experiments in the segmentation lab.

Testing the optimal number of epochs, I tried various values from 10, 20 to 50. Every increase here
improved the IoU slightly. Finally I stayed with 50 epochs because the training already takes a long time now and improvements seem to become smaller.

The learning rate started at 0.05, which is relatively high, but showed a good basis in the segmentation lab. But for the final project this was too high.
The loss was fluctuating a lot. I decreased the learning rate to ```0.001``` and also tried ```0.01```. In the end I stayed on ```0.0015```, because here was the smoothest learning curve in the plots.

### Various follow-me scenarios

The network is currently based on the standard architecture, so there shouldn't be any big changes. Regarding to reuse it for other "follow me" scenarios, we would need to train it with the corresponding dataset and evaluate the new hyper parameters, like we did here.

## Future Enhancements

1. Collect more data to train the model. This could easily be done by investing more time.
2. Optimize the hyper parameters. Instead of guessing and running each step by
   hand, we could use an evolutionary algorithm to optimize the parameters for us. Since we already have the parameters and our fitness value (IoU), this should be
   reachable without big effort.
