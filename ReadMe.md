### Har
# Human Activity Recognition
## Dataset Description
Machine Learning model applied to pre engineered feature(561).But initial dataset have only 9 inertial signal all in 128 dimensions. Those was getting from accelerometer and gyroscope. Deep Learing model will be applied. As deep learning model can develop feature by itself.We could easily used 9 inertial signal.

But as dataset is divided into 9 different txt file.Dataset need to rearrange to apply desired model.

For Example: "body_acc_x" are 128 dimensions/128 timestamps with 7352 windows. Besides 9 signals.
## Loading Data

Used a ``` list=[]``` and append 9 different signal to the list.Apply ``` as_matrix ``` to convert each inertial signal to flat matrix before appending. Thats produce (9,7352,128) matrix.And then apply np.transpose(list,(x,y,z)) to rearrange dimensions (x,y,z),3d matrix.

```pd.get_dummies('abca')```

|  | a| b| c|
|--|--|--|--|  
| 0| 1| 0| 0|
| 1| 0| 1| 0|
| 2| 0| 0| 1|
| 3| 1| 0| 0|

## Introducing Tensorflow and Keras

```np.random.seed(n)``` is used to create random numbers.which doesn't change every time re run the program. opposite: ```np.random.rand(n)```

```tf.ConfigProto``` 

TensorFlow CPUs and GPUs Configuration[above code](https://medium.com/@liyin2015/tensorflow-cpus-and-gpus-configuration-9c223436d4ef)

```np.random.seed(42)```

```import tensorflow as tf```

```tf.set_random_seed(42)```

To understand the above code[click here ](https://github.com/tensorflow/tensorflow/issues/29101)

## LSTM Parameters

```model.add(LSTM(n_hidden,input_shape=(timesteps,input_dim)))```

**Input shape**

2D tensor with shape ```(timesteps, input_dim)```


**units/hidden layers**: Positive integer, dimensionality of the output space.

### Output layers

```model.add(Dense(n_classes,activation='sigmoid'))```


