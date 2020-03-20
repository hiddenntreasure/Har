### Har
# Human Activity Recognition
## Abstract
Machine Learning model applied to pre engineered feature(561).But initial dataset have only 9 inertial signal all in 128 dimensions. Those was getting from accelerometer and gyroscope. Deep Learing model will be applied. As deep learning model can develop feature by itself.We could easily used 9 inertial signal.

But as dataset is divided into 9 different txt file.Dataset need to rearrange to apply desired model.

For Example: "body_acc_x" are 128 dimensions/128 timestamps with 7352 windows. Besides 9 signals.

Used a ``` list=[]``` and append 9 different signal to the list.Apply ``` as_matrix ``` to convert each inertial signal to flat matrix before appending. Thats produce (9,7352,128) matrix.And then apply np.transpose(list,(x,y,z)) to rearrange dimensions (x,y,z),3d matrix.

```pd.get_dummies('abca')```

|  | a| b| c|
|--|--|--|--|
   
| 0| 1| 0| 0|

| 1| 0| 1| 0|

| 2| 0| 0| 1|

| 3| 1| 0| 0|

```np.random.seed(n)``` is used to create random numbers.which doesn't change every time re run the program. opposite: ```np.random.rand(n)```
