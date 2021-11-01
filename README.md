# 3D Recognition

## Code Structure

## Tasks and Datasets

### Classification

### Segmentation

### ModelNet 10

### ModelNet 100

### ShapeNet

## PointNet

## PointNet++

## Dynamic Graph CNN

## PointCNN

## Point Transformer

### ModelNet40

We report the mean accuracy within each category (mAcc) and the overall accuracy (OA) in all instances.

For each experiment, we train the model for 20 epochs (except for the default setting). For the default setting, the model is trained for 150 epochs using Adam optimizer with 0.001 learning rate and batch size of 8. The number of points is 1024. Using these hyperparameters, we get 66.08 on mAcc and 57.85 on OA.

We also ablate some of the hyperparameters, including number of points used and the learning rate. 

#### Number of Points
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1024 (150 epochs) | 66.08  | 57.85  |
|  1024 (20 epochs) |  49.71 |  39.72 |
|  512 | 32.29  |  21.11 |
|  256 |  30.34 | 20.72  |

![image](https://user-images.githubusercontent.com/35536646/139611980-ed02b7ba-7771-4976-8bea-b82ce67f3737.png)

#### Learning Rate 
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1e-3 (150 epochs) | 66.08  | 57.85  |
|  1e-3 (20 epochs) |  49.71 |  39.72 |
|  2e-4 | 32.29  |  21.11 |
|  5e-3 |  30.34 | 20.72  |


### ModelNet10

Following ModelNet40, we train the model for 20 epochs, and the default hyperparameters are unchanged. Using the default hyperparameters, we get 66.08 on mAcc and 57.85 on OA.

We again ablate number of points used and the learning rate. 

#### Number of Points
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1024 (150 epochs) |   |   |
|  1024 (20 epochs) |   |   |
|  512 |   |   |
|  256 |  |   |



#### Learning Rate 
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1e-3 (150 epochs) | 66.08  | 57.85  |
|  1e-3 (20 epochs) |  49.71 |  39.72 |
|  2e-4 | 32.29  |  21.11 |
|  5e-3 |  30.34 | 20.72  |


### ShapeNet

For ShapeNet, we train the model for 40 epochs, and the default hyperparameters follow that of modelNet40 and 10. Using the default hyperparameters, we get 66.08 on mAcc and 57.85 on OA.

We again ablate number of points used and the learning rate. 

#### Number of Points
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1024 (150 epochs) |   |   |
|  1024 (20 epochs) |   |   |
|  512 |   |   |
|  256 |  |   |


#### Learning Rate 
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1e-3 (150 epochs) | 66.08  | 57.85  |
|  1e-3 (20 epochs) |  49.71 |  39.72 |
|  2e-4 | 32.29  |  21.11 |
|  5e-3 |  30.34 | 20.72  |
