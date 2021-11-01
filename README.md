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

### ShapeNet40

We report the mean accuracy within each category (mAcc) and the overall accuracy (OA) in all instances.

For each experiment, we train the model for 20 epochs (except for the default setting). For the default setting, the model is trained for 150 epochs using Adam optimizer with 0.001 learning rate and batch size of 8. The number of points is 1024. Using these hyperparameters, we get 66.08 on mAcc and 57.85 on OA.

We also ablate 

|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1024 (150 epochs) | 66.08  | 57.85  |
|  1024 (20 epochs) |  49.71 |  39.72 |
|  512 | 32.29  |  21.11 |
|  256 |  30.34 | 20.72  |

![image](https://user-images.githubusercontent.com/35536646/139611980-ed02b7ba-7771-4976-8bea-b82ce67f3737.png)
