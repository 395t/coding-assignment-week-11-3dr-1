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

For all of the following experiments, the model is trained for 30 epochs using SGD as the optimizer. A learning rate of 0.001 is used with a momentum of 0.9 and a dropout rate of 0.5. We keep the default ratio of 1024:20 for the number of points to number of nearest neighbors. We ablate the number of points.

### ModelNet10
|   | mAcc | OA |
|:-|:-:|:-:|
| 512  | 93.81 | 94.05 |
| 1024 | 93.68 | 93.83 |
| 2048 | 93.96 | 94.05 |

![mAccModelNet10](https://user-images.githubusercontent.com/34489261/139623587-f4436351-2047-4911-aec0-e9ffae5cb713.png)
![OAModelNet10](https://user-images.githubusercontent.com/34489261/139623595-ed633f06-4f88-48f6-8b46-9e9ff3c4ad6e.png)

### ShapeNet
|   | mAcc | OA |
|:-|:-:|:-:|
| 512  | 74.98 | 92.56 |
| 1024 | 77.85 | 93.58 |
| 2048 | 76.11 | 91.82 |

![mAccShapeNet](https://user-images.githubusercontent.com/34489261/139623613-b7fa1588-7124-4248-9018-d3bbacc35ca2.png)
![OAShapeNet](https://user-images.githubusercontent.com/34489261/139623618-9b4cec9f-b5b0-464c-b024-57aa32a32d2d.png)

### Segmentation Task Example
![snapshot00](https://user-images.githubusercontent.com/34489261/139623648-fd8af807-423c-41bf-8e4a-d3ff81db7876.png)

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
|  2e-4 |  |   |
|  5e-3 |   |   |


### ModelNet10

Following ModelNet40, we train the model for 20 epochs, and the default hyperparameters are unchanged. Using the default hyperparameters, we get 66.08 on mAcc and 57.85 on OA. (We can only train for 45 epochs due to resource constraints.)

We again ablate number of points used and the learning rate. 

#### Number of Points
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1024 (45 epochs) |   70.59 | 65.67  |
|  1024 (20 epochs) |  67.84	| 63.32  |
|  512 |   |   |
|  256 |  |   |

![image](https://user-images.githubusercontent.com/35536646/139623357-490a96c9-6cf2-45dc-b588-316565407356.png)


#### Learning Rate 
|   | mAcc  | OA  |
|:-|:-:|:-:|
|  1e-3 (20 epochs) |  67.84	| 63.32|
|  2e-4 | 91.85  | 91.36  |
|  5e-3 |  10.96 | 10.00  |


### ShapeNet

For ShapeNet, we train the model for 40 epochs, and the default hyperparameters follow that of modelNet40 and 10. Using the default hyperparameters, we get 66.08 on mAcc and 57.85 on OA.

We again ablate number of points used and the learning rate. 

#### Number of Points
|   | Acc  | meanIOU  | InstanceIOU | 
|:-|:-:|:-:|:-:|
|  1024 (40 epochs) |  90.97	|  72.29 |	79.64   |
|  1024 (20 epochs) | 87.32  | 64.79  | 73.98 |
|  512 |   |   |  |
|  256 | 88.69 | 67.99  | 76.05 |

![image](https://user-images.githubusercontent.com/35536646/139623380-4ac0ee5d-f29e-4bc8-9816-9339f5884bb0.png)

![image](https://user-images.githubusercontent.com/35536646/139623398-189f2d9e-fa46-4055-ac0a-2025573d8658.png)


#### Learning Rate 
|   | Acc  | meanIOU  | InstanceIOU | 
|:-|:-:|:-:|:-:|
|  1e-3 (20 epochs) | 87.32  | 64.79  | 73.98 |
|  2e-4 |   |   |  |
|  5e-3 |   |  |  |
