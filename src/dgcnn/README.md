# DGCNN.pytorch
## Contents
- [Point Cloud Classification](#point-cloud-classification)
- [Point Cloud Part Segmentation](#point-cloud-part-segmentation)

**Note:** If a CUDA out-of-memory error is encountered, reducing the batch size or test batch size may help.
**Note:** All following commands default use all GPU cards. To specify the cards to use, add `CUDA_VISIBLE_DEVICES=0,1,2,3` before each command, where the user uses 4 GPU cards with card index `0,1,2,3`. You can change the card number and indexes depending on your own needs.

&nbsp;
## Point Cloud Classification
### Run the training script:

- 1024 points

``` 
python main_cls.py --exp_name=cls_1024 --num_points=1024 --k=20 
```

- 2048 points

``` 
python main_cls.py --exp_name=cls_2048 --num_points=2048 --k=40 
```

### Run the evaluation script after training finished:

- 1024 points

``` 
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/cls_1024/models/model.t7
```

- 2048 points

``` 
python main_cls.py --exp_name=cls_2048_eval --num_points=2048 --k=40 --eval=True --model_path=outputs/cls_2048/models/model.t7
```

### Run the evaluation script with pretrained models:

- 1024 points

``` 
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=pretrained/model.cls.1024.t7
```

- 2048 points

``` 
python main_cls.py --exp_name=cls_2048_eval --num_points=2048 --k=40 --eval=True --model_path=pretrained/model.cls.2048.t7
```

### Performance:
ModelNet40 dataset

|  | Mean Class Acc | Overall Acc |
| :---: | :---: | :---: |
| Paper (1024 points) | 90.2 | 92.9 |
| This repo (1024 points) | **90.9** | **93.3** |
| Paper (2048 points) | 90.7 | 93.5 |
| This repo (2048 points) | **91.2** | **93.6** |

&nbsp;
## Point Cloud Part Segmentation
**Note:** The training modes **'full dataset'** and **'with class choice'** are different. 

- In **'full dataset'**, the model is trained and evaluated in all 16 classes and outputs mIoU 85.2% in this repo. The prediction of points in each shape can be any part of all 16 classes.
- In **'with class choice'**, the model is trained and evaluated in one class, for example airplane, and outputs mIoU 84.5% for airplane in this repo. The prediction of points in each shape can only be one of the parts in this chosen class.

### Run the training script:

- Full dataset

``` 
python main_partseg.py --exp_name=partseg 
```

- With class choice, for example airplane 

``` 
python main_partseg.py --exp_name=partseg_airplane --class_choice=airplane
```

### Run the evaluation script after training finished:

- Full dataset

```
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7
```

- With class choice, for example airplane 

```
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7
```

### Run the evaluation script with pretrained models:

- Full dataset

``` 
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=pretrained/model.partseg.t7
```

- With class choice, for example airplane 

``` 
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=pretrained/model.partseg.airplane.t7
```

### Performance:

ShapeNet part dataset

| | Mean IoU | Airplane | Bag | Cap | Car | Chair | Earphone | Guitar | Knife | Lamp | Laptop | Motor | Mug | Pistol | Rocket | Skateboard | Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Shapes | | 2690 | 76 | 55 | 898 | 3758 | 69 | 787 | 392 | 1547 | 451 | 202 | 184 | 283 | 66 | 152 | 5271 | 
| Paper | **85.2** | 84.0 | **83.4** | **86.7** | 77.8 | 90.6 | 74.7 | 91.2 | **87.5** | 82.8 | **95.7** | 66.3 | **94.9** | 81.1 | **63.5** | 74.5 | 82.6 |
| This repo | **85.2** | **84.5** | 80.3 | 84.7 | **79.8** | **91.1** | **76.8** | **92.0** | 87.3 | **83.8** | **95.7** | **69.6** | 94.3 | **83.7** | 51.5 | **76.1** | **82.8** |

### Visualization:
#### Usage:

Use `--visu` to control visualization file. 

- To visualize a single shape, for example the 0-th airplane (the shape index starts from 0), use `--visu=airplane_0`. 
- To visualize all shapes in a class, for example airplane, use `--visu=airplane`. 
- To visualize all shapes in all classes, use `--visu=all`. 

Use `--visu_format` to control visualization file format. 

- To output .txt file, use `--visu_format=txt`. 
- To output .ply format, use `--visu_format=ply`. 

Both .txt and .ply file can be loaded into [MeshLab](https://www.meshlab.net) for visualization. For the usage of MeshLab on .txt file, see issue [#8](https://github.com/AnTao97/dgcnn.pytorch/issues/8) for details. The .ply file can be directly loaded into MeshLab by dragging.

The visualization file name follows the format `shapename_pred_miou.FILE_TYPE` for prediction output or `shapename_gt.FILE_TYPE` for ground-truth, where `miou` shows the mIoU prediction for this shape.

#### Full dataset:

- Output the visualization file of the 0-th airplane with .ply format

```
# Use trained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7 --visu=airplane_0 --visu_format=ply

# Use pretrained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=pretrained/model.partseg.t7 --visu=airplane_0 --visu_format=ply
```

- Output the visualization files of all shapes in airplane class with .ply format

```
# Use trained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7 --visu=airplane --visu_format=ply

# Use pretrained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=pretrained/model.partseg.t7 --visu=airplane --visu_format=ply
```

- Output the visualization files of all shapes in all classes with .ply format

```
# Use trained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7 --visu=all --visu_format=ply

# Use pretrained model
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=pretrained/model.partseg.t7 --visu=all --visu_format=ply
```

#### With class choice, for example airplane:

- Output the visualization file of the 0-th airplane with .ply format

```
# Use trained model
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7 --visu=airplane_0 --visu_format=ply

# Use pretrained model
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=pretrained/model.partseg.airplane.t7 --visu=airplane_0 --visu_format=ply
```

- Output the visualization files of all shapes in airplane class with .ply format

```
# Use trained model
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7 --visu=airplane --visu_format=ply

# Use pretrained model
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=pretrained/model.partseg.airplane.t7 --visu=airplane --visu_format=ply
