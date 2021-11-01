## PointNet++: : Deep Hierarchical Feature Learning on Point Sets in a Metric Space

We run and test the pointnet++ model on ModelNet10 and ModelNet40 for classification, and on ShapeNet for segmentation. For the segmentation task, we use 8 out of the 16 object classes.

### Installation

We use the PointNet++ implementation available in [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) package. This requires installing the package before running the 2 scripts here:

```
python3 -m venv my_env
source my_env/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Next, install the binaries for PyTorch 1.10.0 by simply running:

```
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip3 install torch-geometric
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu113` depending on your PyTorch installation (`torch.version.cuda`).


### Instructions to run the code


To run classification on the ModelNet datasets and segmentation on the ShapeNet dataset, run the following commands from your virtual environment:

```
python3 classification.py --epochs 50 --dataset ModelNet10 --num_points 1024
python3 classification.py --epochs 50 --dataset ModelNet40 --num_points 1024
python3 segmentation.py --epochs 50
```

Modify the `--num_points` flag to change the total number of points in the point cloud. The points are uniformly sampled on the mesh faces according to their face area.
