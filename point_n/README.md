# Point-GN
A Non-Parametric Network Using Gaussian Positional Encoding for Point Cloud Classification

## Introduction

In this paper, we propose Point-GN, a non-parametric network designed for efficient and accurate 3D point cloud classification.

## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/Marzieh-Mohammadi/Point-GN.git
cd Point-GN

conda create -n pointgn python=3.7
conda activate pointgn

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit

pip install -r requirements.txt
pip install pointnet2_ops_lib/.
```

### Dataset
Please download the following datasets: [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), [ScanObjectNN](https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip), and [ShapeNetPart](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Then, create a `data/` folder and organize the datasets as
```
data/
|–– h5_files/
|–– modelnet40_ply_hdf5_2048/
|–– shapenetcore_partanno_segmentation_benchmark_v0_normal/
```

## Point-GN 
### Shape Classification

For ModelNet40 dataset, just run:
```bash
python run_nn_cls.py --dataset mn40
```

For ScanObjectNN dataset, just run:
```bash
python run_nn_cls.py --dataset scan --split 1
```
Please indicate the splits at `--split` by `1,2,3` for OBJ-BG, OBJ-ONLY, and PB-T50-RS, respectively.

### Part Segmentation
For ShapeNetPart, just run:
```bash
python run_nn_seg.py
```
You can increase the point number `--points` and k-NN neighbors `--k` into `2048` and `128`.

## Point-PN
### Shape Classification

Point-PN is the parametric version of Point-GN with efficient parameters and simple 3D operators.

For ModelNet40 dataset, just run:
```bash
python run_pn_mn40.py --msg <output filename>
```

For ScanObjectNN dataset, just run:
```bash
python run_pn_scan.py --split 1 --msg <output filename>
```
Please indicate the splits at `--split` by `1,2,3` for OBJ-BG, OBJ-ONLY, and PB-T50-RS, respectively.


## Citation
```
```

## Contact
If you have any question about this project, please contact mrziehmohamadi@gmail.com
