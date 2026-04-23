# SCAP: Semantic Prototype Alignment for Robust Point Cloud Registration

> **📢 News (2026-02-12):** Our paper has been accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**!


## 🚀 Dataset Environment

Our model is trained with the following environment (recommended for full reproducibility):

- **Python** 3.8.8  
- **PyTorch** 1.9.1 with torchvision 0.10.1 (CUDA 11.1)  
- **PyTorch3D** 0.6.0  
- **MinkowskiEngine** 0.5.4  

Other required packages can be installed via:

```bash
pip install -r src/requirements.txt
Hardware recommendation:
We used an NVIDIA Titan RTX (24 GB) for training. An RTX 3090 or A6000 is also suitable.
Training times are approximately 2–2.5 days per dataset.
```
📁 Data and Preparation
Follow the instructions below to download each dataset. Your folder structure should look like this:
```bash
text
.
├── data/
│   ├── indoor/
│   │   ├── test/
│   │   │   ├── 7-scenes-redkitchen/
│   │   │   │   ├── cloud_bin_0.info.txt
│   │   │   │   ├── cloud_bin_0.pth
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── train/
│   │   │   ├── 7-scenes-chess/
│   │   │   │   ├── cloud_bin_0.info.txt
│   │   │   │   ├── cloud_bin_0.pth
│   │   │   │   └── ...
│   │   ├── test_3DLoMatch_pairs-overlapmask.h5
│   │   ├── test_3DMatch_pairs-overlapmask.h5
│   │   ├── train_pairs-overlapmask.h5
│   │   └── val_pairs-overlapmask.h5
│   └── modelnet40_ply_hdf5_2048
│       ├── ply_data_test0.h5
│       ├── ply_data_test1.h5
│       └── ...
├── src/
└── README.md
```
3DMatch / 3DLoMatch
Download the processed 3DMatch dataset from the Predator project site (or use the preprocessed version linked in the original RegTR repository).

Place the contents into ../data/indoor/.

(Optional but recommended) Pre‑compute overlapping points for the overlap loss to speed up training:

'''
bash
cd src
python data_processing/compute_overlap_3dmatch.py
ModelNet
Download the PointNet‑processed ModelNet40 dataset from this official link and unzip into ../data/modelnet40_ply_hdf5_2048/.
'''

## 💻 Framework
<img width="1021" height="718" alt="image" src="https://github.com/user-attachments/assets/f21e399b-5c4d-4035-833a-2554f6602e14" />


## 🧠 Pretrained Models
We provide pretrained models for 3DMatch and ModelNet.
Download them from ModelNet and 3Dmatch (e.g., Google Drive, Zenodo) and unzip into the trained_models/ folder.

Expected structure after unzipping:
```
text
trained_models/
├── 3dmatch/
│   └── ckpt/
│       └── model-best.pth
└── modelnet/
    └── ckpt/
        └── model-best.pth
```
## 📊 Inference / Evaluation
Run the following commands from the src/ directory.

Note: Due to non‑determinism in GPU‑based KPConv neighborhood computation, results may vary slightly between runs.

3DMatch / 3DLoMatch

# 3DMatch benchmark (registration success <20cm)
python test.py --dev --resume ../trained_models/3dmatch/ckpt/model-best.pth --benchmark 3DMatch

# 3DLoMatch benchmark (more challenging)
python test.py --dev --resume ../trained_models/3dmatch/ckpt/model-best.pth --benchmark 3DLoMatch
ModelNet / ModelLoNet

# ModelNet
python test.py --dev --resume ../trained_models/modelnet/ckpt/model-best.pth --benchmark ModelNet
If you have defined a ModelLoNet split, use the same command.

## 🏋️ Training
To train the network from scratch, run the following commands from the src/ directory.

```
3DMatch
bash
python train.py --config conf/3dmatch.yaml
ModelNet
bash
python train.py --config conf/modelnet.yaml
```


## 🚀Citation
```bibtex
@ARTICLE{11395294,
  author={Zhou, Jingyu and Ma, Yunfeng and Jiang, Shuai and Wang, Yaonan and Liu, Min},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SCAP: Semantic Prototype Alignment for Robust Point Cloud Registration}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Point cloud compression;Semantics;Prototypes;Noise;Transformers;Robustness;Feature extraction;Accuracy;Robots;Estimation;Point cloud registration;rigid transformation estimation;semantic prototype},
  doi={10.1109/TCSVT.2026.3664224}}

```


## 🙏 Acknowledgements
We thank the authors of the following open‑source projects for making their code available:
RegTR, PTT, Predator, D3Feat, KPConv, DETR – their publicly released source code greatly facilitated this work.
