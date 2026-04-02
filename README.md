# NESI: Shape Representation via Neural Explicit Surface Intersection

### [[Paper](https://dl.acm.org/doi/10.1145/3742893)] [[Project Page](https://www.cs.ubc.ca/labs/imager/tr/2025/nesi/)]

## Python dependencies
Create a Python 3.11 environment:
```
conda create -n nesi python=3.11
conda activate nesi
pip install --upgrade pip
pip install -r ./requirements.txt
```
Then install [Pytorch](https://pytorch.org/get-started/locally/) according to your CUDA version.

---
## Usage

Example for running ESI view selection:
```
python select_views.py -i <path_to_input_mesh> -o $<output_folder> --n-jobs <number_of_threads> [--fast]
```

Example for generating training data:
```
python sample_training_data.py -i <path_to_"view_directions.txt"> -o $<output_folder> --mask
```

Example for training NESI:
```
python train_nesi.py --dataset-path <path_to_"training_samples"> --mask --normal
```

Example for evaluating NESI:
```
python eval_nesi.py --model-path <path_to_results>
```

## Citation
If you find our method useful for your research, please cite our paper:

```
@article{zhang2025nesi,
author = {Zhang, Congyi and Yang, Jinfan and Hedlin, Eric and Takikawa, Suzuran and Vining, Nicholas and Yi, Kwang Moo and Wang, Wenping and Sheffer, Alla},
title = {NESI: Neural Explicit-Shape-Intersection-Based Geometry Representation},
year = {2025},
issue_date = {October 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {5},
issn = {0730-0301},
url = {https://doi.org/10.1145/3742893},
doi = {10.1145/3742893},
journal = {ACM Trans. Graph.},
month = jul,
articleno = {166},
numpages = {23},
keywords = {Implicit Surfaces, Explicit Surfaces, Solid Modeling, CSG, Parametric Curves \& Surfaces}
}
```

## Acknowledgement
This is partially built upon the [implementation](https://github.com/nv-tlabs/nglod) of [Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces](https://research.nvidia.com/labs/toronto-ai/nglod/).