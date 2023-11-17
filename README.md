# CapsGI-pytorch
We run on a DELL server with 2 * Intel(R) Xeon(R) Silver 4210, 4 * NVIDIA TITAN V (12G), 10 * 32GB DDR4 RAM and 1 * 8TB hard disk.

Requirements
====
You can create a virtual environment first via:
```
conda create -n your_env_name python=3.8.5
```

You can install all the required tools using the following command:
```
# CUDA 10.2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```

Overview
====

to generate the initial node features. 
Specially, we perform eigen-decomposition on the normalized adjacency matrix for the feature initialization.

* `models/` contains the implementation of the CapsGI loss (`CapsGI.py`) and the binary classifier (`clf_model.py`).

* `layers/` contains the implementation of the GIN layer (`graphcnn.py`), the MLP layer (`mlp.py`), the averaging readout (`readout.py`), and the bi-linear discriminator (`discriminator.py`). `readout.py` and `discriminator.py` are copied from the source code of [Deep Graph Infomax](https://github.com/PetarV-/DGI)[2]. `mlp.py` is copied from the source code of [GIN](https://github.com/weihua916/powerful-gnns)[1]. `graphcnn.py` is revised based on the corresponding implemention in [GIN](https://github.com/weihua916/powerful-gnns).

* `util.py` is used for loading and pre-processing the dataset.

Running the code
====
To run the scheme, execute:
```
$ python main_CapsGI.py --dataset amazon
```

Reference
====
[1] Petar Velickovic, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R. Devon Hjelm. 2019. Deep Graph Infomax. In ICLR.

[2] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019. How Powerful are Graph Neural Networks?. In ICLR.
