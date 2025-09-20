\# U-Net for Biological Image Segmentation



This project provides scripts to train U-Net models for two specific biological image segmentation tasks: \*\*Cytoskeleton Segmentation\*\* and \*\*Vesicle Segmentation\*\*. Each task uses a different data augmentation strategy tailored to its specific challenges.



---



\## Requirements



Before you begin, ensure you have the necessary Python libraries installed. It is recommended to use a virtual environment.



```bash

pip install torch torchvision torchaudio

pip install numpy opencv-python tqdm scipy

```

You can also create a `requirements.txt` file with the following content and install using `pip install -r requirements.txt`:

```

torch

torchvision

torchaudio

numpy

opencv-python

tqdm

scipy

```

A CUDA-enabled NVIDIA GPU is highly recommended for training.



---



\## Dataset Structure



Both training scripts expect the dataset to follow the same directory structure.



\- \*\*Image files\*\*: Can be any common image format (e.g., `.png`, `.jpg`, `.tif`).

\- \*\*Mask files\*\*: Must be in NumPy format (`.npy`), and their filenames must exactly match the corresponding image filenames (excluding the extension).



```

/path/to/your/dataset/

├── train/

│   ├── img/

│   │   ├── image\_001.png

│   │   ├── image\_002.png

│   │   └── ...

│   └── mask/

│       ├── image\_001.npy

│       ├── image\_002.npy

│       └── ...

└── val/

&nbsp;   ├── img/

&nbsp;   │   ├── image\_101.png

&nbsp;   │   ├── image\_102.png

&nbsp;   │   └── ...

&nbsp;   └── mask/

&nbsp;       ├── image\_101.npy

&nbsp;       ├── image\_102.npy

&nbsp;       └── ...

```



---



\## Training Models



Choose the script that best fits your segmentation target.



\### A) Cytoskeleton Segmentation



This task involves segmenting complex, fine-grained structures. The `train\_cytoskeleton\_seg.py` script is recommended as it uses a robust data augmentation pipeline to learn \*\*shape and texture features\*\* while being insensitive to brightness variations.



\*\*Key Features:\*\*

\- \*\*CLAHE\*\*: Enhances local contrast to make fine details clearer.

\- \*\*Geometric Augmentations\*\*: Includes flips and elastic deformations to learn morphological features.

\- \*\*Controlled Brightness/Contrast\*\*: Teaches the model brightness invariance in a stable manner.

\- \*\*Standardized Normalization\*\*: Ensures consistent input distribution for robust performance.



\*\*Training Command:\*\*

```bash

python train\_cytoskeleton\_seg.py \\

&nbsp;   --train-img-dir /path/to/your/dataset/train/img \\

&nbsp;   --train-mask-dir /path/to/your/dataset/train/mask \\

&nbsp;   --val-img-dir /path/to/your/dataset/val/img \\

&nbsp;   --val-mask-dir /path/to/your/dataset/val/mask \\

&nbsp;   --output-dir ./checkpoints\_cytoskeleton \\

&nbsp;   --epochs 100 \\

&nbsp;   --batch-size 4

```

The best model will be saved as `checkpoints\_cytoskeleton/best\_cytoskeleton\_model.pth`.



\### B) Vesicle Segmentation



This task involves segmenting blob-like vesicle structures. The `train\_vesicle\_seg.py` script uses a specific augmentation strategy involving \*\*wide-range intensity scaling\*\*. This approach aims to make the model invariant to a very large spectrum of brightness levels.



\*\*Key Features:\*\*

\- \*\*Extreme Intensity Scaling\*\*: Randomly multiplies image intensity by a very large factor.

\- \*\*Non-linear Transformation\*\*: Occasionally squares pixel values to create different intensity distributions.



\*\*Training Command:\*\*

```bash

python train\_vesicle\_seg.py \\

&nbsp;   --train-img-dir /path/to/your/dataset/train/img \\

&nbsp;   --train-mask-dir /path/to/your/dataset/train/mask \\

&nbsp;   --val-img-dir /path/to/your/dataset/val/img \\

&nbsp;   --val-mask-dir /path/to/your/dataset/val/mask \\

&nbsp;   --output-dir ./checkpoints\_vesicle \\

&nbsp;   --epochs 100 \\

&nbsp;   --batch-size 4

```

The best model will be saved as `checkpoints\_vesicle/best\_vesicle\_model.pth`.



---



\## Command-line Arguments



Both scripts share the same set of command-line arguments for configuration.



| Argument | Required/Optional | Default | Description |

| :--- | :---: | :---: | :--- |

| `--train-img-dir` | \*\*Required\*\* | - | Path to the training images directory. |

| `--train-mask-dir`| \*\*Required\*\* | - | Path to the training masks directory (`.npy` files). |

| `--val-img-dir` | \*\*Required\*\* | - | Path to the validation images directory. |

| `--val-mask-dir` | \*\*Required\*\* | - | Path to the validation masks directory (`.npy` files). |

| `--epochs` | Optional | `50` | Total number of training epochs. |

| `--batch-size` | Optional | `4` | Number of samples per batch. |

| `--lr` | Optional | `0.001` | Learning rate for the optimizer. |

| `--num-workers` | Optional | `4` | Number of worker processes for data loading. |

| `--device` | Optional | `cuda` | Device to use for training (e.g., 'cuda', 'cuda:0', 'cpu'). |

| `--output-dir` | Optional | `checkpoints`| Directory to save model checkpoints. |



