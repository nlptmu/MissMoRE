This repository provides the core PyTorch implementation for **MissMoRE** (*a multi-gate recursion network for automated dental charting from missing teeth classification on panoramic radiographs*).

## Repository Overview

This project includes modular implementations designed to be easily adapted for custom training pipelines, $K$-fold cross-validation, and fine-tuning:

1. **`MissMoRE.py`**: Complete PyTorch module implementation of the MissMoRE architecture, including multi-gate recursion mechanisms and classification heads.
2. **`Dataloaders.py`**: Data loading and preprocessing pipeline featuring custom ROI cropping logic, image transformations, and dataset augmentations.

---

## Getting Started & Usage

The code provided in this repository is modular and task-agnostic. You can easily integrate both the model architecture and dataloader into your own experimental workflows:

### 1. Model Adaptation
Import or load the `MissMoRE` module directly into your training script with "facebook/convnext-base-224-22k-1k" or any other models you wish to experiemnt with:
```python

model = MissMoRE(model="facebook/convnext-base-224-22k-1k")

```

### 2. DataModule Instantiation

The data loading pipeline is modularized into dedicated PyTorch DataModules for training and validation (`MissingTeethTrainDM` and `MissingTeethValDM`), supporting quadrant-based ROI selection, custom augmentations, and train/val splitting.

```python

import pandas as pd
from dataloaders import MissingTeethTrainDM, MissingTeethValDM,train_augmentations, validation augmentations

# 1. Load your dataset metadata
df = pd.read_csv("Dataset.csv")

# 2. Select Region of Interest (ROI) quadrant:
# 'q1': Upper-Right [17, 16, 15, 14, 13, 12, 11]
# 'q2': Upper-Left  [21, 22, 23, 24, 25, 26, 27]
# 'q3': Lower-Left  [31, 32, 33, 34, 35, 36, 37]
# 'q4': Lower-Right [41, 42, 43, 44, 45, 46, 47]
roi = 'q1'

# 3. Set train/validation split percentage (default used in paper: 0.6 / 60%)
split_pct = 0.6

# 4. Initialize DataModules
train_datamodule = MissingTeethTrainDM(
    df=df, 
    transform=train_augmentations, 
    roi=roi, 
    split_pct=split_pct
)

validation_datamodule = MissingTeethValDM(
    df=df, 
    transform=validation_augmentations, 
    roi=roi, 
    split_pct=split_pct
)

```

---

### 3. Dataset Structure

To run the implementation on your own dataset, prepare a CSV file (`Dataset.csv`) with the following columns:

* **`ImagePaths`**: Relative or absolute path to each panoramic radiograph image.
* **FDI Tooth Columns (28 columns total)**: Individual binary indicators (`1` = missing teeth, `0` = present teeth) for each permanent tooth:
  * **Upper-Right (Quadrant 1)**: `17`, `16`, `15`, `14`, `13`, `12`, `11`
  * **Upper-Left (Quadrant 2)**: `21`, `22`, `23`, `24`, `25`, `26`, `27`
  * **Lower-Left (Quadrant 3)**: `31`, `32`, `33`, `34`, `35`, `36`, `37`
  * **Lower-Right (Quadrant 4)**: `41`, `42`, `43`, `44`, `45`, `46`, `47`

*Note: Third molars (`18`, `28`, `38`, `48`) are excluded.*

#### Quadrant Label Arrays
When loaded via `Dataloaders.ipynb`, these 28 columns are automatically grouped into **7-dimensional binary arrays** per quadrant:

* **Upper-Right**: `[17, 16, 15, 14, 13, 12, 11]`
* **Upper-Left**: `[21, 22, 23, 24, 25, 26, 27]`
* **Lower-Left**: `[31, 32, 33, 34, 35, 36, 37]`
* **Lower-Right**: `[41, 42, 43, 44, 45, 46, 47]`

#### Example `Dataset.csv`
```csv
ImagePaths,17,16,15,14,13,12,11,21,22,23,24,25,26,27,31,32,33,34,35,36,37,41,42,43,44,45,46,47
images/image1.jpg,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
images/image2.jpg,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0
```

---

### Core Dependencies

* torch                         2.11.0+cu126
* albumentationsx               2.3.1
* opencv-python-headless        4.12.0.88
* transformers                  4.57.3

---

### Citation

If you use this code or build upon it, please cite our paper:
```
pending citation
```

---

### License
This project is licensed under the Apache-2.0 License - see the LICENSE file in the repository for details.
