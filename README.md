# Intel Image Classification

## Project Overview
This project classifies natural scene images into six Intel image classes:

- buildings
- forest
- glacier
- mountain
- sea
- street

The repository uses a **feature-engineering + classical ML** pipeline:

1. Extract deep image embeddings from a pretrained **DenseNet161** model.
2. Extract handcrafted **HOG (Histogram of Oriented Gradients)** features.
3. Concatenate both feature sets into one feature vector per image.
4. Normalize features and reduce dimensionality with **PCA (95% explained variance)**.
5. Train a **linear SVM** classifier and evaluate with accuracy + confusion matrix.

## Repository Structure
- `dense.ipynb`: Feature extraction workflow (DenseNet161 and HOG) and pickle creation.
- `Training.ipynb`: Training/evaluation workflow using saved pickle features.
- `Untitled.ipynb`: Early/partial duplicate of the training notebook.

## End-to-End Steps Used in the Project

### 1) Prepare class labels and paths
The notebooks define the six categories and set train/test folder paths pointing to the Intel scene dataset layout (`seg_train/seg_train` and `seg_test/seg_test`).

### 2) Extract DenseNet161 embeddings (`dense.ipynb`)
- Load pretrained `torchvision.models.densenet161`.
- Remove classifier head with `model.classifier = nn.Identity()` to use penultimate embeddings.
- Preprocess each image (RGB conversion, resize to 224×224, ImageNet normalization).
- Run a forward pass and flatten each output embedding.
- Save `[feature, label]` pairs to pickle files.

### 3) Extract HOG descriptors (`dense.ipynb`)
- Resize images to 128×64.
- Compute HOG descriptors (`orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`).
- Save HOG `[feature, label]` pairs for both train and test splits.

### 4) Load and combine features (`Training.ipynb`)
- Load DenseNet and HOG pickle files.
- Split into train/test feature matrices and label vectors.
- Concatenate HOG + DenseNet features for each sample.
- Normalize feature matrices before modeling.

### 5) Dimensionality reduction and model training (`Training.ipynb`)
- Fit PCA with `n_components=0.95` on training features.
- Transform train and test features with the fitted PCA model.
- Train `sklearn.svm.SVC` with a linear kernel.

### 6) Evaluate model (`Training.ipynb`)
- Report train/test accuracy.
- Generate predictions on the test set.
- Compute confusion matrix and visualize it with seaborn heatmap.

## Notes
- The notebook paths are currently hardcoded to local Windows/Linux directories and may need updates for other environments.
- `Untitled.ipynb` appears to be an unfinished exploratory copy and is not required for the main pipeline.
