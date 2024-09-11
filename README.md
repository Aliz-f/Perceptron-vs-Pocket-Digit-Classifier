# Handwritten Digit Recognition Using Perceptron and Pocket Algorithms

This repository contains implementations of the **Perceptron Learning Algorithm (PLA)** and **Pocket Algorithm** for handwritten digit recognition. The goal is to classify digits using two features extracted from the images—**average intensity** and **symmetry**—and improve the model's performance by adding a third custom feature.

## Overview

The handwritten digit dataset used in this project is from the **US Postal Service Zip Code** dataset, where each image is represented as a 16x16 pixel grayscale image. The first task is to implement binary classification between two digits. The final number of your UD ID is one of the digits, and the other is chosen conveniently for replicating the results in the slides.

### Main Tasks:
1. **Feature Extraction**: 
   - Extract two features—**average intensity** and **symmetry**—from the images for initial binary classification.
   - Add a third custom feature in later stages to improve performance.
   
2. **Perceptron Learning Algorithm (PLA)**:
   - Implement the PLA for binary classification using the extracted features.
   - Train the model and compute the error on both training and test sets.

3. **Pocket Algorithm**:
   - Extend the PLA by implementing the Pocket Algorithm to handle non-linearly separable data.
   - Compare performance with the standard PLA.

4. **Visualization**:
   - Visualize the decision boundary and error metrics \(E_{in}\) and \(E_{out}\) for both algorithms.

## Data Description

The dataset contains images of digits, where:
- **DigitsTraining**: Contains 7291 training examples.
- **DigitsTesting**: Contains 2007 test examples.
- Each row in the dataset consists of 256 pixels representing a 16x16 image and the label (digit number) in the first column.

## Feature Extraction

### 1. **Average Intensity**:
   - The mean pixel value of the image, capturing the brightness of the digit.

### 2. **Symmetry**:
   - Calculated by comparing the original image with its vertically and horizontally flipped versions. Measures how symmetric the digit is.

### 3. **Custom Feature**:
   - You will add a third feature (e.g., **edge density**, **aspect ratio**, or another image characteristic) to improve classification performance.

## Algorithms

### Perceptron Learning Algorithm (PLA)
The PLA is a linear classifier used to separate two classes of digits using the extracted features. It updates weights iteratively to classify the points correctly.

### Pocket Algorithm
The Pocket Algorithm extends PLA by handling non-linearly separable data. It "pockets" the best-performing weights encountered during training to improve generalization on unseen data.

## Implementation

### Notebooks

1. **`classifier(PLA).ipynb`**:
   - Implements the Perceptron Learning Algorithm with 2D feature space.
   
2. **`classifier(Pocket).ipynb`**:
   - Implements the Pocket Algorithm for 2D feature space and compares it with PLA.

3. **`classifier(PLA)-3D.ipynb`**:
   - Implements the Perceptron Learning Algorithm in a 3D feature space (with a third feature).
   
4. **`classifier(Pocket)-3D.ipynb`**:
   - Implements the Pocket Algorithm in a 3D feature space and compares it with PLA.

### Results
- **PLA**: Shows the decision boundary and errors for 200 iterations using two features (average intensity and symmetry).
- **Pocket Algorithm**: Visualizes the decision boundary and errors using the best-performing weights found during training.
- **3D Feature Space**: Adding a third feature improves the decision boundary and reduces the error.

## How to Run the Code

1. Clone the repository:
    ```bash
    git clone https://github.com/Aliz-f/Perceptron-vs-Pocket-Digit-Classifier.git
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the provided Jupyter notebooks to view the implementation and run the code.

    - `classifier(PLA).ipynb` for 2D feature space using PLA.
    - `classifier(Pocket).ipynb` for 2D feature space using Pocket Algorithm.
    - `classifier(PLA)-3D.ipynb` for 3D feature space using PLA.
    - `classifier(Pocket)-3D.ipynb` for 3D feature space using Pocket Algorithm.

4. Run each notebook to train the models and visualize the results.
ision_boundary(X_train, y_train, weights, bias, iteration=200)


## Results and Conclusion

- **2D Classification**: Using two features (average intensity and symmetry), the PLA and Pocket Algorithm were implemented for binary classification. Pocket Algorithm shows better generalization when the data is not linearly separable.
  
- **3D Classification**: Adding a third feature improves classification performance, resulting in better separation of the classes and a lower error rate for both algorithms.
