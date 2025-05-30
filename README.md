# 🧠 Alzheimer's MRI Classification with Augmented Dataset

This project leverages **Convolutional Neural Networks (CNNs)** to classify MRI brain scans into four stages of Alzheimer's Disease using the **Augmented Alzheimer's MRI Dataset.** The pipeline follows a structured deep learning workflow including EDA, preprocessing, augmentation, CNN modeling, evaluation, and visualization.

---

## 📂 Dataset
- **Source:** [Augmented Alzheimer MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
- **Classes:**
  - **Mild Demented**
  - **Moderate Demented**
  - **Non Demented**
  - **Very Mild Demented**
- **Structure:**
  - Combined both OriginalDataset and AugmentedAlzheimerDataset
  - Dynamic splits using train_test_split with stratification:
    - **56%** Training.
    - **14%** Validation
    - **30%** Testing
      
---

## 🛠 Tools and Frameworks
- **Programming Language:** Python
- **Libraries:**
  - TensorFlow & Keras
  - NumPy & Pandas
  - Matplotlib & Seaborn
  - Scikit-learn
  - PIL (for EDA image analysis)
- **Kaggle:** Kaggle + Local Notebook

---

## 🔍 Project Workflow

### 1. Dataset Loading
- Used kagglehub API to download the full dataset.
- Constructed a custom DataFrame with image paths and labels from both OriginalDataset and AugmentedAlzheimerDataset.
- Verified class distributions and image consistency.
  
### 2. Exploratory Data Analysis (EDA)
- Visualized sample images per class.
- Analyzed class imbalance, image size variability, and color modes.
- Explored brightness distributions and source differences.

### 3. Data Preprocessing
- Resized all images to **224 x 224**.
- Applied **MobileNetV2 preprocessing** function for scaling pixel values.
- Split dataset using stratified train_test_split.

### 4. Data Augmentation
- Implemented augmentations through ImageDataGenerator:
  - Random rotations
  - Zooming
  - Contrast shifts
  - Horizontal flips
    
### 5. Model Architecture
- Built a deep CNN using tf.keras.Sequential:
  - Multiple Conv2D + BatchNormalization blocks
  - MaxPooling2D for spatial downsampling
  - Dropout for regularization
  - Dense layers with ReLU for classification
  - Final softmax layer for multi-class output
    
### 6. Optimizer: SGD (Stochastic Gradient Descent)
- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Regularization:** Included EarlyStopping with patience=3 and best weight restoration
- **Epochs:** 30 (with early stopping)

### 7. Evaluation
- Evaluated model on the test set using:
  - Final test accuracy and loss
  - **Classification report** (precision, recall, F1-score)
  - **Confusion matrix** heatmap
  
- Visualized:
  - Accuracy & loss curves
  - Misclassifications via confusion matrix

---

## 📊 Results
- **Test Accuracy:** Achieved ~93% accuracy on the held-out test set.
- **Classification Report:** Strong precision and recall across all 4 classes.
- **Confusion Matrix:** Showed robust separation between classes, especially Non-Demented and Moderate Demented.

---

## 🖼 Visualizations
1. **EDA Insights:**
  - Class distribution bar charts
  - Image resolution histograms
  - Brightness & color mode distributions
2. **Model Training:**
    - Accuracy and loss vs. epoch plots
    - EarlyStopping detection
3. **Performance Analysis:**
  - Confusion matrix heatmap
  - Classification report with F1-scores

 
