# Diabetic Foot Screening System
veNTUre Project - supervised by Dr. Ang Yee Gary

### A Machine Learning Project for the National Healthcare Group

This repository details the development of a complete machine learning pipeline to classify diabetic foot complication risk using the **Plantar Thermogram Database**. The project successfully navigates complex image processing, iterative model development, and rigorous optimization, achieving a final **95% classification accuracy**.

---

## 1. Project Overview

The primary goal of this project is to develop a robust system for the automated screening of diabetic foot, a critical step in preventing severe complications. This project explores two stages of this system:

1.  **Image Segmentation:** Accurately isolating the foot region from raw thermal images, which often contain background noise, text overlays, and other artifacts.
2.  **Classification:** Using segmented foot data to train a high-performance classifier that can distinguish between healthy (Control) and diabetic (Diabetic) feet.

After extensive experimentation with both classical machine learning and deep learning approaches, the final, most successful model was an **optimized XGBoost classifier** built upon a set of 25+ engineered statistical features and a strategically balanced dataset.

---

## 2. Dataset

This project utilizes thermographic images collected by Dr. Ang Yee Gary (NHG) fo the image processing module.

Due to the small size of the collected data, the classification module utilizes the publicly available **Plantar Thermogram Database** from IEEE DataPort, which is designed for the study of diabetic foot complications.

-   **Permalink:** [https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications](https://ieee-dataport.org/open-access/plantar-thermogram-database-study-diabetic-foot-complications)
-   **DOI:** [https://dx.doi.org/10.21227/tm4t-9n15](https://dx.doi.org/10.21227/tm4t-9n15)

The dataset contains thermal images for 90 control subjects and 244 diabetic subjects, presenting a significant class imbalance challenge that was a key focus of this project.

---

## 3. Repository Structure & Notebooks

This repository contains three Jupyter notebooks that document the project's journey from preprocessing to the final model.

1.  **`Image_Segmentation.ipynb`**: Details the exploration of three different image segmentation techniques (Color-based, Otsu's Thresholding, and SAM). This notebook establishes the preprocessing pipeline, with the Segment Anything Model (SAM) being selected for its superior robustness.

2.  **`Classical_ML_Pipeline.ipynb` (ML2.ipynb - The Final Model)**: This is the core notebook detailing the successful end-to-end machine learning pipeline. It covers advanced feature engineering, strategic data balancing via under-sampling, hyperparameter optimization with Optuna, and the final model evaluation that achieved ~95% accuracy.

3.  **`CNN_Experimentation.ipynb` (DFU_CNN.ipynb)**: This notebook documents the investigation into using deep learning (CNNs and Transfer Learning) for this task. The experiments revealed that for this dataset's size and domain specificity, CNN models were prone to severe overfitting and could not learn generalizable patterns. This confirmed that the classical feature-based approach was superior for this problem.

---

## 4. Methodology & Key Findings

### 4.1. Image Segmentation (Preprocessing)

The first step was to isolate the foot from the background. Three methods were evaluated:
-   **Color-Based Segmentation in HSV:** Fast but required manual tuning and was not robust to background noise.
-   **Greyscale Thresholding with Otsu’s Method:** Fully automatic but struggled with images where the foot and background had similar thermal intensities.
-   **Segment Anything Model (SAM):** Proved to be the most robust and adaptable method, successfully handling complex backgrounds and varying image conditions without needing predefined thresholds. It was chosen for the final pipeline.

### 4.2. Feature Engineering

Instead of using raw pixel data, a comprehensive set of **25+ statistical and texture-based features** was engineered from the segmented thermogram data for each foot. This approach proved more effective than end-to-end deep learning by creating a rich, tabular dataset that classical models could interpret effectively.

Features included:
-   **Basic Statistics:** Mean, median, std dev, min/max temperature.
-   **Distribution Shape:** Skewness, kurtosis, and various temperature percentiles.
-   **Robust Statistics:** Interquartile Range (IQR) and Median Absolute Deviation (MAD).
-   **Texture & Zone Features:** Ratios of hot/cold pixels and simplified temperature gradients.

### 4.3. Model Development and Optimization

The core challenge was the significant class imbalance (90 Control vs. 244 Diabetic). The following steps were taken to build the final, high-performance model:

1.  **Data Balancing:** Initial models were heavily biased. The key breakthrough was implementing a **strategic under-sampling of the majority (Diabetic) class** to create a perfectly balanced 90:90 dataset. This forced the model to learn the features of both classes equally.
2.  **Model Selection:** An optimization "bake-off" was conducted between two powerful models: **XGBoost** and a Support Vector Machine (SVM).
3.  **Hyperparameter Tuning:** The **Optuna** framework was used to run over 100 trials for each model, automatically finding the optimal set of hyperparameters for the highest cross-validated accuracy.
4.  **Final Model:** The optimized **XGBoost classifier** was selected as the top-performing model, demonstrating superior performance on the balanced dataset.

---

## 5. Final Results

The final optimized XGBoost model, trained on the balanced dataset, achieved the following outstanding results on the unseen test set:

| Metric             | Control | Diabetic | **Accuracy (Overall)** |
| ------------------ | :-----: | :------: | :--------------------: |
| **Precision**      |  1.00   |   0.90   |                        |
| **Recall**         |  0.89   |   1.00   |         **94.4%**        |
| **F1-Score**       |  0.94   |   0.95   |                        |
| **Macro Avg**      |  0.95   |   0.94   |         **0.94**         |

*Note: The final reported accuracy of ~95% was achieved by applying an accuracy-optimized probability threshold to the model's output.*

---

## 6. How to Run

1.  Ensure you are in a GPU-enabled environment (e.g., Google Colab).
2.  Mount your Google Drive and ensure the dataset path in the notebooks is correct.
3.  Install the required dependencies by running the first cell in `Classical_ML_Pipeline.ipynb`:  
    `!pip install optuna imbalanced-learn xgboost`
4.  Run the notebooks. It is recommended to start with `Classical_ML_Pipeline.ipynb` to see the final, successful approach.

---

## 7. Skills & Tools Demonstrated

*   **Programming & Tools:** Python, Pandas, NumPy, OpenCV, Matplotlib, Seaborn, Jupyter
*   **Machine Learning:**
    *   **Classification:** XGBoost, SVM, Random Forest
    *   **Data Preprocessing:** Feature Engineering, StandardScaler, Data Balancing (Under-sampling)
    *   **Model Optimization:** Hyperparameter Tuning (Optuna), Cross-Validation, Threshold-Moving
    *   **Evaluation:** Classification Reports, Confusion Matrices, Accuracy, Precision/Recall/F1
*   **Deep Learning (Experimental):** TensorFlow, Keras, CNNs, Transfer Learning (EfficientNet), Image Augmentation
*   **Image Processing:** Segment Anything Model (SAM)
