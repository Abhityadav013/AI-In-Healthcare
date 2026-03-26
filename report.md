# Alzheimer’s Disease Severity Classification from MRI Images Using EfficientNetB0 with Patient-Aware Data Splitting

## 1. Introduction

Alzheimer’s disease is a progressive neurological disorder that affects memory, thinking, and behavior. Early and accurate detection is important for clinical support and disease management. Medical imaging, especially MRI, is widely used in the analysis of structural brain changes associated with dementia.

The aim of this project was to build a deep learning model that can classify MRI images into four categories of dementia severity:

- Non Demented
- Very mild Dementia
- Mild Dementia
- Moderate Dementia

The project was initially based on a Kaggle notebook and later converted into a clean local PyTorch project that could run efficiently on macOS using Apple Silicon GPU acceleration through PyTorch MPS.

---

## 2. Objective

The main objectives of this project were:

- To preprocess and organize the MRI image dataset for local training
- To build an image classification model using transfer learning
- To ensure correct data splitting without leakage between train, validation, and test sets
- To evaluate the model using reliable classification metrics
- To analyze the limitations caused by class imbalance and limited patient diversity

---

## 3. Dataset

This project uses the **OASIS Alzheimer’s MRI image dataset** obtained from Kaggle.

The dataset contains four class folders:

- Non Demented
- Very mild Dementia
- Mild Dementia
- Moderate Dementia

### 3.1 Raw Class Distribution

| Class | Number of Images |
|---|---:|
| Non Demented | 67,222 |
| Very mild Dementia | 13,725 |
| Mild Dementia | 5,002 |
| Moderate Dementia | 488 |

The dataset is highly imbalanced, with the majority of images belonging to the Non Demented class and very few images in the Moderate Dementia class.

---

## 4. Initial Project State

The project was initially adapted from a Kaggle notebook into a modular local PyTorch codebase. In the early stages, the system produced extremely high validation accuracy, close to **99.8%**, which appeared suspicious for a medical imaging classification task with four classes and severe imbalance.

After analysis, it was found that the earlier approach had a serious risk of **data leakage**. Since multiple MRI slices belong to the same patient, splitting images without strict patient-level separation can place slices from the same patient into both training and evaluation sets. This makes the model appear much better than it actually is.

Therefore, the initial near-perfect accuracy was not considered reliable.

---

## 5. Problems Found in the Initial Version

The following issues were identified:

### 5.1 Data Leakage Risk
The original split logic did not guarantee full patient-level separation across training, validation, and test sets.

### 5.2 Unrealistic Performance
Validation performance near 99.8% strongly suggested leakage or an incorrect split design.

### 5.3 Aggressive Balancing
The earlier training logic used a much larger balancing target, which could increase memorization instead of generalization.

### 5.4 Notebook-Oriented Structure
Some parts of the earlier implementation still reflected notebook-style experimentation and lacked stronger diagnostics and modular checks.

---

## 6. Changes Made to the Project

To make the system more correct, reliable, and academically valid, several changes were introduced.

### 6.1 Patient-Aware Train / Validation / Test Split
The `data_loader.py` pipeline was redesigned so that:

- Each patient belongs to only one split
- Overlap checks are performed between train, validation, and test
- Diagnostics are printed for:
  - Unique patient counts
  - Patient overlap
  - Scan overlap
  - File path overlap
  - Per-class distribution in each split

This was the most important correction in the project.

### 6.2 Training-Only Balancing
Class balancing is now applied **only to the training set**.  
Validation and test sets remain untouched so they better reflect real-world class imbalance.

### 6.3 Hyperparameter Tuning
The configuration was updated to reduce overfitting:

| Parameter | Old Value | New Value |
|---|---:|---:|
| Train samples per class | 7000 | 2000 |
| Learning rate | 1e-4 | 3e-5 |
| Dense units | 512 | 256 |
| Weight decay (`L2_REG`) | 0.001 | 1e-4 |
| Early stopping patience | 5 | 8 |
| Epochs | 50 | 30 |

### 6.4 MRI-Friendly Transforms
Color jitter was removed because it is less appropriate for medical MRI images.  
The final training transforms were limited to:
- Resize
- Horizontal flip
- Small rotation
- Normalization

### 6.5 Improved Evaluation
The evaluation system was expanded to include:
- Accuracy
- Balanced accuracy
- Macro F1
- Macro precision
- Macro recall
- Cohen’s Kappa
- Matthews Correlation Coefficient
- Log loss
- ROC AUC
- Top-2 accuracy
- Confusion matrix
- Classification report

---

## 7. Final Methodology

### 7.1 Model
The final model used was **EfficientNetB0** with transfer learning in PyTorch.

The architecture consists of:
- Pretrained EfficientNetB0 backbone
- Batch normalization
- Dropout
- Fully connected hidden layer
- Final classification layer with 4 outputs

### 7.2 Input Settings
- Image size: **128 × 128**
- Batch size: **32**
- Optimizer: **AdamW**
- Loss function: **CrossEntropyLoss**
- Device: **MPS (Apple Silicon GPU)**

### 7.3 Final Splitting Behavior
The final data split was patient-disjoint:

| Split | Unique Patients |
|---|---:|
| Train | 222 |
| Validation | 55 |
| Test | 70 |

Diagnostics confirmed:

- Patient overlap = 0
- Scan overlap = 0
- File path overlap = 0

This means the final evaluation is leakage-free.

---

## 8. Training Behavior

The final training process showed the following pattern:

- Training accuracy kept improving
- Validation accuracy reached a moderate range
- Validation loss stopped improving after a few epochs
- Early stopping selected the best checkpoint automatically

This indicates that the model began to overfit after the first few epochs, but the saved checkpoint still represents the best valid model found during training.

---

## 9. Final Results

The final evaluation on the test set produced the following metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.7113 |
| Balanced Accuracy | 0.4013 |
| Macro F1 | 0.3804 |
| Macro Precision | 0.3694 |
| Macro Recall | 0.4013 |
| Cohen’s Kappa | 0.3211 |
| MCC | 0.3269 |
| Log Loss | 0.6777 |
| ROC AUC Macro | 0.7907 |
| Top-2 Accuracy | 0.9144 |

### 9.1 Per-Class Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non Demented | 0.8856 | 0.8003 | 0.8408 | 13,298 |
| Very mild Dementia | 0.3337 | 0.4893 | 0.3968 | 2,806 |
| Mild Dementia | 0.2584 | 0.3156 | 0.2841 | 976 |
| Moderate Dementia | 0.0000 | 0.0000 | 0.0000 | 244 |

---

## 10. Result Analysis

The final results are much more realistic than the initial near-perfect results and are therefore more trustworthy.

### 10.1 Strong Performance on Majority Class
The model performs well on the **Non Demented** class, with high precision and recall.

### 10.2 Weak Minority-Class Performance
Performance is much weaker on:
- Very mild Dementia
- Mild Dementia
- Moderate Dementia

### 10.3 Failure on Moderate Dementia
The model failed to correctly classify the Moderate Dementia class.

This is mainly due to:
- Extremely small number of images
- Very small number of unique patients
- Inability to form a strong validation split for this class in a leak-free 3-way patient split

### 10.4 Importance of Macro Metrics
Although the overall accuracy is **71.13%**, the macro F1 score is only **0.3804** and balanced accuracy is **0.4013**.

This shows that overall accuracy alone can be misleading in imbalanced medical datasets.  
Macro metrics provide a more honest view of the model’s behavior across all classes.

---

## 11. Key Findings

The most important findings of this project are:

1. **Patient-level splitting is critical** in MRI-based classification tasks.
2. Extremely high validation accuracy can be misleading if leakage exists.
3. After leakage was removed, performance became lower but more realistic.
4. The model can classify the majority class reasonably well.
5. Severe class imbalance significantly reduces performance on minority classes.
6. Moderate Dementia remains especially difficult because of insufficient data diversity.

---

## 12. Limitations

This project has several important limitations.

### 12.1 Severe Class Imbalance
The dataset is dominated by the Non Demented class.

### 12.2 Very Small Moderate Dementia Group
Moderate Dementia has too few unique patients for robust generalization.

### 12.3 Overfitting Tendency
Even after corrections, the model begins to overfit after a few epochs.

### 12.4 Limited MRI Representation
The project uses 2D image slices rather than full 3D patient-level volumetric analysis.

---

## 13. Future Work

Several improvements can be explored in future work:

- Weighted loss or weighted sampling
- Stronger class imbalance handling
- Trying alternative pretrained backbones
- Better hyperparameter tuning
- Patient-level aggregation across slices
- Collection of more Moderate Dementia data
- Use of additional metadata or multimodal inputs

---

## 14. Conclusion

This project developed a local PyTorch-based MRI classification system for Alzheimer’s disease severity using EfficientNetB0 and Apple Silicon GPU acceleration.

The most important outcome of the project was not only the final accuracy, but the correction of the evaluation pipeline to ensure a **leak-free patient-aware split**. After fixing the data leakage issue, the model achieved **71.13% test accuracy**, with strong performance on the Non Demented class but weaker results on minority classes.

These results show that the model has some predictive ability, but also demonstrate the challenges of imbalanced medical imaging datasets. The final system is therefore more scientifically reliable than the initial version and provides a realistic baseline for further work.

---

## 15. Tools and Technologies Used

- Python
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn
- Pillow
- macOS Apple Silicon (MPS)

---

## 16. Repository / Implementation Notes

The project was converted from a Kaggle notebook into a modular local codebase with the following main files:

- `config.py`
- `src/data_loader.py`
- `src/model.py`
- `src/train.py`
- `src/evaluate.py`
- `utils/device.py`
- `utils/metrics.py`
- `utils/plots.py`
- `utils/seed.py`

This structure improves readability, reproducibility, and maintainability compared to the original notebook-based implementation.