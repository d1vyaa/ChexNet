#  CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning

---

##  Overview

CheXNet is a deep learning algorithm that detects **pneumonia** from chest X-ray images at a level exceeding practicing radiologists. This implementation is based on the original Stanford paper by Rajpurkar et al. (2017), using a **121-layer DenseNet** trained on the NIH **ChestX-ray14** dataset to classify 14 thoracic diseases simultaneously.

> **Key Result:** CheXNet achieves an F1 Score of **0.435** vs. the radiologist average of **0.387**, outperforming all four radiologists tested.

---

##  Architecture

```
Input: Chest X-Ray Image (224×224×3)
         ↓
DenseNet-121 (ImageNet pretrained)
  ├── Dense Block 1  (6 layers,  64 → 256 feature maps)
  ├── Transition Layer 1
  ├── Dense Block 2  (12 layers, 128 → 512 feature maps)
  ├── Transition Layer 2
  ├── Dense Block 3  (24 layers, 256 → 1024 feature maps)
  ├── Transition Layer 3
  └── Dense Block 4  (16 layers, 512 → 1024 feature maps)
         ↓
Global Average Pooling
         ↓
Dense Layer (14 outputs, Sigmoid activation)
         ↓
Output: Probability vector for 14 pathologies
```

### Why DenseNet?
In very deep networks, gradients can vanish before reaching earlier layers. DenseNet solves this by **connecting each layer to every other layer** within a block — instead of summing feature maps (like ResNet), it **concatenates** them, preserving and reusing all learned features.

---

##  Dataset: ChestX-ray14

| Split      | Patients | Images  |
|------------|----------|---------|
| Training   | 28,744   | 98,637  |
| Validation | 1,672    | 6,351   |
| Test       | 389      | 420     |
| **Total**  | **30,805** | **112,120** |

- Source: Wang et al. (2017), NIH Clinical Center
- Each image is labeled with up to **14 thoracic pathology labels** extracted automatically from radiology reports
- Images downscaled to **224×224**, normalized by mean and standard deviation
- Training data augmented with **random horizontal flipping**
- No patient overlap between splits
- Dataset Link:  
https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## 🧮 Loss Function

The model optimizes a **weighted binary cross-entropy loss** across all 14 classes:

$$L(X, y) = \sum_{c=1}^{14} \left[ -y_c \log p(Y_c = 1 \mid X) - (1 - y_c) \log p(Y_c = 0 \mid X) \right]$$

Where the class weights handle label imbalance:
- `w⁺ = |N| / (|P| + |N|)` — weight for positive examples
- `w⁻ = |P| / (|P| + |N|)` — weight for negative examples

---


##  Training Configuration

| Hyperparameter       | Value                            |
|----------------------|----------------------------------|
| Base Model           | DenseNet-121 (ImageNet weights)  |
| Image Size           | 224 × 224                        |
| Batch Size (per GPU) | 64                               |
| Initial LR           | 0.001                            |
| LR Decay             | ×0.1 when val_loss plateaus      |
| Epochs               | 10                               |
| Optimizer            | Adam                             |
| Precision            | Mixed (float16 + float32)        |
| Multi-GPU Strategy   | `tf.distribute.MirroredStrategy` |

---

##  Results

### Comparison with Radiologists (Pneumonia F1 Score)

| Model / Reader    | F1 Score (95% CI)     |
|-------------------|-----------------------|
| Radiologist 1     | 0.383 (0.309 – 0.453) |
| Radiologist 2     | 0.356 (0.282 – 0.428) |
| Radiologist 3     | 0.365 (0.291 – 0.435) |
| Radiologist 4     | 0.442 (0.390 – 0.492) |
| **Radiologist Avg.** | **0.387 (0.330 – 0.442)** |
| **CheXNet**       | **0.435 (0.387 – 0.481)** ✅ |

### AUROC vs. Previous CNNs (ChestX-ray14)

| Pathology         | Wang et al. (2017) | Yao et al. (2017) | CheXNet     |
|-------------------|--------------------|-------------------|-------------|
| Atelectasis       | 0.716              | 0.772             | **0.8094**  |
| Cardiomegaly      | 0.807              | 0.904             | **0.9248**  |
| Effusion          | 0.784              | 0.859             | **0.8638**  |
| **Pneumonia**     | 0.633              | 0.713             | **0.7680**  |
| Emphysema         | 0.815              | 0.829             | **0.9371**  |
| Hernia            | 0.767              | 0.914             | **0.9164**  |
| *(all 14 classes)* | —                 | —                 | **Best on all 14** ✅ |

---

##  Project Structure

```
ChexNet/
│
├── ChexNet_Longer_Training.ipynb   # Main training notebook
├── best_model.weights.h5           # Saved best model weights (generated after training)
├── trained_net.png                 # ROC curve output (generated after evaluation)
└── README.md
```

---

##  Key Implementation Details

- **Data Pipeline**: Uses `tf.data` with parallel image loading, augmentation, and prefetching for high-throughput GPU training
- **Class Imbalance**: Handled via per-batch dynamic positive/negative weight computation in the custom loss function
- **Sampling Strategy**: Weighted sampling (40,000 images) biases toward multi-label cases for better disease coverage
- **Checkpointing**: Saves best model weights based on validation loss using `ModelCheckpoint`
- **Evaluation**: Per-class ROC curves + overall AUROC score on held-out test set

---

##  Limitations

- Only **frontal-view** chest X-rays used — lateral views excluded
- Only **4 radiologists** used for human performance benchmark
- Neither the model nor radiologists had access to **patient history**
- Original images (1024×1024) are downsampled to 224×224 — roughly a **20–50× reduction** in image information vs. clinical-quality images
- CXR14 labels are auto-extracted from reports and may not perfectly match images

---

##  References

- Rajpurkar, P., Irvin, J., Zhu, K., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2016). *Densely Connected Convolutional Networks.* arXiv:1608.06993
- Wang, X., Peng, Y., Lu, L., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.*
- Stanford ML Group: https://stanfordmlgroup.github.io/projects/chexnet/


B.Tech Computer Science, Delhi Technological University
GitHub: [@d1vyaa](https://github.com/d1vyaa)
