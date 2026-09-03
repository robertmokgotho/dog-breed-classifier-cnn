# dog-breed-classifier-cnn

A Python command-line application that uses pre-trained Deep Learning models (Convolutional Neural Networks) to identify dog breeds from images and compare model performance, accuracy, and runtime execution.

Developed as part of the **AWS AI & ML Scholarship Program** / **Udacity AI Programming with Python Nanodegree**.

---

## 📌 Project Overview

This project evaluates three pre-trained Convolutional Neural Network (CNN) architectures—**VGG**, **ResNet**, and **AlexNet**—on an image classification task with two primary objectives:

1. **Dog vs. Not-Dog Classification:** Distinguish whether an image depicts a dog or a non-dog entity.
2. **Breed Classification:** Correctly identify the specific breed for images classified as dogs.

---

## 📊 Key Results & Summary Comparison

Based on execution across the benchmark dataset (`workspace/pet_images/`):

| Model Architecture | Total Images | Dog Classification Accuracy | Breed Classification Accuracy | Not-a-Dog Accuracy | Total Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VGG** | 40 | **100.0%** | **93.3%** | **100.0%** | ~3.5s |
| **ResNet** | 40 | 90.0% | 90.0% | 100.0% | ~1.2s |
| **AlexNet** | 40 | 100.0% | 80.0% | 100.0% | **~0.5s** |

### **Best Model Selection:**
* **Overall Winner:** **VGG** achieved the highest overall accuracy, correctly identifying 100% of dog/not-dog labels and obtaining the highest breed identification accuracy (93.3%).
* **Efficiency Trade-off:** While VGG provides superior accuracy, **AlexNet** or **ResNet** offer faster inference times when computational efficiency is prioritized over maximum precision.

---

## 📁 Repository Structure

```text
.
├── requirements.txt                  # Frozen Python environment dependencies
├── .gitignore                        # Prevents venv and cache files from uploading
├── README.md                         # Project documentation
└── workspace/                        # Main project working directory
    ├── check_images.py               # Main program entry point
    ├── get_input_args.py             # Handles command-line arguments parsing
    ├── get_pet_labels.py             # Extracts ground truth labels from image filenames
    ├── classify_images.py            # Performs image classification using pre-trained CNNs
    ├── calculates_results_stats.py   # Computes accuracy percentages and count statistics
    ├── print_results.py              # Formats and displays terminal execution summaries
    ├── classifier.py                 # Wrapper function for PyTorch pre-trained models
    ├── dognames.txt                  # Reference list of valid dog breed names
    ├── pet_images/                   # Benchmark dataset images
    └── run_models_batch.sh           # Shell script for batch execution across architectures
