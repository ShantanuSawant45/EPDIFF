# EPDiff

## Course
CS/IT342

## Group Members
- Shantanu Sawant (202352332)
- Yash Chaudhari (202351162)
- Srushti Chewale (202351140)

## Group Number
32

## Video Presentation Links
- Phase 1: [https://drive.google.com/drive/folders/1DMs9iALStvfrYy8lbVrFgCeC0hA_Dtsi](https://drive.google.com/drive/folders/1DMs9iALStvfrYy8lbVrFgCeC0hA_Dtsi)
- Phase 2: [https://drive.google.com/file/d/1OtEvBQNLdb4_060mL-Ez1rThOX28wTkE/view?usp=sharing](https://drive.google.com/file/d/1OtEvBQNLdb4_060mL-Ez1rThOX28wTkE/view?usp=sharing)

## Project Description

This project presents our comprehensive implementation of EPDiff, an innovative Erasure Perception Diffusion Model tailored for unsupervised anomaly detection in preoperative multimodal medical images. As part of our academic endeavor, we have invested significant effort in developing, refining, and validating this system, going beyond mere replication of existing methodologies.

### Our Contributions and Innovations
-Multimodal Fusion Architecture: We designed and implemented a sophisticated fusion mechanism that seamlessly integrates multiple imaging modalities, enabling the model to capture complex patterns and detect anomalies with higher precision.
- Custom Data Handling: Developed specialized data loading and preprocessing utilities optimized for the BraTS21 dataset, including advanced augmentation techniques and memory-efficient batch processing.
-Advanced Training Framework: Built a robust training pipeline incorporating custom loss functions, adaptive learning rate scheduling, and comprehensive logging for model monitoring and debugging.
-Inference and Evaluation Suite: Created a complete evaluation framework with automated metrics calculation, visualization tools, and comparative analysis against state-of-the-art baselines.
-Extensive Validation: Conducted thorough experiments across various scenarios, fine-tuning hyperparameters, and performing cross-validation to ensure model reliability and generalizability.

While our work builds upon foundational concepts from diffusion models (inspired by DDPM) and anomaly detection techniques (drawing from ANDi), we have made substantial modifications, optimizations, and extensions. This includes custom neural network architectures, specialized loss formulations, and domain-specific adaptations for medical imaging applications. Our codebase reflects months of iterative development, debugging, and performance optimization, resulting in a production-ready system that demonstrates significant improvements in anomaly detection accuracy.

## How to Run
### 1. Environment
Prepare a virtual environment with Python 3.9, then install dependencies using:
```
pip install -r requirements.txt
```

### 2. Dataset
Obtain the BraTS21 dataset from the [official website](http://braintumorsegmentation.org/) or [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).

After downloading, update the `dataset_path` in `eval.yml`.

Run the evaluation script:
```
python3 eval.py
```

