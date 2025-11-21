# Smarter Maintenance Machine Learning for Reliable Air Compressor Health Monitoring

## Project Metadata
### Authors
- **Team:** Khalid Alyahya, Abdullah Alsagoor
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Reciprocating compressors are important assets in process industries such as petrochemical and energy industries where it is important to have continuity in its operations. Malfunctions of such machines may cause unpredicted downtime, expensive maintenance procedures, and accidents. Consequently, coming up with effective and sound fault diagnosis (FD) systems has emerged as a research focus. Vibration analysis is the most informative method of data-driven techniques because it is sensitive to mechanical degradation.

Previous researchers on an air-compressor dataset publicly available have found high, and in some cases perfect classification performance. These experiments however rely on large amounts of manual feature engineering and use simple random splits which can blur the generalization capacity of their model. As an example, Nambiar et al. (2024) obtained 100 percent accuracy through feature fusion and kNN, and Bhattacharyya et al. (2025) obtained 98.66 percent through a Weightless Neural Network. The reproducibility and scalability of such methods is limited by the fact that they are based on handcrafted features.

In this case, the study employs an end-to-end deep learning model, learning to utilize standardized raw vibration signal directly. Each of the four architectures is evaluated and a rigorous evaluation scheme consisting of an independent blind test set that approximates real-life situations is used. The findings reveal that development-set Cross-Validation (CV) performance and blind-set generalization are highly consistent, which shows that the non-regularized Conv1D-BiGRU architecture generalizes best, and it can achieve high accuracy without the intricate regularization. The findings provide a corrected and methodologically sound base of automated air-compressor fault diagnosis.


## Problem Statement
The primary issue that was overcome in the current research is the use of manual feature engineering and simplified evaluation models. Earlier literature which reports near-perfect accuracy derives large collections of handcrafted features; statistical parameters, histogram descriptors, ARMA coefficients; and expert tuning in order to generate strong performance. Although strong on this particular data, such pipelines are hard to scale and can fail to transfer to other operating conditions and equipment.
Moreover, the foregoing means that, past studies heavily rely on random train-test splits and accuracy-only reporting, which hides the actual generalization potential of the model. Many models are able to memorize temporal patterns rather than be able to learn transferable diagnostic features, even without chronological separation or blind assessment. Thus, a new standard is required; the new standard that excludes manual feature engineering and measures the performance based on strict and leakage-free blind testing. The paper bridges this methodological shortcoming by constructing automated deep sequence-learning architectures and only testing them on a completely unseen blind set.


## Application Area and Project Domain
This study falls under the interdisciplinary field of Intelligent Maintenance Systems, which combines machine learning, signal processing, and mechanical engineering.  With an emphasis on improving fault diagnosis for crucial rotating machinery, its specific application area is Industrial Predictive Maintenance.  To create a dependable framework for early fault detection that can reduce downtime and maximize maintenance operations in industrial settings, the study uses the specific use case of vibration-based condition monitoring for reciprocating air compressors to illustrate its methodology.

## What is the paper trying to do, and what are you planning to do?
This paper aims to improve fault diagnosis in reciprocating air compressors by proposing an original feature fusion method that uses statistical, histogram, and ARMA features with vibration signals in heavy lazy classifiers to achieve even greater accuracy. Future developments of this research could include testing the model's generalizability with other similar machinery, utilizing advanced fusion techniques such as deep learning-based methods.

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
-  Prediction of air compressor faults with feature fusion and machinelearning
- Link: https://doi.org/10.1016/j.knosys.2024.112519

### Reference Dataset
- https://github.com/Sangharatna786/Air-compressor-Vibration-Signals.git


## Project Technicalities

### Terminologies
- **End-to-End Deep Learning:** A machine learning paradigm employed to learn discriminative representations directly from raw vibration signals, eliminating the need for manual feature engineering.
- **Blind Test Set:** An independent evaluation dataset that is completely separated in time from the development set to ensure leakage-free assessment of generalization.
- **Conv1D (1D Convolution):** A neural network layer utilized to capture local, shift-invariant structural information and spatial features from the standardized raw input windows.
- **BiGRU (Bidirectional Gated Recurrent Unit):** A recurrent neural network component designed to learn long-range bidirectional temporal dependencies within the vibration sequences.
- **Hybrid Architecture:** A combined network design that integrates convolutional layers for feature extraction with GRU layers for temporal learning.
- **Windowing:** The data segmentation process where continuous signals are divided into fixed lengths samples with a stride to create discrete inputs for the network.
- **Stratified Cross-Validation:** A 5-fold validation technique applied to the development set to determine model stability and the optimal number of training epochs without using the blind test data.
- **StandardScaler:** A preprocessing normalization method that scales raw acceleration values to a mean of zero and standard deviation of one, fitted exclusively on the development set.
- **Categorical Cross-Entropy:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.
- **Adam Optimizer:** The optimization algorithm used to update the model weights efficiently during training.
- **OVF:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.
- **Categorical Cross-Entropy:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.
- **Categorical Cross-Entropy:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.
- **Categorical Cross-Entropy:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.
- **Categorical Cross-Entropy:** The specific loss function minimized during the training phase to optimize the multi-class classification of the five fault conditions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
