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

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report (PDF):** [Project Report](/report.pdf)
- **Report (Editable Version):** [Project Report Editable Version](/report_Editable_Version.docx)
- **Code:** [Project Code](/Code.ipynb)

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
- **OVF:** Outlet valve fluttering.
- **IOVF:** Inlet-outlet valve fluttering.
- **IVF:** Inlet valve fluttering.
- **CVF:** Check valve fault.

### Problem Statements
- **Problem 1:** There is a need to improve the accuracy of fault diagnosis in reciprocating air compressors beyond what current methods achieve.
- **Problem 2:** The generalizability of the proposed model to other similar types of machinery has not yet been tested or validated.
- **Problem 3:** Current approaches have not yet fully utilized advanced fusion techniques, such as deep learning-based methods, to enhance performance.

### Loopholes or Research Areas
- **Feature Dependence:** Reliance on handcrafted features limits portability and risks encoding dataset artifacts.
- **Evaluation Methodology:** Usage of random splits ignores time dependence, causing data leakage and inflating accuracy.
- **Generalization Assessment:** Absence of unseen blind-set testing fails to distinguish between true learning and dataset memorization.

### Proposed Solution
The solution includes:

- **Automated Learning:** Replaces manual feature engineering with deep sequence-learning models to learn discriminative representations directly from raw signals.
- **Pipeline Architecture:** Utilizes convolutional layers for automated local feature extraction combined with a BiGRU layer to capture hierarchical temporal relationships.

## Model Workflow
The workflow of the Automated Deep Sequence-Learning Model is designed to detect air compressor faults directly from raw vibration signals, eliminating the need for manual feature engineering:

1. **Input:**
   - **Raw Signal Acquisition:** The model accepts continuous blocks of raw vibration signals collected from a single-stage reciprocating air compressor.
   - **Preprocessing and Windowing:** The raw signals are standardized (normalized to mean 0 and standard deviation 1) and segmented into fixed windows of 150 samples with a stride of 10.
   - **Chronological Splitting:** To ensure realistic testing, the data is strictly divided by time into a development set (4,430 windows) and a completely unseen, non-overlapping blind test set (430 windows).

2. **Deep Learning Process:**
   - **Automated Feature Extraction:** The input windows (shaped as Batch, 1, 150) are fed into Conv1D layers. These layers act as automated feature extractors, capturing local, shift-invariant structural information and vibration spikes.
   - **Temporal Sequence Learning:** The output from the convolutional layers is passed to Bidirectional GRU (BiGRU) layers. These layers learn the bidirectional temporal context and long-range dependencies within the signal sequence.
   - **Optimization:** The model (specifically the best-performing Model 4) is trained using the Adam optimizer and categorical cross-entropy loss, utilizing 5-fold stratified cross-validation to determine the optimal number of epochs.

3. **Output:**
   - **Fault Classification:** The final processed representation is decoded to classify the equipment state into one of five categories: CVF (Control Valve Fault), GOOD (Healthy), IOVF (Inlet/Outlet Valve Fault), IVF (Inlet Valve Fault), or OVF (Outlet Valve Fault).

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch and other libraries for their amazing work.
