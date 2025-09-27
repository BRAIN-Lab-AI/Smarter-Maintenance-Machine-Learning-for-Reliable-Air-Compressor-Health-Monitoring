# Smarter-Maintenance-Machine-Learning-for-Reliable-Air-Compressor-Health-Monitoring

## Project Metadata
### Authors
- **Team:** Khalid Alyahya, Abdullah Alsagoor
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Air compressors are essential to several industries, including manufacturing and energy, and their operation needs to be reliable. It is critically important to have a way to detect mechanical failures in compressor systems to prevent operational downtime and unsafe working conditions.

Traditional methods of fault detection, such as routine checks by operators, are being recognized as inadequate due to their subjective nature and inefficiency. Although data-driven approaches for detection based on vibration analysis offer potential improvements, many models are based on historical data for rather simplistic and limited mechanical fault types. Furthermore, their performance is restricted in practice because of the inherent difficulty of distinguishing which curated dynamic signal features are most correlated to compressor condition when the data is highly complex.

This means there is a need to develop progressively 'smarter' diagnostic systems. Better informing models and decision making by utilizing different signal feature classifications, also known as feature fusion, will allow a better assessment of compressor state. This advancement is important for transitioning maintenance programs from reactive to predictive maintenance, providing improved reliability and reduced costs.

Meta Data: Fault/Operating Mode (GOOD, IVF, OVF, IOVF), sampling range (10,000 samples or 0.0001s based readings), amplitude units m/s2, sensor configuration (Dytran 3055 B1), compressor rating (power supply, service pressure), data acquisition system configurations (NI-4432)

## Problem Statement
At present, many existing fault diagnosis techniques for machines are generally inaccurate due to relying on limited types of characteristics and are generally not applicable for use across different machinery types, or in some instances to a narrow extent of offerings. This research project minimizes these limitations by providing a feature fusion prediction technique, where statistical features, histogram features, and ARMA features for air compressor fault prediction, shift some of the predictive accuracy base. However, the present shift in design and methodology still hinges on the questions of whether it will be generalizable across multi-stage compressors in practice, and whether predictive generalizability can be claimed or proven.

## Application Area and Project Domain
This study falls under the interdisciplinary field of Intelligent Maintenance Systems, which combines machine learning, signal processing, and mechanical engineering.  With an emphasis on improving fault diagnosis for crucial rotating machinery, its specific application area is Industrial Predictive Maintenance.  To create a dependable framework for early fault detection that can reduce downtime and maximize maintenance operations in industrial settings, the study uses the specific use case of vibration-based condition monitoring for reciprocating air compressors to illustrate its methodology.

## What is the paper trying to do, and what are you planning to do?
This paper aims to improve fault diagnosis in reciprocating air compressors by proposing an original feature fusion method that uses statistical, histogram, and ARMA features with vibration signals in heavy lazy classifiers to achieve even greater accuracy. Future developments of this research could include testing the model's generalizability with other similar machinery, utilizing advanced fusion techniques such as weighted or deep learning-based methods, incorporating data modalities beside vibration signals like the acoustic or thermal signals, and ultimately, building an end-to-end real-time predictive maintenance system for industry use.

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

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
