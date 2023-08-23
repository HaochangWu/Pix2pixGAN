# Received Signal Strength Reconstruction Using Pix2pix Generative Adversarial Network

## Introduction
In this project, we propose a pix2pix generative adversarial network (pix2pix GAN) based RSS reconstruction model. This model can generate a dense RSS fingerprint map based only on the location information of the wireless access point (AP). The accuracy and efficiency of the proposed model are verified in an indoor environment.

## Installation

### Prerequisites
- **Software**: Jupyter Notebook recommended or Anaconda.
- **Libraries**: numpy, pandas, sklearn, os, opencv, matplotlib, tensorflow, copy, glob.

## Usage & Code Structure

### Overview
This repository includes the pix2pix GAN architecture code, training data, test results, result visualization, the main program for the GAN model, and all other related files. Detailed annotations are made within each code section to explain the function of each code block.

### Structure
- **MAIN.ipynb**: The main training program for the GAN model. Simply run this file to begin training.
- **Module Folder**: Contains the structural code for the components of the GAN model, including the generator and discriminator.
- **Static Folder**: Consists of a "png" folder containing all training image data, and the optimal generator and discriminator models after training.
- **Result_Already Folder**: Contains the test images for each training round, indicating the number of epochs of training. Each image consists of three parts (from left to right: input image, generated image, ground truth image).
- **Result Folder**: Used to save a random image from the test set after each epoch during training to visualize the performance of the Pix2pix GAN model.
- **Visualization.ipynb**: Program for visualizing the results.
- **Predict.ipynb**: Accepts any input image you wish to predict and outputs the Pix2pix GAN's prediction.
- **Note**: "MAIN.py", "Visualization.py", and "Predict.py" are identical to the above three files but run on different software.

## Contributing
Researchers interested in contributing to this model may consider the following research directions:
1. How to optimize and theoretically support the comparison of two loss functions in the GAN model.
2. Methods for selecting input locations to achieve the optimal prediction model with minimal training data.
3. The impact of the number of layers in the generator (Unet++) on this type of training data.

## Contact
For any questions or inquiries, please email [haochang.wu@ucdconnect.ie](mailto:haochang.wu@ucdconnect.ie).
# Received Signal Strength Reconstruction Using Pix2pix Generative Adversarial Network

## Introduction
In this project, we propose a pix2pix generative adversarial network (pix2pix GAN) based RSS reconstruction model. This model can generate a dense RSS fingerprint map based only on the location information of the wireless access point (AP). The accuracy and efficiency of the proposed model are verified in an indoor environment.

## Installation

### Prerequisites
- **Software**: Jupyter Notebook recommended or Anaconda.
- **Libraries**: numpy, pandas, sklearn, os, opencv, matplotlib, tensorflow, copy, glob.

## Usage & Code Structure

### Overview
This repository includes the pix2pix GAN architecture code, training data, test results, result visualization, the main program for the GAN model, and all other related files. Detailed annotations are made within each code section to explain the function of each code block.

### Structure
- **Module Folder**: Contains the structural code for the components of the GAN model, including the generator and discriminator.
- **Static Folder**: Consists of a "png" folder containing all training image data, and the optimal generator and discriminator models after training.
- **Result_Already Folder**: Contains the test images for each training round, indicating the number of epochs of training. Each image consists of three parts (from left to right: input image, generated image, ground truth image).
- **Result Folder**: Used to save a random image from the test set after each epoch during training to visualize the performance of the Pix2pix GAN model.
- **MAIN.ipynb**: The main training program for the GAN model. Simply run this file to begin training.
- **Visualization.ipynb**: Program for visualizing the results.
- **Predict.ipynb**: Accepts any input image you wish to predict and outputs the Pix2pix GAN's prediction.
- **Note**: "MAIN.py", "Visualization.py", and "Predict.py" are identical to the above three files but run on different software.

## Contributing
Researchers interested in contributing to this model may consider the following research directions:
1. How to optimize and theoretically support the comparison of two loss functions in the GAN model.
2. Methods for selecting input locations to achieve the optimal prediction model with minimal training data.
3. The impact of the number of layers in the generator (Unet++) on this type of training data.

## Contact
For any questions or inquiries, please email [haochang.wu@ucdconnect.ie](mailto:haochang.wu@ucdconnect.ie).
