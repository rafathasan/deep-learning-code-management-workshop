
# Workshop on Hands-on Deep Learning Coding & Code Management

  

Welcome to the repository for the "Workshop on Hands-on Deep Learning Coding & Code Management," organized by the Center for Computational & Data Sciences (CCDS.ai) at Independent University Bangladesh (IUB) . This repository contains the code and materials for Session 2, focusing on Deep Learning Code Management.

  


## 📅 Event Details
-  **Event Name:** Workshop on Hands-on Deep Learning Coding & Code Management 
-  **Organized by:** Center for Computational & Data Sciences ([CCDS](https://ccds.ai/)) at Independent University Bangladesh ([IUB](https://www.iub.edu.bd/)) 
-  **Session:** Day 2 - Session 2 - Deep Learning Code Management 
-  **Event Link:** [Workshop Event](https://ccds.ai/event/workshop-on-hands-on-deep-learning-coding-code-management/)

## 🌐 Session Overview

  

In this session, we delve into effective code management practices for deep learning projects. The code for the session is implemented in Python using PyTorch and PyTorch Lightning framework, utilizing various packages for data processing and model training.

  

## 📂 Project Structure

  

-  **`code/`**: Contains the code examples and exercises for Session 2.

-  **`data/`**: Data modules with train, val, and test dataloaders, including data preparation and splitting files using PyTorch Lightning.

-  **`configs/`**: YAML files containing arguments for different components.

-  **`models/`**: Model definitions for training, validation, and testing, as well as logic for optimizer, LR scheduler, score metric, and loggers.

-  **`utils/`**: Necessary functions and classes to load config files and other utilities.

  

-  **`cli.py`**: Command line interface for training and testing experiments.

  

-  **`__data__/`** (optional): Temporary folder to download the dataset.

-  **`__logs__/`** (optional): Temporary folder to store checkpoints, logs, and other experiment-related information.

  

## 🛠️ Libraries and Dependencies

  

The project is implemented in Python using PyTorch and PyTorch Lightning framework, along with the following packages:

  

-  `albumentations==1.3.1`

-  `click==8.1.7`

-  `numpy==1.26.4`

-  `Pillow==9.3.0`

-  `Pillow==10.2.0`

-  `pytorch_lightning==2.0.9.post0`

-  `PyYAML==6.0.1`

-  `torch==2.1.1+cu118`

-  `torchmetrics==1.2.0`

-  `torchvision==0.16.1+cu118`

  

## 🚀 Getting Started

  

1. Clone this repository to your local machine:

  

```bash

git clone https://github.com/your-username/deep-learning-code-management-workshop.

```

  

2. Navigate to the project directory:

  

```bash

cd deep-learning-code-management-workshop

```

  

3. Explore the relevant folders, such as code/ and configs/, for session-specific content.


## 💻 Usage

Use the `cli.py` command line interface to train and test experiments. For example:

```bash

python cli.py train --config configs/train_config.yaml
python cli.py test --config configs/test_config.yaml`
```
## 🤝 Contributions

If you have any improvements, suggestions, or corrections, feel free to open an issue or submit a pull request. Your contributions are highly appreciated!

## 📝 License

This project is licensed under the MIT License.

Happy coding!🎉
