# PropSafeVision

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)

A deep-learning toolkit for detecting inappropriate real estate images using PyTorch and Vision Transformers.

## 🚀 Overview

PropSafeVision is a proof-of-concept image moderation pipeline tailored for real estate platforms. It leverages:

- A custom PyTorch CNN classifier trained on a 15,000-image dataset, achieving **87% accuracy**  
- Experimental integration of Hugging Face’s Vision Transformer (ViT) for potential performance gains

This project helps real estate platforms automatically flag and filter user-uploaded images that violate content guidelines (e.g., inappropriate, irrelevant, or low-quality images).

## 📂 Repository Structure

```
PropSafeVision/
│
├── data/                   
│   ├── train/              
│   ├── val/                
│   └── test/               
│
├── src/                    
│   ├── train.py            
│   ├── eval.py            
│   ├── models/             
│   ├── data/               
│   └── utils/              
│
├── notebooks/              
│   └── vit_experiments.ipynb  
│
├── checkpoints/            
│   └── cnn/                
│
├── requirements.txt        
├── LICENSE                 
└── README.md               
```

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**: Check with `python3 --version`.
- **pip**: Python package manager.
- **Git**: For cloning the repository.
- **CUDA** (optional): If training on GPU, ensure CUDA and cuDNN are installed and compatible with PyTorch.

## 🔧 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MichaelLeHoang/PropSafeVision.git
   cd PropSafeVision
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**  
   - Place your training, validation, and test images in the following structure:
     ```
     data/
     ├── train/
     │   ├── safe/
     │   └── inappropriate/
     ├── val/
     │   ├── safe/
     │   └── inappropriate/
     └── test/
         ├── safe/
         └── inappropriate/
     ```
   - Create an annotation file (e.g., `labels.csv`) in the `data/` directory with the following format:
     ```
     filename,label
     train/safe/image1.jpg,0
     train/inappropriate/image2.jpg,1
     val/safe/image3.jpg,0
     ...
     ```
     - `label`: 0 for safe, 1 for inappropriate.

## ▶️ Usage

### 1. Train the CNN Model
Train the custom CNN model using the following command:
```bash
python src/train.py \
  --arch cnn \
  --data-dir data \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir checkpoints/cnn \
  --device cuda  # Use 'cpu' if no GPU is available
```

### 2. Evaluate on the Test Set
Evaluate the trained model on the test dataset:
```bash
python src/eval.py \
  --model-path checkpoints/cnn/best.pth \
  --data-dir data/test \
  --device cuda
```

### 3. Run Vision Transformer (ViT) Experiments
Explore the ViT model using the provided Jupyter notebook:
```bash
jupyter notebook notebooks/vit_experiments.ipynb
```
Follow the notebook instructions to fine-tune the Hugging Face ViT model on your dataset.

## 📈 Results

| Model                  | Accuracy |
|------------------------|----------|
| Custom CNN             | 87%      |
| HuggingFace ViT (preliminary) | 89% (ongoing) |

## 🛠️ Configuration Options

All scripts (`train.py`, `eval.py`) support the following command-line arguments:

- `--arch`: Model architecture (`cnn` or `vit`)
- `--data-dir`: Root directory of your image folders
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--output-dir`: Directory to save model checkpoints
- `--device`: Device to use (`cpu` or `cuda`)

Run `python src/train.py --help` or `python src/eval.py --help` for a full list of options.

## 🔍 Troubleshooting

- **CUDA Out of Memory**: Reduce the `--batch-size` or switch to `--device cpu`.
- **ModuleNotFoundError**: Ensure all dependencies are installed (`pip install -r requirements.txt`) and the virtual environment is activated.
- **Data Loading Errors**: Verify the `data/` directory structure and `labels.csv` file format.
- **Jupyter Notebook Not Found**: Install Jupyter with `pip install jupyter` and try again.

## 🔮 Future Work

- **Hyperparameter Tuning**: Use Optuna for automated hyperparameter optimization.
- **Data Augmentation**: Integrate advanced augmentation pipelines with Albumentations.
- **Ensemble Models**: Combine CNN and ViT for better performance.
- **Deployment**: Deploy the model as a REST API using FastAPI or TorchServe.

## 🤝 Contributing

We welcome contributions! Follow these steps to contribute:

1. **Fork the repository**  
   Click the "Fork" button at the top right of the repository page.

2. **Clone your fork**  
   ```bash
   git clone https://github.com/yourusername/PropSafeVision.git
   cd PropSafeVision
   ```

3. **Create a feature branch**  
   ```bash
   git checkout -b feature/YourFeature
   ```

4. **Make your changes**  
   - Follow the existing code style (PEP 8 for Python).
   - Add tests for new features in the `src/tests/` directory.
   - Update documentation if necessary.

5. **Commit your changes**  
   ```bash
   git commit -m 'Add new feature: YourFeature'
   ```

6. **Push to your fork**  
   ```bash
   git push origin feature/YourFeature
   ```

7. **Open a Pull Request**  
   - Go to the original repository and click "New Pull Request".
   - Describe your changes in detail and link any related issues.

### Contribution Guidelines
- Ensure your code passes all tests (`pytest src/tests/`).
- Keep commits atomic and descriptive.
- Address any feedback during the PR review process.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, feel free to open an issue or contact the maintainers at [m4le@uwaterloo.ca](mailto:m4le@uwaterloo.ca).

