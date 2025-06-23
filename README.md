# MedGen_AI: Multimodal Medical Report Generation

## 🏥 Project Overview

MedGen_AI is an advanced multimodal AI system that generates comprehensive medical reports from chest X-ray images. The system combines computer vision and natural language processing to analyze radiographic images and produce detailed diagnostic reports, leveraging the power of Large Language Models (LLMs) and attention mechanisms.

## 🎯 Problem Statement

Medical report generation from X-ray images is a critical task that requires:
- Accurate interpretation of complex radiographic features
- Generation of comprehensive, clinically relevant reports
- Integration of visual and textual information
- Efficient processing for clinical workflow integration

## 🏗️ Architecture

### Core Components

1. **Vision Encoder**: DenseNet-121 with Attention Module
   - Pre-trained on ImageNet
   - Custom attention mechanism for disease-specific feature extraction
   - 14 disease categories with attention weights

2. **Language Model**: Llama-2-7B with QLoRA
   - 4-bit quantization for memory efficiency
   - LoRA fine-tuning for parameter-efficient training
   - Causal language modeling for report generation

3. **Multimodal Fusion**: PreCarDiv Model
   - Visual feature projection to language model dimension
   - Attention-based feature combination
   - End-to-end training with custom loss function

### Model Architecture Details

```
Input X-ray → DenseNet-121 → Attention Module → Visual Features
                                                    ↓
Diagnostic Prompt → Llama-2-7B → Text Embeddings → Fusion Layer
                                                    ↓
                                              Combined Embeddings
                                                    ↓
                                              Report Generation
```

## 📊 Dataset

- **Source**: CheXpert+ dataset with preprocessed chest X-ray images
- **Size**: 100 samples (configurable)
- **Split**: 60% train, 20% validation, 20% test
- **Features**: 
  - X-ray images (RGB, normalized)
  - Diagnostic prompts
  - Target medical reports
  - 14 disease categories

## 🚀 Features

- **Multimodal Integration**: Seamless fusion of visual and textual information
- **Attention Mechanism**: Disease-specific attention for improved accuracy
- **Parameter Efficiency**: QLoRA fine-tuning with 4-bit quantization
- **Memory Optimization**: Gradient checkpointing and mixed precision training
- **Custom Training**: Specialized trainer for multimodal data
- **Evaluation Metrics**: BLEU score for report quality assessment

## 📋 Requirements

### System Requirements
- CUDA-compatible GPU (recommended)
- Python 3.8+
- 16GB+ RAM (for full model loading)

### Python Dependencies
```bash
pip install transformers sentencepiece transformers[sentencepiece] accelerate datasets peft trl bitsandbytes torch torchvision pillow pandas numpy opencv-python tqdm scikit-learn matplotlib sacrebleu
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/charangajjala/MedGen_AI.git
   cd MedGen_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face token**
   - Get your Hugging Face token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Uncomment and add your token in the notebook:
   ```python
   login(token="YOUR_HF_TOKEN_HERE")
   ```

4. **Prepare dataset**
   - Place your X-ray images in the appropriate directory
   - Update the CSV file path in the notebook
   - Ensure the dataset follows the expected format

## 📖 Usage

### Training the Model

1. **Open the notebook**
   ```bash
   jupyter notebook multimodal_final.ipynb
   ```

2. **Configure parameters**
   - Adjust training parameters in Cell 3
   - Modify model configurations as needed
   - Set appropriate batch sizes for your hardware

3. **Run training**
   - Execute cells sequentially
   - Monitor training progress and loss
   - Check validation metrics

### Inference

```python
# Load trained model
model = Model_PreCarDiv(visual_feature_dim, llm_model)
model.load_state_dict(torch.load('path_to_checkpoint.pth'))

# Generate report
image = load_and_preprocess_xray('path_to_xray.jpg')
diagnostic_prompt = "Describe the findings in this chest X-ray."
report = model.generate(image, diagnostic_prompt)
```

## 📈 Performance

- **Model Parameters**: 12,587,008 trainable parameters
- **Training**: 20 epochs with early stopping
- **Evaluation**: BLEU score for report quality
- **Memory Usage**: Optimized with 4-bit quantization

## 🔧 Configuration

### QLoRA Parameters
```python
lora_r = 8                    # LoRA attention dimension
lora_alpha = 8               # Alpha parameter for LoRA scaling
lora_dropout = 0.1           # Dropout probability
```

### Training Parameters
```python
num_train_epochs = 20        # Number of training epochs
learning_rate = 5e-4         # Initial learning rate
per_device_train_batch_size = 1  # Batch size per GPU
max_grad_norm = 0.3          # Gradient clipping
```

### Quantization Settings
```python
use_4bit = True              # 4-bit precision
bnb_4bit_compute_dtype = "float16"  # Compute dtype
bnb_4bit_quant_type = "nf4"  # Quantization type
```

## 📚 Documentation

### Presentation Materials
- **Project Presentation**: `Precardiv ppt-G0010.pdf` - Comprehensive project overview and methodology
- **Research Poster**: `Precardiv_Poster-G0010.pdf` - Concise research summary and results

### Key Classes

#### PreCarDiv_Dataset
Custom dataset class for multimodal medical data:
- Handles X-ray images and text pairs
- Implements attention-based feature extraction
- Manages tokenization and label creation

#### Model_PreCarDiv
Main multimodal model:
- Combines visual and textual features
- Implements custom forward pass
- Provides generation capabilities

#### CustomTrainer
Specialized trainer for multimodal training:
- Handles custom loss computation
- Manages device placement
- Supports evaluation during training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **G.C. Charan** - *Initial work* - [charangajjala](https://github.com/charangajjala)

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and model access
- **Meta AI** for Llama-2-7B model
- **CheXpert+** dataset contributors
- **PyTorch** community for deep learning framework

## 📞 Contact

- **Email**: charangajjala7@gmail.com
- **GitHub**: [@charangajjala](https://github.com/charangajjala)

## 🔗 Links

- **Repository**: https://github.com/charangajjala/MedGen_AI
- **Hugging Face**: https://huggingface.co/meta-llama/Llama-2-7b-hf
- **Dataset**: CheXpert+ (requires access permissions)

---

**Note**: This project requires appropriate access permissions for the Llama-2-7B model and medical datasets. Please ensure compliance with data usage agreements and medical data privacy regulations. 