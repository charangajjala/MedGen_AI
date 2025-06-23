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

![](Assets/2025-06-22-22-08-55.png)

### 🔬 Detailed Architecture Explanation

#### **1. Vision Processing Pipeline**

**DenseNet-121 Encoder:**
- Takes chest X-ray images as input (RGB, 224×224 pixels)
- Extracts hierarchical features through convolutional layers
- Outputs feature maps of shape (batch_size, 1024, 7, 7)

**Attention Module:**
The attention module processes spatial features from DenseNet and applies disease-specific attention weights. It takes the flattened spatial features and computes attention weights for each of the 14 diseases across 49 spatial patches. The output provides disease-specific visual features with shape (batch_size, 14, 1024).

**Key Features:**
- **49 patches**: 7×7 spatial grid from DenseNet features
- **14 diseases**: Each disease gets its own attention weights
- **Disease-specific attention**: Different attention patterns for different pathologies
- **Output**: (batch_size, 14, 1024) - disease-specific visual features

#### **2. Language Model Processing**

**Llama-2-7B with QLoRA:**
- **4-bit Quantization**: Reduces memory footprint by 75%
- **LoRA Adaptation**: Only trains 12.5M parameters instead of 7B
- **Target Modules**: q_proj, k_proj, v_proj, o_proj in attention layers

**Text Processing:**
The system combines diagnostic prompts with target reports for training. The combined input is tokenized and processed through the language model's embedding layer to create text embeddings.

#### **3. Multimodal Fusion Mechanism**

**PreCarDiv Model Architecture:**
The model includes a visual projection layer that maps visual features to the language model's hidden dimension, enabling seamless integration of visual and textual information.

**Fusion Process:**

1. **Visual Feature Projection:**
   The disease-specific visual features are reshaped and projected to match the language model's hidden dimension using a linear transformation layer.

2. **Text Embedding Extraction:**
   Text embeddings are extracted from the language model's embedding layer, creating representations that can be combined with visual features.

3. **Multimodal Concatenation:**
   Visual and textual embeddings are concatenated along the sequence dimension, creating a combined multimodal representation that the language model can process.

#### **4. Training Strategy**

**Custom Loss Function:**
The training uses a custom loss function that only computes loss on the target report tokens, ignoring visual tokens and prompt tokens by masking them with -100. This ensures the model learns to generate reports while being conditioned on visual features.

**Training Configuration:**
The model uses QLoRA parameters including rank 8 for LoRA adaptation, alpha scaling factor of 8, and dropout of 0.1 for regularization. Training parameters include a learning rate of 5e-4, batch size of 1 per device, and gradient clipping at 0.3.

#### **5. Inference Process**

**Report Generation:**
During inference, the model prepares multimodal embeddings by combining visual features with text embeddings. It then uses the language model's generation capabilities with beam search (5 beams) and n-gram repetition prevention to generate comprehensive medical reports.

### 🔄 Data Flow Summary

1. **Input Processing:**
   - X-ray image → DenseNet-121 → Spatial features
   - Diagnostic prompt → Tokenization → Text embeddings

2. **Attention Mechanism:**
   - Spatial features → Disease-specific attention → 14 disease features
   - Each disease gets weighted attention across 49 spatial patches

3. **Multimodal Fusion:**
   - Visual features → Linear projection → Language model dimension
   - Concatenate with text embeddings → Combined multimodal representation

4. **Generation:**
   - Combined embeddings → Llama-2-7B → Autoregressive text generation
   - Beam search with n-gram repetition prevention

### 🎯 Key Innovations

1. **Disease-Specific Attention**: Unlike standard vision transformers, this model learns disease-specific attention patterns
2. **Efficient Multimodal Fusion**: Direct concatenation with learned projection instead of complex cross-attention
3. **Parameter-Efficient Training**: QLoRA enables fine-tuning of large models with limited resources
4. **End-to-End Training**: Single model trained jointly for vision and language tasks

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
   - Uncomment and add your token in the notebook

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

The model can be used for inference by loading the trained model weights and generating reports from new X-ray images. The process involves preprocessing the input image, creating a diagnostic prompt, and using the model's generation capabilities to produce comprehensive medical reports.

## 📈 Performance

- **Model Parameters**: 12,587,008 trainable parameters
- **Training**: 20 epochs with early stopping
- **Evaluation**: BLEU score for report quality
- **Memory Usage**: Optimized with 4-bit quantization

## 🔧 Configuration

### QLoRA Parameters
The model uses LoRA rank of 8, alpha scaling factor of 8, and dropout probability of 0.1 for regularization.

### Training Parameters
Training configuration includes 20 epochs, learning rate of 5e-4, batch size of 1 per device, and gradient clipping at 0.3.

### Quantization Settings
The model employs 4-bit precision with float16 compute dtype and nf4 quantization type for memory efficiency.

## 📚 Documentation

### Presentation Materials
- **Project Presentation**: `Precardiv ppt-G0010.pdf` - Comprehensive project overview and methodology
- **Research Poster**: `Precardiv_Poster-G0010.pdf` - Concise research summary and results

### Key Classes

#### PreCarDiv_Dataset
Custom dataset class for multimodal medical data that handles X-ray images and text pairs, implements attention-based feature extraction, and manages tokenization and label creation.

#### Model_PreCarDiv
Main multimodal model that combines visual and textual features, implements custom forward pass, and provides generation capabilities.

#### CustomTrainer
Specialized trainer for multimodal training that handles custom loss computation, manages device placement, and supports evaluation during training.

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
