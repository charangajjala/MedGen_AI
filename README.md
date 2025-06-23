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
- Takes chest X-ray images as input (224×224×3 RGB)
- Extracts hierarchical features through dense connections
- Outputs feature maps of size (batch_size, 1024, 7, 7)
- Features are flattened to (batch_size, 49, 1024) for attention processing

**Attention Module:**
```python
class AttentionModule(nn.Module):
    def __init__(self, in_features, num_diseases, num_patches):
        # Learnable attention weights for each disease
        self.attention_weights = nn.Parameter(torch.randn(num_diseases, num_patches, 1))
    
    def forward(self, x):
        # Apply softmax to get attention distribution
        attention_weights = torch.softmax(self.attention_weights, dim=1)
        # Weighted combination of visual features per disease
        attended_features = torch.einsum('bpc,pdc->bdc', x, attention_weights)
        return attended_features, attention_weights
```

**Key Insight:** The attention mechanism learns disease-specific spatial attention patterns, allowing the model to focus on relevant regions for each of the 14 disease categories.

#### **2. Language Model Processing**

**Llama-2-7B with QLoRA:**
- Uses 4-bit quantization (NF4) to reduce memory footprint
- LoRA adapters target attention layers: `["q_proj","k_proj","v_proj","o_proj"]`
- Processes diagnostic prompts and generates medical reports
- Maintains causal language modeling capabilities

**QLoRA Configuration:**
```python
lora_r = 8          # Rank of LoRA adapters
lora_alpha = 8      # Scaling factor
lora_dropout = 0.1  # Dropout for regularization
```

#### **3. Multimodal Fusion Mechanism**

**PreCarDiv Model Architecture:**
```python
class Model_PreCarDiv(nn.Module):
    def __init__(self, visual_feature_dim, llm_model):
        # Linear projection to match LLM hidden dimension
        self.visual_projection = nn.Linear(visual_feature_dim, llm_model.config.hidden_size)
    
    def prepare_features(self, visual_features, input_ids):
        # 1. Reshape visual features: (batch, 14_diseases, 1024) → (batch*14, 1024)
        visual_features_reshaped = visual_features.view(batch_size * num_diseases, -1)
        
        # 2. Project to LLM dimension: (batch*14, 1024) → (batch*14, 4096)
        visual_embeddings_reshaped = self.visual_projection(visual_features_reshaped)
        
        # 3. Reshape back: (batch*14, 4096) → (batch, 14, 4096)
        visual_embeddings = visual_embeddings_reshaped.view(batch_size, num_diseases, -1)
        
        # 4. Get text embeddings from LLM
        text_embeddings = self.llm_model.get_input_embeddings()(input_ids)
        
        # 5. Concatenate: (batch, 14+seq_len, 4096)
        combined_embeddings = torch.cat((visual_embeddings, text_embeddings), dim=1)
        
        return visual_features, combined_embeddings
```

#### **4. How Multimodal Fusion Works**

**Step-by-Step Process:**

1. **Visual Feature Extraction:**
   - X-ray image → DenseNet-121 → 1024-dimensional features per spatial location
   - Attention mechanism applies disease-specific weights to spatial regions
   - Results in 14 disease-specific feature vectors per image

2. **Feature Alignment:**
   - Visual features (1024-dim) are projected to LLM hidden dimension (4096-dim)
   - This ensures compatibility between vision and language representations
   - Maintains disease-specific structure through reshaping operations

3. **Temporal Fusion:**
   - Visual features are prepended to text embeddings
   - Sequence: [Visual_Disease_1, Visual_Disease_2, ..., Visual_Disease_14, Text_Tokens]
   - LLM processes the combined sequence as a single input

4. **Attention-Based Integration:**
   - LLM's self-attention mechanism naturally attends to both visual and textual tokens
   - Cross-modal attention allows text generation to be influenced by visual features
   - Disease-specific visual features guide report generation

#### **5. Training Strategy**

**Custom Loss Function:**
```python
def compute_loss(self, model, inputs, return_outputs=False):
    visual_features = inputs['visual_features'].to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    labels = inputs['labels'].to(device)
    
    # Forward pass through multimodal model
    outputs = model(visual_features, input_ids, attention_mask, labels)
    loss = outputs.loss  # Standard causal language modeling loss
    
    return (loss, outputs) if return_outputs else loss
```

**Key Training Aspects:**
- **Label Masking:** Only target report tokens contribute to loss (visual tokens masked with -100)
- **Gradient Flow:** Gradients flow through both vision and language components
- **End-to-End:** All components are trained simultaneously for optimal integration

#### **6. Inference Process**

**Report Generation:**
```python
def generate(self, visual_features, input_ids, attention_mask, max_length=512):
    # 1. Prepare multimodal embeddings
    _, combined_embeddings = self.prepare_features(visual_features, input_ids)
    
    # 2. Generate text using LLM with visual context
    generated_ids = self.llm_model.generate(
        inputs_embeds=combined_embeddings,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,  # Beam search for better quality
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    return generated_ids
```

**Why This Architecture Works:**

1. **Disease-Specific Attention:** The attention mechanism allows the model to focus on different image regions for different diseases, mimicking radiologist behavior.

2. **Seamless Integration:** By projecting visual features to the same dimension as text embeddings, the LLM can naturally process both modalities.

3. **Contextual Generation:** The LLM generates reports with full awareness of both the visual findings and the diagnostic prompt.

4. **Parameter Efficiency:** QLoRA allows fine-tuning of the large LLM with minimal additional parameters while maintaining performance.

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
