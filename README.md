# **Sentiment Analysis with Sarcasm Detection**  

![Project Banner](https://img.shields.io/badge/Status-Completed-brightgreen.svg) ![Technologies](https://img.shields.io/badge/Technologies-PyTorch%20%7C%20HuggingFace-blue)  

## **Overview**  
Sarcasm sentiment analysis is a challenging Natural Language Processing (NLP) task aimed at detecting sarcastic expressions where the literal meaning differs from the intended sentiment. This project focuses on fine-tuning transformer-based models (Encoder, Decoder, and Hybrid architectures) to improve sentiment classification in sarcastic text.  

## **Features**  
- Fine-tuned transformer architectures for sarcasm detection: Encoder-based, Decoder-based, and Hybrid models.  
- Preprocessing pipeline for tokenization, text cleaning, and data augmentation.  
- Comparative evaluation of transformer models based on accuracy, precision, recall, and F1 scores.  
- Use of sarcasm-labeled datasets for training and testing.  

## **Technologies Used**  
- **Programming Language**: Python  
- **Libraries/Frameworks**:  
  - PyTorch  
  - Hugging Face Transformers  
  - TensorFlow  
  - Scikit-learn  
  - Pandas & NumPy  

## **Repository Highlights**  
- `/src/`: Contains all the code for data preprocessing, model fine-tuning, and evaluation.  
- `/notebooks/`: Jupyter notebooks for experiments and visualizations.  
- `/data/`: Sample sarcasm-labeled datasets (link to download full datasets).  
- `/models/`: Saved models and checkpoints.  

## **Project Workflow**  
1. **Data Preprocessing**:  
   - Cleaned and tokenized text data for sarcasm-labeled datasets.  
   - Performed text augmentation and embedding extraction.  
2. **Model Development**:  
   - Fine-tuned transformer-based models (e.g., BERT, GPT-2, T5).  
   - Explored Encoder, Decoder, and Hybrid architectures.  
3. **Evaluation and Analysis**:  
   - Conducted model evaluation with sarcasm-labeled datasets.  
   - Analyzed metrics and refined models based on results.  

## **Results**  
- Achieved a notable improvement in sarcasm sentiment classification compared to baseline models.  
- The Hybrid architecture demonstrated the best performance, with a balanced trade-off between accuracy and speed.  

## **Project Team**  
- **Team Size**: 3 Members  
- **Roles**: Model Architect, Data Preprocessor, Evaluator  

## **How to Run the Project**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Aastha031295/FineTuning  
   cd FineTuning  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Download the dataset (link provided in `/data/README.md`) and place it in the `/data/` folder.  
4. Run the training script:  
   ```bash  
   python train.py --model_type encoder  
   ```  
5. Evaluate the model:  
   ```bash  
   python evaluate.py --model_path /models/saved_model.pt  
   ```  

## **Future Work**  
- Explore multilingual sarcasm detection for non-English datasets.  
- Incorporate domain-specific sarcasm datasets for improved model generalizability.  


## **Acknowledgments**  
Special thanks to [Hugging Face](https://huggingface.co/) for providing pre-trained transformer models and the open-source NLP community for valuable datasets and tools.
