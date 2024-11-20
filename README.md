# **Sentiment Analysis with Sarcasm Detection**  

![Project Banner](https://img.shields.io/badge/Status-Completed-brightgreen.svg) ![Technologies](https://img.shields.io/badge/Technologies-PyTorch%20%7C%20HuggingFace-blue)  

## **Overview**  
Sentiment Analysis is an essential part of natural language processing (NLP) that helps to identify the sentiment or emotional tone behind a series of text. Sarcasm sentiment analysis is a challenging Natural Language Processing (NLP) task aimed at detecting sarcastic expressions where the literal meaning differs from the intended sentiment. This project focuses on fine-tuning transformer-based models (Encoder, Decoder, and Hybrid architectures) to compare different transformer architectures (Encoder, Decoder and Hybrid Model) in terms of accuracy for sentiment analysis on the same dataset.
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

## **Project Workflow**  
1. **Data Preprocessing**:
   - Cleaned and tokenized text data for sarcasm-labeled datasets.  
   - Performed data augmentation.  
2. **Experiment SetUp**:
   - https://huggingface.co/datasets/stanfordnlp/imdb -Used IMDB Dataset along with manually annotated 1100 rows of sarcasm       data for better sarcasm handling
   - Fine-tuned transformer-based models (e.g., BERT, GPT-2, T5).  
   - Explored Encoder, Decoder and Hybrid architectures.  
4. **Evaluation and Analysis**:  
   - Evaluated results using different models with sarcasm-labeled datasets.
   - Analyzed metrics based on results.  

## **Results**  
The experiment compares three transformer architectures—BERT (Encoder), FLAN T5 (Hybrid), and LLAMA-2 (Decoder)—on the same custom sarcasm dataset. BERT, an encoder-based model, achieved the highest accuracy of 86.70%, indicating its superior contextual understanding for sentiment analysis. FLAN T5, a hybrid model, showed a lower accuracy of 82.10%, reflecting a trade-off between context retention and generation capabilities. LLAMA-2, a decoder-only model, had the lowest accuracy at 79.40%, suggesting limitations in capturing the nuanced language required for effective sentiment classification. Overall, BERT’s encoder-based architecture outperforms the other models in this task.

## **Project Team**  
- **Team Size**: 3 Members  
- **Roles**: Data Preprocessing, Fine Tuning 

## **How to Get this Project**  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Aastha031295/NLP-SarcasmDetection-FineTuning
   cd NLP-SarcasmDetection-FineTuning 
   ```  

## **Future Work**  
Future work may include the use of other pretrained transformers like RoBERTa model or higher end models as T5 variants to enhance sarcasm detection for sentiment analysis. The context can be improved using multi-task learning for emotion tasks 
and masking relevant words. Data augmentations such as paraphrasing are useful to introduce volatility in the training data contributing for a more resilient model. Moreover, the utilization of external cues or context-specific approaches could enforce understanding for better sarcasm detection.

## **Acknowledgments**  
Special thanks to [Hugging Face](https://huggingface.co/) for providing pre-trained transformer models and the open-source NLP community for valuable datasets and tools.
