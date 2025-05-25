# 📝 Sentiment Analysis on Amazon Fine Food Reviews

This project leverages both classical natural language processing (NLP) techniques and modern transformer-based deep learning models to perform sentiment analysis on product reviews. Specifically, it contrasts the performance of rule-based models like VADER with transformer architectures such as RoBERTa using the Amazon Fine Food Reviews dataset.

---

## 📌 Project Overview

**Objective:**  
To classify product reviews into three sentiment categories—**positive**, **neutral**, or **negative**—using a combination of rule-based and deep learning-based approaches.

**Methods Implemented:**
- Rule-based sentiment analysis using **VADER** (NLTK)
- Context-aware classification using **RoBERTa** (via Hugging Face Transformers)

---

## 📂 Dataset

**Source:** [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

**Dataset Highlights:**
- Over 568,000 customer reviews
- Notable fields include: `Text`, `Score`, `Summary`, and `Time`

---

## 🧰 Tools & Technologies

| Category           | Tools & Libraries                                |
|--------------------|--------------------------------------------------|
| Data Processing    | `pandas`, `numpy`                                 |
| Visualization      | `matplotlib`, `seaborn`                           |
| Natural Language Processing | `nltk`, `VADER`                         |
| Deep Learning      | `transformers` (RoBERTa)                          |
| Utilities          | `tqdm`, `TensorFlow` (for hardware/GPU checks)   |

---

## 🔍 Workflow Summary

### 1. **Data Preprocessing & Exploratory Data Analysis (EDA)**
- Load and inspect the dataset
- Visualize distribution of star ratings
- Explore temporal trends and token usage

### 2. **VADER Sentiment Analysis**
- Apply POS tagging and tokenization
- Use `SentimentIntensityAnalyzer` to compute sentiment scores
- Categorize reviews based on compound polarity

### 3. **RoBERTa-Based Sentiment Analysis**
- Utilize Hugging Face's `pipeline("sentiment-analysis")`
- Generate predictions from the model
- Perform comparative evaluation with VADER outputs

---

## 📊 Sample Visualizations

- **Review Score Distribution:** Histogram of star ratings  
- **POS Tagging & Entity Recognition:** Visual breakdown of token types  
- **Sentiment Output Comparison:** Side-by-side predictions from VADER and RoBERTa  

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn nltk transformers tqdm
```

### 2. Download Dataset
Place the `Reviews.csv` file into the root directory. You can download it from the Kaggle link above.

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook sentiment-analysis.ipynb
```

### 4. Download Required NLTK Resources (if needed)
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

---

## 📈 Results & Key Observations

| Model   | Strengths                                      | Limitations                                |
|---------|------------------------------------------------|---------------------------------------------|
| VADER   | Lightweight, fast, interpretable               | Less effective on complex or nuanced texts  |
| RoBERTa | High contextual accuracy, robust performance   | Requires more computational resources       |


