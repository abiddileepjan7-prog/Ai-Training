# 📝 NLP Text Preprocessing Project

A beginner-friendly **Natural Language Processing (NLP)** project built with **Python** that demonstrates the complete text preprocessing pipeline along with popular text vectorization techniques used in Machine Learning.

---

## 🚀 Features

This project performs the following NLP tasks:

- ✅ Text Preprocessing
  - Convert text to lowercase
  - Remove punctuation
  - Remove extra whitespaces

- ✅ Tokenization
  - Split text into individual words using NLTK

- ✅ Stopword Removal
  - Remove common English stopwords

- ✅ Stemming
  - Reduce words to their root form using Porter Stemmer

- ✅ Lemmatization
  - Convert words to their dictionary base form using WordNet Lemmatizer

- ✅ Bag of Words (BoW)
  - Convert text into numerical vectors using word frequency

- ✅ TF-IDF
  - Generate weighted numerical representations of text

- ✅ Word2Vec *(Optional)*
  - Learn dense word embeddings using Gensim

---

# 📂 Project Structure

```
nlp_preprocessing/
│
├── main.py
├── preprocessing.py
├── tokenizer.py
├── stopwords_removal.py
├── stemmer.py
├── lemmatizer.py
├── bag_of_words.py
├── tfidf.py
├── wordtovector.py
├── setup.py
├── requirements.txt
└── README.md
```

---

# 🔄 Workflow

```
                User Input
                     │
                     ▼
          Text Preprocessing
                     │
                     ▼
              Tokenization
                     │
                     ▼
          Stopword Removal
                     │
                     ▼
               Stemming
                     │
                     ▼
             Lemmatization
                     │
                     ▼
        Feature Extraction
          ├── Bag of Words
          ├── TF-IDF
          └── Word2Vec
```

---

# 📊 Example

## Input

```
Hello, WORLD!!! Welcome to Natural Language Processing.
```

## Output

```
Original Text
--------------
Hello, WORLD!!! Welcome to Natural Language Processing.

Cleaned Text
--------------
hello world welcome to natural language processing

Tokens
--------------
['hello', 'world', 'welcome', 'to', 'natural', 'language', 'processing']

After Stopword Removal
--------------
['hello', 'world', 'welcome', 'natural', 'language', 'processing']

Stemmed Words
--------------
['hello', 'world', 'welcom', 'natur', 'languag', 'process']

Lemmatized Words
--------------
['hello', 'world', 'welcome', 'natural', 'language', 'processing']
```

---

# 🛠️ Technologies Used

- Python
- NLTK
- Scikit-learn
- Gensim (Word2Vec)
- VS Code

---

# 📦 Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/nlp_preprocessing.git
```

### Navigate to the project

```bash
cd nlp_preprocessing
```

### Create a virtual environment

**Windows**

```bash
python -m venv venv
```

Activate it

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK resources

```bash
python setup.py
```

### Run the project

```bash
python main.py
```

---

# 📚 Concepts Covered

- Text Cleaning
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
- Bag of Words
- TF-IDF
- Word2Vec Basics

---

# 🎯 Learning Outcomes

After completing this project, you'll understand:

- How NLP preprocessing works
- Why text needs to be cleaned before analysis
- Difference between stemming and lemmatization
- How text is converted into numerical vectors
- How feature extraction techniques are used in Machine Learning

---

# 🚧 Future Enhancements

- Part-of-Speech (POS) Tagging
- Named Entity Recognition (NER)
- Sentiment Analysis
- Text Classification
- FastAPI Integration
- Streamlit Web Application
- BERT Embeddings
- Sentence Transformers

---

# 👨‍💻 Author

**Abid Dillep Jan**

Python | Machine Learning | Natural Language Processing | FastAPI

If you found this project helpful, feel free to ⭐ the repository!
