# Off-topic-essay

## Reference

```
@InProceedings{10.1007/978-3-031-79035-5_24, author="Silva, Joyce M. and Anchi{\^e}ta, Rafael T. and de Sousa, Rog{\'e}rio F. and Moura, Raimundo S.",
editor="Paes, Aline and Verri, Filipe A. N.",
title="Investigating Methods to Detect Off-Topic Essays",
booktitle="Intelligent Systems",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="346--357",
abstract="Automated Essay Scoring is one of the most important educational applications of natural language processing. It helps teachers with automatic assessments, providing a cheaper, faster, and more deterministic approach than humans when scoring essays. Nevertheless, off-topic essays pose challenges in this area, causing an automated grader to overestimate the score of an essay that does not adhere to a proposed topic. Thus, detecting off-topic essays is important for dealing with unrelated text responses to a given topic. This paper explored approaches based on handcrafted features to feed supervised machine-learning algorithms, tuning a BERT model, and prompt engineering with a large language model. We assessed these strategies in a public corpus of Portuguese essays, achieving the best result using a fine-tuned BERT model with a 75{\%} balanced accuracy. Furthermore, this strategy was able to identify low-quality essays.",
isbn="978-3-031-79035-5"
}
```

## Requirements

* Python (version 3.13.0 or later)
* ```
  pip install -r requirements.txt
  ```

## Usage

```
git clone https://github.com/liara-ifpi/Off-topic-essay.git
```

```
cd Off-topic-essay
```

```
├── essays/
│   ├── prompts.csv           # File containing the essay topics
│   ├── essay-br.csv          # Essay corpus
├── embedding/
│   └── embeddings            # Pre-trained embedding model (Word2Vec)
├── resultados_training.csv   # File generated with extracted features (training set)
├── resultados_testing.csv    # File generated with extracted features (testing set)
├── features.py               # Class for feature extraction
├── util.py                   # Utility functions
└── main.py                   # Main code
```

```
python main.py
```
