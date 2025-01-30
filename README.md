# Off-topic-essay

## Reference

`@InProceedings{10.1007/978-3-031-79035-5_24, author="Silva, Joyce M. and Anchi{`

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
