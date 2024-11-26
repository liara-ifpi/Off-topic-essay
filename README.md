# Off-topic-essay

## Reference

```
@inproceedings{sbracis,
author    = {Joyce M. Silva and Rafael T. Anchiêta and Rogério F. de Sousa and Raimundo S. Moura},
title  = {Investigating Methods to Detect Off-Topic Essays},
booktitle = {Anais da XIII Brazilian Conference on Intelligent Systems},
year    = {2024},                                                                                                                                                                           publisher = {SBC},
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
