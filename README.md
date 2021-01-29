# SeqGan TF2 Implementation For Custom Datasets

## Installation and Usage

Requirements are:
- TensorFlow == 2.4
- Numpy == 1.19.5
- Pandas == 1.1.5
- NLTK == 3.5

Install requirements.
```
pip install -r requirements.txt
```
### Generate Dataset
Generate dataset for SeqGan from IMDB Dataset.
`preprocessing.py` clears, tokenizes, split and saves dataset. 

It will generate 3 file: 
1. `dataset/positives.txt`
2. `dataset/negatives.txt`
3. `pretrained_models/tokenizer.pickle` is a tokenizer file. Saved because decode the model generate sentences back to words.
```
python utils/preprocessing.py
```

### Train
Start to train SeqGan on preprocessed data.
```
python train.py
```

### Test
`test.py` load the trained generator model then generate sentences. That number is equal to batch size(default is 64). 

Then decode with `pretrained_models/tokenizer.pickle` that is the tokenizer we use tokenize.

Then prints all sentences line by line.
```
python test.py
```

