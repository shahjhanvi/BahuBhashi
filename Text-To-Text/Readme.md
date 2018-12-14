## Dependencies
- [European Parliament Proceedings Parallel Corpus](http://www.statmt.org/europarl/v7/de-en.tgz)
  _(It contains two files. One is the dataset of English Sentences and other is the dataset of German sentences)_
- [TensorFlow v1.9.0](https://www.tensorflow.org/install/#download-and-setup)
- [Python v3.6.2](https://www.python.org/downloads/release/python-370/)
- [Natural Language ToolKit](https://www.nltk.org/)

## Imports Required 
- nmt_data_utils.py
- nmt_model_utils.py
- NMT_Model.py

## Architecture
#### Sequence2Sequence Model
- Bidirectional RNN (LSTM or GRU)
- Bahdanau Attention
- Adam Optimizer
- Beam Search or Greedy Decoding

## Results
![alt text](Images/Results.JPG)

## Acknowledgements
- [Thomasschmied - Neural_Machine_Translation_Tensorflow](https://github.com/thomasschmied/Neural_Machine_Translation_Tensorflow)
- [Overcoming the Language Barrier with Speech Translation Technology](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.1019&rep=rep1&type=pdf)
- [Speech-to-Speech Translation: A Review](https://pdfs.semanticscholar.org/0fa1/911622a6c0a3dd43fefbdf2695ebdb7e10fa.pdf)
- [Speechalator: two-way speech-to-speech translation on a consumer PDA](https://www.cs.cmu.edu/~awb/papers/eurospeech2003/speechalator.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE) file for details

