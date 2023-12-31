# Sentiment Analysis
Implementation of a BiRNN and BiGRU for sentiment analysis done with three word embeddings: GloVe, FastText and Word2Vec.

---

### Download the models before using the notebook
* GloVe (download done in notebook)
* FastText
    * Install FastText : [FastText installation](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module)
    * Download `cc.en.300.bin.gz` [Download with command line](https://fasttext.cc/docs/en/crawl-vectors.html#download-directly-with-command-line-or-from-python:~:text=Download%20directly%20with%20command%20line%20or%20from%20python)
    * Adapt the dimension to 100 : [Adapt dimension](https://fasttext.cc/docs/en/crawl-vectors.html#download-directly-with-command-line-or-from-python:~:text=10%0AQuery%20word%3F-,Adapt%20the%20dimension,-The%20pre%2Dtrained)
    * Convert the BIN file to a VEC file (you can use [this code](https://stackoverflow.com/questions/58337469/how-to-save-fasttext-model-in-vec-format/58342618#58342618:~:text=15-,To%20obtain%20VEC%20file,-%2C%20containing%20merely%20all))
* Word2Vec
  * Download model `40` on [vectprs.nlpl.eu](http://vectors.nlpl.eu/repository/) : [Download](http://vectors.nlpl.eu/repository/20/40.zip)

Then put `cc.en.100.vec` and the folder `40` in the same folder as the notebook

### Installation
We use the same conda environment as described in d2l.ai : [Installing the Deep Learning Framework and the d2l Package](https://www.d2l.ai/chapter_installation/index.html#:~:text=conda%20activate%20d2l-,Installing%20the%20Deep%20Learning%20Framework%20and%20the%20d2l%20Package,-%C2%B6),

also install `gensim` and `scipy`
+ ``` conda install -c conda-forge gensim ```
+ ``` conda install -c conda-forge scipy ```
