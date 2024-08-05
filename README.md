# Irony-Detection-of-Arabic-Tweets-

This project investigates irony detection in Lebanese dialect comments on Twitter to enhance digital communication understanding.We will collect tweets from Lebanese users and manually annotate them to create a dataset. Using this dataset, we will train a neural network, aiming for an initial accuracy of over 78% in classifying tweets. The project aims to contribute to sentiment analysis by addressing challenges in multilingual contexts.
This project also represents my senior project for a BS in Computer science at Phoenicia University

# Installation

To set up this project, follow these steps:

1. Clone the repository:

   `bash`
   `git clone https://github.com/your-username/your-repository.git`

2. cd your-repository

3. `python -m venv venv`
   `.\venv\Scripts\activate`

4. `pip install -r requirements.txt`

5. Add the arabic embedding file

# Arabic Embedding file

The Arabic word embedding file mentioned in the config (wiki.ar.vec) is from the FastText pre-trained word vectors. You can download it from the official FastText website. Here's how you can get it:

1. Go to the FastText website's pre-trained word vectors page:
   https://fasttext.cc/docs/en/pretrained-vectors.html

2. Scroll down to the section "Pre-trained word vectors"

3. Look for the Arabic language (ar) in the list

4. Click on the "bin+text" link next to Arabic. This will download a file named cc.ar.300.bin.gz

5. Once downloaded, extract the .gz file. You should now have a file named cc.ar.300.bin

6. After extraction, you should have a file with the .vec extension, which you can rename to wiki.ar.vec

7. Place this file in the /data/fasttext/ directory of your project (or update the config file to point to wherever you place it)
