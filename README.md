# Neurologic Decoding with LTL

Neurologic Decoding - [Link](https://github.com/GXimingLu/neurologic_decoding)
RecipeGPT Model - [Link](https://drive.google.com/file/d/1Ij1uBxBxb4WrCE1UBiMYqFapp5pzP3v_/view?usp=sharing)

Files we modified:

- convert_data.py: This is a new script we wrote to parse data from the recipe1M+ dataset into a format that could be read by decode.sh in the Neurologic Decoding repo. It can produce a constraints file with no constraints, CNF constraints, or LTL constraints.
- lexical_constraints.py: We merged the PositiveState and NegativeState class into one TrieManager class. We rewrote the clause class so that logical handling is performed by this class rather than in the ConstrainedHypothesis class. This is where we perform handling for the until constraint. - decode.py: We made some small modifications to the main function so that it worked with the recipe_GPT2 model that we downloaded from hugging face.
- decode.sh: We modified this to use accept more arguments for hyperparameter tuning.
- hypers.sh: This new script runs decode.sh with different choices of hyperparameter values.
- lm/evaluation.py: This new script performs evaluation on our generated text. It calculates BLEU score, Coverage score, and Order score.
- lm/topK.py: We modified this script to enable/disable the grouping/selection phase of Neurologic Decoding
- unilm/utils_seq2seq.py: We modified this script to handle parsing the JSON formatted constraints. We had to rework how constraints were formatted in order to encode the until operator.

First you must run the convert_data.py script in a folder containing test data from the recipe1M+ data set. This will generate two text files, one containing the input and expected output for each sample on a new line and another containing the constraints for each sample in JSON format. You can copy these files to the dataset folder in Neurologic Decoding. Make sure to download a pretrained recipe_GPT model in huggingface format. You can then run decode.sh. The first argument is the name of the constraints and text file you generated with convert_data.py. The second argument is the path to the recipe_GPT model. The third argument is the name of the output file. The next three arguments are for hyperparameter values. A single output file is generated with the generated text for each data sample. You can now run lm/evaluation.py. The --split argument is the same as the first argument for decode.sh. The --generated_text argument is the name of the output file you generated. The --line_graph argument will generate a line graph using multiple output files as seen in our paper.

Required software:
Python 3.7
ipython==7.21.0
pandas==1.1.5
sklearn
tensorboard==2.4.1
tensorboardX==2.1
transformers==3.0.2
torch==1.7.0
rouge==1.0.1
nltk==3.7
numpy==1.16.6
