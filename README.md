# SBU CSE 538 Natural Language Processing

## Neurologic Decoding with LTL constraints

This project is an extension of the original paper - [Neurologic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints, by Lu et al](https://aclanthology.org/2021.naacl-main.339.pdf).

The authors of this paper have shared their original implementation repository which can be found [here](https://github.com/GXimingLu/neurologic_decoding).

### Overview
NeuroLogic Decoding is a decode-level technique of constrained text generation which uses standard predicate logic (in the Conjunctive Normal Form) on tokens to add hard constraints on the generation of text instead of using the context alone, thus allowing more control over the output text. The underlying logic is still based on Beam Search, however it makes use of a set of Positive and Negative literals in a constained hypothesis. 

Our goal of this project was to extend the original NeuroLogic Decoding algorithm by introducing a subset of the Linear Temporal Logic (specifically, the "***until***" connective) constraints to the problem. Our motivation to do so, was to introduce a stricter level of sequential constraints wherein the occurrence of the next token is not dependent on the previous tokens, simply as context, but as a more robust order in the form of a directed acyclic graph.

For this reason, we use the Recipe generation datasest - [Recipe1M+](https://arxiv.org/pdf/1810.06553v2.pdf), which consists of more than 1 million recipes scraped from the web. It is particularly useful for us to work on recipes as they have a specific order in which the instructions are to be followed. For example, the ingredients are added to the cooking mix one-by-one in a specific order. This aligns with our problem environment. 

We use the ***until*** operator in the following manner for this implementation:

    (-A U B)  which means "not A until B"

Example - 

**(-stir U cheese)** : This means that the word stir should not occur **until** cheese has been predicted. 

### Implementation Details

Implementation of the ***until*** logic 
- The ***until*** logic is implemented in the Clause class of the ```lexical_constraints.py``` file. It stores a dictionary of all possible words that the current token depends on. Each time a constraint is met or is no longer effective, the associated tokens from the dictionary are popped.
- The ConstrainedHypothesis class also maintains a list of untils in the clauses that have occurred so far.

Data 
- Added ```recipes.constranit.txt``` and ```tiny.constraint.txt``` in the ```dataset/clean/constraint``` directory as input files for recipe generation.


Additional tasks 
- Created ```lm/evaluation.py``` which are used to evaluate the output text by comparing with expected tests and calculates the BLEU score, Coverage (Percentage of the expected ingredients occuring in the output), Order score (Metric to evaluate the expected sequence by obtaining the Levenshtein Edit distance between expected keywords.
- We also added the ```hypers.sh``` script to test out different hyperparameter combinations for optimal results.

Files we created/modified:

- ```convert_data.py```: This is a new script we wrote to parse data from the recipe1M+ dataset into a format that could be read by decode.sh in the Neurologic Decoding repo. It can produce a constraints file with no constraints, CNF constraints, or LTL constraints.
- ```lexical_constraints.py```: We merged the PositiveState and NegativeState class into one TrieManager class. We rewrote the Clause class so that logical handling is performed by this class rather than in the ConstrainedHypothesis class. This is where we perform handling for the until constraint. 
- ```lm/decode.py```: We made some small modifications to the main function so that it worked with the recipe_GPT2 model that we downloaded from hugging face.
- ```lm/decode.sh```: We modified this to use accept more arguments for hyperparameter tuning.
- ```lm/hypers.sh```: This new script runs decode.sh with different choices of hyperparameter values.
- ```lm/evaluation.py```: This new script performs evaluation on our generated text. It calculates BLEU score, Coverage score, and Order score.
- ```lm/topK.py```: We modified this script to enable/disable the grouping/selection phase of Neurologic Decoding
- ```unilm/utils_seq2seq.py```: We modified this script to handle parsing the JSON formatted constraints. We had to rework how constraints were formatted in order to encode the until operator.



### Running the project

You can follow the steps below to run this project,

1. First you must run the ```convert_data.py``` script in a folder containing test data from the recipe1M+ dataset. This will generate two text files, one containing the input and expected output for each sample on a new line and another containing the constraints for each sample in JSON format. You can copy these files to the ```dataset/``` folder in Neurologic Decoding. 
2. Make sure to download a pretrained recipe_GPT model in huggingface format. Please follow the steps given in the next two sections to download the recipe_GPT model as well as setting up the Python environment for this project. Change directory to ```lm/``` to run the below steps. 
3. **Test NeuroLogic Decoding** : ```decode.sh```. We use certain arguments in Python which are given below.
The first argument is the name of the constraints and text file you generated with ```convert_data.py```. The second argument is the path to the recipe_GPT model. The third argument is the name of the output file. The next three arguments are for hyperparameter values. A single output file is generated with the generated text for each data sample. You can now run lm/evaluation.py. The --split argument is the same as the first argument for decode.sh. The --generated_text argument is the name of the output file you generated. The --line_graph argument will generate a line graph using multiple output files as seen in our paper.

You can run the following script for the same

    bash decode.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_FILE_NAME BEAM_SIZE PRUNE_FACTOR BETA

  Here,

  - DEVICE_ID : No. of GPU devices connected
  - SPLIT : Dev / Test dataset to be used
  - MODEL_PATH : Path of the Recipe_GPT model
  - OUTPUT_FILE_NAME : Path to save the output generated text
  - BEAM_SIZE : Beam size (to obtain the top K hypothesis)
  - PRUNE_FACTOR : Fraction of candidates to keep based on scores (to eliminate candidates with low scores at each step)
  - BETA : Reward factor for in-progress constraint

  4. **Test Standard Beam Search** : Alternatively, you may also test the standard Beam. Search algorithm using the ```beam_search.sh``` script as a comparative analysis of the output generated. We use certain arguments in Python which are given below.
The first argument is the path to the recipe_GPT model. The second argument is the path of the input file.  The third argument is the name of the output file. There are additional arguments for optimizing the Beam Search such minimum and maximum target length, batch size for prediction and beam size.

You can run the following script for the same

    bash beam_search.sh DEVICE_ID SPLIT MODEL_PATH OUTPUT_FILE_NAME

  Here,

  - DEVICE_ID : No. of GPU devices connected
  - SPLIT : Dev / Test dataset to be used
  - MODEL_PATH : Path of the Recipe_GPT model
  - OUTPUT_FILE_NAME : Path to save the output generated text


### RecipeGPT model

We use a GPT model trained on [Recipe1M+](https://arxiv.org/pdf/1810.06553v2.pdf) dataset. Since our focus was mainly on the decoding logic than the model itself, we utilized a pretrained huggingface model for our implementation. The model can be extracted from huggingface and installed using the following set of commands:

```
git lfs clone https://huggingface.co/jky594176/recipe_GPT2
git lfs install
mv gpt2-finetuned-recipes-cooking gpt2
```

### Setting up the project dependencies

The project runs on Python 3.7 and has a list of python dependencies which are listed below and in the ```huggingface.txt``` file.

You can perform the full setup for the python libraries using the following command:

    pip install -r huggingface.txt

Software and package list:
- Python 3.7
- ipython==7.21.0
- pandas==1.1.5
- sklearn
- tensorboard==2.4.1
- tensorboardX==2.1
- transformers==3.0.2
- torch==1.7.0
- rouge==1.0.1
- nltk==3.7
- numpy==1.16.6

## Contributors
- [Andrew Burford](https://github.com/aburford)
- [Harshit Barot](https://github.com/finessebarot)
- [Abhishek Revadekar](https://github.com/Abhishek-612)
