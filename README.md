Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence

NIST

The NIST metric is derived from the BLEU metric, where the n-gram is taken into account. This in return gives more credability to a system if the n-gram match is "difficult" and less credability to a n-gram which is "easy." A sample usage is when the word "and that" matches correctly would be weighed higher whereas "upon therefore" would be matched lower even if the meaning is similar.

BLEU

The BLEU metric  averages the precision a n-gram from n=1 to n=4, and applies a len penalty if
the generated sentence is shorter than the best matching. A sample usage is when you have two sentences "the the the the the the" and "the cat is on the mat." It would compare the n-grams from 1 to 4 and determine the length penality and how true the comparison is, all while being averaged.


WER

The WER metric is derived from the Levenshtein distance, and is primarly used to compare different structures as well as model improvments within a structure. It is calculated with the formula Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken. A sample usage would be its use in calculating Automatic Speech Recognition (ASR), where the WER would be used as a metric to see how well a system or software did text transcription.

LCS

The LCS metric is as its name suggest, find the longest common substring and the more similar the two strings are. It is done by finding all substrings of one string, X, of length n. A sample usage would be using it as a base metric to compare various other/new string similary metrics.

Edit Dist

The Edit Distance metric is a process of mathetically understanding how different two strings are to one another by counting the minimum number of operations required to transform one string into the other. A sample usage would be in computational biology, where you would want to figure out if shorter strings match in complex and larger biological strings.

References:
1. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.536.5232&rep=rep1&type=pdf
2. https://en.wikipedia.org/wiki/Edit_distance


Example on how I am running the code on the terminal:
(base) swamivenkat@Swamis-MacBook-Pro strings-for-similarity-v-swami % python sts_pearson.py

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | 0.493 | 0.593 | 0.464
BLEU | 0.371 | 0.433 | 0.353
WER | -0.362 | -0.452| -0.364
LCS | 0.463 | 0.468| 0.504
Edit Dist | 0.033 | -0.175| -0.039

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).


## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.
