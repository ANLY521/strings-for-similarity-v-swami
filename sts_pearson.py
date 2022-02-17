from scipy.stats import pearsonr
import argparse
from util import parse_sts
from nltk.translate.nist_score import sentence_nist
from nltk import edit_distance, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    #looking at the sts_nist.py code

    nist_metric, bleu_metric, wer_metric, lcs_metric, ed_metric = [],[],[],[],[]

    for words in texts:
        t1, t2 = words
        # input tokenized text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        #Calculating NIST metric
        try:
            nist_metric = sentence_nist([t1_toks, ], t2_toks)
        except ZeroDivisionError:
            # print(f"\n\n\nno NIST, {i}")
            nist_metric = 0.0

        #Calculating bleu metrics
        try:
            bleu_metric = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_metric = 0.0
            bleu_metric.append(bleu_score)

        #Calculating WER metric
        wer_val = edit_distance(t1_toks, t2_toks)/(len(t1_toks)+len(t2_toks))
        wer_metric.append(wer_val)

        #Calculating LCS metric
        lcs_val = SequenceMatcher(None, t1, t2).ratio()
        lcs_metric.append(lcs_val)

        #Calculating ED metric
        ed_val = edit_distance(t1, t2)
        ed_metric.append(ed_val)


    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")

    for metric_name in score_types:
        score = 0.0
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)