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
        #https://www.nltk.org/_modules/nltk/translate/nist_score.html
        try:
            nist_val = sentence_nist([t1_toks, ], t2_toks)
        except ZeroDivisionError:
            # print(f"\n\n\nno NIST, {i}")
            nist_val = 0.0
        nist_metric.append(nist_val)

        #Calculating bleu metrics
        #https://www.askpython.com/python/bleu-score
        try:
            bleu_val = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=SmoothingFunction().method0)
        except ZeroDivisionError:
            bleu_val = 0.0
        bleu_metric.append(bleu_val)

        #Calculating WER metric
        #https://www.programcreek.com/python/?CodeExample=calculate+wer
        wer_val = edit_distance(t1_toks, t2_toks)/(len(t1_toks)+len(t2_toks))
        wer_metric.append(wer_val)

        #Calculating LCS metric
        #https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
        lcs_val = SequenceMatcher(None, t1, t2).ratio()
        lcs_metric.append(lcs_val)

        #Calculating ED metric
        #https://stackoverflow.com/questions/2460177/edit-distance-in-python
        ed_val = edit_distance(t1, t2)
        ed_metric.append(ed_val)

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")

    all_metrics = []
    all_metrics.append(nist_metric)
    all_metrics.append(bleu_metric)
    all_metrics.append(wer_metric)
    all_metrics.append(lcs_metric)
    all_metrics.append(ed_metric)

    for val in range(0,len(score_types)):
        score = pearsonr(all_metrics[val],labels)[0]
        print(f"{score_types[val]} correlation: {score:.03f}")


    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-train.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)