import numpy as np
from sklearn.metrics import pairwise_distances

def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])

def extract_hypotheses(fs_prediction_fname, num_hypotheses):
    predictions = []
    confidences = []
    with open(fs_prediction_fname) as prediction_f:
        line = prediction_f.readline()
        while line != '':
            ##### processing a block
            prediction_f.readline() # skip timing
            for i in range(num_hypotheses):
                if i == 0: # NOTE: can adjust this later to get more predictions
                    prediction_confidence_line = prediction_f.readline()
                    prediction_split = prediction_confidence_line.split('\t')
                    # print(prediction_split)
                    prediction = ''.join(prediction_split[2].split(' ')).strip()
                    confidence = float(prediction_split[1])
                    predictions.append(prediction)
                    confidences.append(confidence)
                    for _ in range(2): 
                        prediction_f.readline() # skip D and P line
                else: 
                    for _ in range(3): # skip other hypotheses
                        prediction_f.readline()
            line = prediction_f.readline()
            ##### processing a block
    return predictions, confidences

def extract_hypotheses_mbr(fs_prediction_fname, num_hypotheses):
    """Extract best hypotheses from results file produced by sampling from the Transformer.
    Note that these results are produced from 

    Args:
        fs_prediction_fname (str): Name of the prediction file.
        num_hypotheses ([int]): Number of hypotheses that were sampled.
        scoring_function ((str, str) => float): utility function.

    Returns:
        [type]: [description]
    """
    all_predictions = []
    all_confidences = []
    with open(fs_prediction_fname) as prediction_f:
        line = prediction_f.readline()
        while line != '':
            ##### processing a block
            prediction_f.readline() # skip timing line. 
            source_predictions = []
            source_confidences = []
            for _ in range(num_hypotheses):
                prediction_confidence_line = prediction_f.readline()
                prediction_split = prediction_confidence_line.split('\t')
                hypothesis = ''.join(prediction_split[2].split(' ')).strip()
                hyp_confidence = float(prediction_split[1])
                source_predictions.append(hypothesis)
                source_confidences.append(hyp_confidence)

                prediction_f.readline() # skip detokenized line
                prediction_f.readline() # skip token logits line
            safest_prediction, safest_confidence = obtain_safest_hypothesis(source_predictions, source_confidences)
            all_predictions.append(safest_prediction)
            all_confidences.append(safest_confidence)
            line = prediction_f.readline()
            ##### processing a block
    return all_predictions, all_confidences

# TODO; test this
def obtain_safest_hypothesis(predictions, confidences):
    """Obtains the safest hypothesis from samples produced by Transformer.

    Args:
        predictions ([str]): Predictions made from sampling a transformer.
        confidences ([float]): Prediction logits.
    """
    predictions_confidences_sorted = [(p, c) for p, c in sorted(zip(confidences, predictions), reverse=True)]
    confidences_s, predictions_s = zip(*predictions_confidences_sorted)
    # pairwise_l_distances = pairwise_distances(predictions_s, predictions_s, metric=distance)
    pairwise_distances = np.zeros((len(predictions_s), len(predictions_s)))
    for i in range(len(predictions_s)):
        for j in range(len(predictions_s)):
            pairwise_distances[i][j] = distance(predictions_s[i], predictions_s[j])
    # print(pairwise_l_distances)
    sum_distances = pairwise_distances.sum(axis=0)  
    safest_i = sum_distances.argmin()
    final_prediction = predictions_s[safest_i]
    confidence = confidences_s[safest_i]
    return final_prediction, confidence

