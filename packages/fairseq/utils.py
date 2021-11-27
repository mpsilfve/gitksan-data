# TODO: test this
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