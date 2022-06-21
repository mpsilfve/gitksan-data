from packages.fairseq.utils import extract_hypotheses

def test_extract_hypotheses():
    predictions, confidences = extract_hypotheses('tests/results_seen_test.txt', 4)
    assert predictions[0] == "sise'en"
    assert confidences[0] == -0.17507074773311615
    assert len(predictions) == 316
    assert len(confidences) == 316
    assert predictions[1] == "sise'en"
    assert confidences[1] == -0.20760659873485565