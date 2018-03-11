import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    word_ix = 0

    for word_ix, (X, length) in test_set.get_all_Xlengths().items():
        # print ("IX: ", word_ix)
        # print ("X: ", X)
        # print ("length: ", length)
        # Get the word to be guessed
        word = test_set.wordlist[word_ix]
        # Test each model and keep the best score
        best_score = float("-INF")
        guessed_word = None
        probs = {}
        for w, model in models.items():
            try:
                if model is not None:
                    score = model.score(X, length)
                    probs[w] = score
                    if score > best_score:
                        guessed_word = w
                        best_score = score
            except Exception as ex:
                #print("Exception evaluating model for word {}: {}. Model: {}".format(w, ex, model))
                pass
        probabilities.append(probs)
        guesses.append(guessed_word)

    return probabilities, guesses
