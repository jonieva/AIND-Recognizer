import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("inf")
        n_features = self.X.shape[1]

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                model.fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                N = self.X.shape[0]   # Number of data points

                # p = total number of parameters in the model:
                #   n_components * (n_components - 1) --> transition probabilities between states (the last row can be calculated
                #                             because the total probability must sum 1.0, that's the reason of the -1 term)
                #   n_components - 1 --> initial probabilities
                #   n_components * n_features * 2 --> means and variances for each feature
                p = (n_components ** 2) + (n_components * n_features * 2) - 1

                bic = -2. * logL + p * np.log(N)

                if bic < best_score:
                    # Keep the model with the lowest score
                    best_model = model
                    best_score = bic
            except Exception as ex:
                # Nothing to do. Just the model could not be trained with this number of components
                # print("Exception ocurred for word {} and {} components: {}".format(self.this_word, n_components, ex))
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("-inf")
        M = len(self.hwords)     # Total number of categories (words)
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_components)
                model.fit(self.X, self.lengths)
                # Likehood term
                logP = model.score(self.X, self.lengths)
                # Anti-likehood terms
                sum_ = 0.0
                for word in (word for word in self.hwords if word != self.this_word):
                    X, lengths = self.hwords[word]
                    sum_ += model.score(X, lengths)
                sum_ /= (M - 1)

                dic = logP - sum_

                if dic > best_score:
                    best_model = model
                    best_score = dic
            except Exception as ex:
                # Nothing to do. Just the model could not be trained with this number of components
                #print("Exception ocurred for word {} and {} components: {}".format(self.this_word, num_components, ex))
                pass

        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Number of sequences for the current word
        num_sequences = len(self.lengths)

        # Number of folds by default used by KFold (it could be parametrized in the class)
        # By default we will use the same number splits than sequences for this word, but we will limit it
        K = min(num_sequences, 20)

        if K == 0:
            # No data points
            return None
        best_global_score = float("-inf")
        best_global_model = None

        if K == 1:
            # Cross validation cannot be applied. Just return a trained model with the single sequence available
            for num_components in range(self.min_n_components, self.max_n_components + 1):
                try:
                    model = self.base_model(num_components)
                    model.fit(self.X, self.lengths)
                    score = model.score(self.X, self.lengths)
                    if score > best_global_score:
                        # Keep the model that worked best for this number of components
                        best_global_model = model
                        best_global_score = score
                except Exception as ex:
                    # Nothing to do. Just the model could not be trained with this number of components
                    # print("Exception occurred for word {} and {} components: {}".format(self.this_word, num_components,
                    #                                                                    ex))
                    pass
            return best_global_model

        # Apply cross validation
        kf = KFold(n_splits=K, random_state=self.random_state, shuffle=True)
        # print("Total sequences for word {}: {}. Number of splits: {}".format(self.this_word, num_sequences, K))

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                score = 0.0
                nfolds = 0
                best_model = None
                best_score = float("-inf")
                # print ("Testing {} components...".format(num_components))
                for train_index, test_index in kf.split(self.sequences):
                    # Select indexes to make this training
                    # print ("Word: {}".format(self.this_word))
                    # print ("train_index: ", train_index)
                    # print ("test_index: ", test_index)
                    model = self.base_model(num_components)

                    # Get the sequences based on the train index
                    # Unfortunately we have to iterate because self.sequences is a list of lists with variable size
                    seqs = []
                    lengths = []
                    for ix in train_index:
                        for seq in self.sequences[ix]:
                            seqs.append(np.array(seq))
                        lengths.append(self.lengths[ix])
                    x = np.array(seqs)
                    # Train with train set
                    model.fit(x, lengths)

                    # Test with test set
                    seqs = []
                    lengths = []
                    for ix in test_index:
                        for seq in self.sequences[ix]:
                            seqs.append(np.array(seq))
                        lengths.append(self.lengths[ix])
                    x = np.array(seqs)
                    likehood = model.score(x, lengths)

                    # Keep the best model for this number of components so far
                    if likehood > best_score:
                        best_model = model
                        best_score = likehood

                    # Add the score (the final result will be the average)
                    score += likehood
                    nfolds += 1
                # The final components score will be the average score for all the folds
                score /= nfolds
                if score > best_global_score:
                    # Keep the model that worked best for this number of components
                    best_global_model = best_model
                    best_global_score = score
            except Exception as ex:
                # Nothing to do. Just the model could not be trained with this number of components
                #print("Exception occurred for word {} and {} components: {}".format(self.this_word, num_components, ex))
                pass

        return best_global_model