import inspect
import numpy as np
from itertools import combinations
from helpers.classification.validation import cross_validate_classifier
from helpers.utils.validators import validate_x_y_numpy_array, validate_x_y_observation_count
from helpers.utils.logger import Logger


@validate_x_y_numpy_array
@validate_x_y_observation_count
def search_over_observations(X,
                             y,
                             model,
                             to_remove=(0, 1, 2),
                             to_score="mcc",
                             threshold=0.5,
                             metrics=("mcc", "acc", "sen", "spe"),
                             num_folds=10,
                             num_repetitions=20,
                             seed=42,
                             logger=None,
                             verbose=False):
    """
    Search for the best sub-set of observations

    This function searches for the sub-sample of observations that yield the best
    classification performance by iteratively sub-sampling combinations from the
    input feature matrix <X> according to <to_remove>. If for instance it is set
    to (0, 1, 2, ...), the function:
     - uses all observations
     - uses all combinations of S.shape[0] - 1 observations
     - uses all combinations of S.shape[0] - 2 observations
     - etc.

    It returns a list of dicts. In each dict, it holds the exact indices used to
    sub-sample the observations, and the performance measures that quantify how
    well the classification model performed on that particular sub-sample.

    Parameters
    ----------

    X : numpy array
        2D feature matrix (rows=observations, cols=features)

    y : numpy array
        1D labels array

    model : class that implements fit, and predict methods
        Initialized binary classification model

    to_remove : list or tuple, optional, default (0, 1, 2)
        Iterable with the number of observations to remove from sub-samples

    to_score : str, optional, default "mcc"
        Scoring function used to score each sub-sample

    threshold : float, optional, default 0.5
        Threshold for encoding the predicted probability as a class label

    metrics : tuple, optional, default ("mcc", "acc", "sen", "spe")
        Tuple with classification metrics to compute

    num_folds : int, optional, default 10
        Number of cross-validation folds

    num_repetitions : int, optional, default 20
        Number of cross-validation runs

    seed : int, optional, default 42
        Random generator seed

    logger : Logger, optional, default None
        Logger class

    verbose : bool, optional, default False
        Verbosity switch (informing about the performance on each sub-sample)

    Returns
    -------

    List of dicts with the performance on each observation sub-sample

    Raises
    ------

    TypeError
        Raised when X or y is not an instance of np.ndarray

    ValueError
        Raised when X and y have not the same number of rows (observations)
    """

    # Prepare the logger
    logger = logger if logger else Logger(inspect.currentframe().f_code.co_name)

    # Make sure the scoring function is in the metrics
    if to_score not in metrics:
        metrics = [m for m in metrics] + [to_score]

    # Prepare the list of results
    results = []

    # Cross-validate the classifier on the sub-samples of observations
    for i in to_remove:
        for selection in combinations(range(X.shape[0]), X.shape[0] - i):

            # Convert to the list (necessary for array slicing)
            selection = list(selection)

            # Select the subset of observations
            X_try = X[selection]
            y_try = y[selection]

            # Cross-validate the classifier
            cv = cross_validate_classifier(X_try,
                                           y_try,
                                           model,
                                           threshold=threshold,
                                           metrics=metrics,
                                           num_folds=num_folds,
                                           num_repetitions=num_repetitions,
                                           seed=seed,
                                           logger=logger)

            # Compute the scores (mean +- std of all cv metrics)
            scores = [(m, np.mean(cv.get(m)), np.std(cv.get(m))) for m in metrics]
            result = round(float(np.mean(cv.get(to_score))), 4)

            # Append to the results
            results.append({"score": result, "metrics": scores, "selection": selection, "removed": i})

            # Log the performance
            if verbose:
                logger.info("Skipping {} observation(s), {} = {}".format(i, to_score, result))

    return results
