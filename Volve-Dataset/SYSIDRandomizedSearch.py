from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if

class SYSIDRandomizedSearchCV(RandomizedSearchCV):

    #@available_if(_estimator_has("predict"))
    def predict(self, X, y):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X, y)

    #@available_if(_estimator_has("transform"))
    def transform(self, X, y):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.transform(X, y)

