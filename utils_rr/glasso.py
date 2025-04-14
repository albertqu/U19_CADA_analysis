from skglm import GeneralizedLinearEstimator
from skglm.datafits import LogisticGroup
from skglm.penalties import WeightedGroupL2
from skglm.solvers import GroupProxNewton
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import skglm
import adelie as ad
import numpy as np
import pandas as pd
import scipy
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin

class GroupLassoLogisticCV(BaseEstimator, ClassifierMixin):
    """ 
    This class implements a sklearn compatible class for group lasso while using cross validation
    to select the best lambda. This is built on top of the skglm package.
    """
    def __init__(self, group=False, cv=5, seed=230, 
                 grp_ptr=None):
        self.group=group
        if grp_ptr is not None:
            self.grp_ptr=grp_ptr.astype(np.int32)
        else:
            self.grp_ptr = None
        self.cv = cv
        self.kf = KFold(cv, shuffle=True, random_state=seed)
        self.seed=seed

    def fit(self, X, y):
        # penalty = skglm.penalties.L1(lam)
        # datafit = skglm.datafits.Logistic()
        # solver=skglm.solvers.AndersonCD()
        # add group labels
        if self.group:
            if self.grp_ptr:
                grp_ptr = self.grp_ptr
            elif isinstance(X, pd.DataFrame):
                grp_ptr = group_features(X.columns, 
                                         include_end=True).astype(np.int32)
            else:
                raise ValueError("Please provide group_vector or X as a pd.DataFrame")
        else:
            grp_ptr = np.arange(X.shape[1]+1, dtype=np.int32)

        grp_indices = np.arange(X.shape[1], dtype=np.int32)
        weights = np.ones(len(grp_ptr)-1)
        self.grp_ptr = grp_ptr
        self.grp_indices = grp_indices
        self.weights=weights

        # Check input X and y
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.squeeze()
        
        if len(y.shape) > 1:
            y = y.squeeze()

        lam_max = np.linalg.norm(X.T @ y, ord=np.inf)
        lams = lam_max * np.geomspace(1, 1e-7)
        datafit = LogisticGroup(self.grp_ptr, self.grp_indices)
        solver = GroupProxNewton(fit_intercept=True,verbose=0)
        train_losses = np.zeros(len(lams))
        test_losses = np.zeros(len(lams))
        k = self.kf.get_n_splits()
        for train_index, test_index in self.kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for i in range(len(lams)):
                lam = lams[i]
                penalty = WeightedGroupL2(lam, self.weights, self.grp_ptr, self.grp_indices)
                clf = GeneralizedLinearEstimator(datafit, penalty, solver)
                clf.fit(X_train, y_train)
                train_losses[i] += balanced_accuracy_score(y_train, clf.predict(X_train)) / k
                test_losses[i] +=  balanced_accuracy_score(y_test, clf.predict(X_test)) / k
        best_lam = lams[np.argmax(test_losses)]
        self.best_lam = best_lam
        penalty = WeightedGroupL2(best_lam, self.weights, self.grp_ptr, self.grp_indices)
        clf = GeneralizedLinearEstimator(datafit, penalty, solver)
        clf.fit(X, y)
        self.best_clf = clf
        return self
    
    def predict(self, X):
        # Check input X 
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.best_clf.predict(X)
    
    def predict_proba(self, X):
        # Check input X 
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.best_clf.predict_proba(X)
    
    def score(self, X, y):
        return balanced_accuracy_score(y, self.predict(X))
    
class GroupElasticLogisticCV(BaseEstimator, ClassifierMixin):
    """ 
    This class implements a sklearn compatible class for 
    group lasso logistic regression model with cross-validation.

    To use group lasso function, you need to either:
    1. provide X as a pd.DataFrame with format `[group]__[feature]`
    2. provide group_vector as a list of indices for starting index of each group

    diagnostics:
    ```
    # diagnostic plot for lasso path
    # cv_res.plot_loss()
    # dg=ad.diagnostic.diagnostic(state=state)
    # dg.plot_coefficients()
    # dg.plot_devs()
    # dg.plot_benchmark()
    ```
    
    Notes
    -----
    This class solves the following optimization function:
    .. math::
        \min_{\beta} \sum_{i=1}^n \log(1 + \exp(-y_i X_i^T \beta)) + \lambda \sum_g w_g \Bigg( \alpha ||\beta_g||_2 + (1-\alpha) \frac{||\beta_g||_2^2}{2} \Bigg)
    
    where :math:`\beta_g` is the coefficient vector for group g, and :math:`w_g` is the weight for group g.

    This class builds on top of `Adelie package <https://jamesyang007.github.io/adelie/index.html>`

    .. [1] Yang and Hastie (2024). "A Fast and Scalable Pathwise-Solver for Group Lasso and Elastic Net Penalized Regression via Block-Coordinate Descent""
    """

    def __init__(self, alpha=1.0, balanced=True, fit_intercept=True,
                 group=False, group_vector=None, seed=230):
        self.alpha = alpha
        self.balanced = balanced
        self.fit_intercept = fit_intercept
        self.group = group
        self.group_vector = group_vector
        self.seed = seed
        # self.state = None
        # self.cv_res_ = None
        # self.coef_ = None
        # self.intercept_ = 0
        # self.best_lam_ = None
    
    def fit(self, X, y):
        if self.group:
            if self.group_vector:
                group_vector = self.group_vector
            elif isinstance(X, pd.DataFrame):
                group_vector = group_features(X.columns, include_end=False)
            else:
                raise ValueError("Please provide group_vector or X as a pd.DataFrame")
        else:
            group_vector = None
        
        # Check input X and y
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not np.isfortran(X):
            X = np.asfortranarray(X)
        
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Compute weights
        if self.balanced:
            uniq_classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=uniq_classes, y=y)
            sample_weights = np.empty(len(y), dtype=np.float64)
            for i in range(len(uniq_classes)):
                sample_weights[y == uniq_classes[i]] = class_weights[i]
            sample_weights = sample_weights / np.sum(sample_weights)
        else:
            sample_weights = None

        # fit model
        glm_class = ad.glm.binomial(y, 
                                weights=sample_weights,
                                dtype=np.float64)
        cv_res = ad.cv_grpnet(
            X=X,
            glm=glm_class,
            groups=group_vector,
            min_ratio=1e-3,
            seed=self.seed,
            intercept=self.fit_intercept,
            progress_bar=False,
            alpha=self.alpha,
        )

        state = cv_res.fit(
            X=X,
            glm=glm_class,
            groups=group_vector,
            min_ratio=1e-3,
            intercept=self.fit_intercept,
            progress_bar=False,
            alpha=self.alpha
        )

        self.state = state
        self.cv_res_ = cv_res
        self.coef_ = state.betas[-1]
        self.intercept_ = state.intercepts[-1]
        self.best_lam_ = state.lmdas[-1]
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not np.isfortran(X):
            X = np.asfortranarray(X)
        linear_pred = ad.diagnostic.predict(X, self.coef_, np.array([self.intercept_]))
        proba = scipy.special.expit(linear_pred)
        proba = np.stack((1 - proba, proba), axis=-1).squeeze()
        return proba
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1).squeeze()
    
    def score(self, X, y):
        return balanced_accuracy_score(y, self.predict(X))
    
#############################################
############ Helper Functions ###############
#############################################

def group_features(features, include_end=False):
    """ This function takes a list of features and group them based on their prefix, 
    and return a list of integer labels for each group (defined as the starting index of each group), 
    note that each group label is incremented by size of previous group.
    Assume that features are pre-sorted
    Example:
    >>> group_features(['offer_rank', 'ACC', 'commit', 'reward'])
    [0, 1, 2, 3]
    >>> group_features(['offer_rank__b1', 'offer_rank__b2', 'ACC__b1', 'commit__b1'])
    [0, 0, 2, 3]   
    """
    def get_group(feat):
        if '__' in feat:    
            return feat.split('__')[0]
        else:
            return feat
    group_inds = [0]
    curr_group = get_group(features[0])
    for i in range(1, len(features)):
        g = get_group(features[i])
        if g != curr_group:
            group_inds.append(i)
            curr_group = g
    if include_end:
        group_inds.append(len(features))
    return np.array(group_inds)


#################################################
################ Experiment #####################
#################################################
def experiment_adelie_models(X_train, y_train, X_test, y_test, X_cols,
                             alpha=1.0, balanced=True, group=False, seed=230):
    """ 
    Goal: fix sampling imbalance issue, experiment with alpha=1.0, 0.5, 0 as a solution;
    eventually compare with group lasso
    benchmark answer with f1, bac, acc
    """

    # add group labels
    if group:
        groups = group_features(X_cols, include_end=False)
    else:
        groups = None

    if not np.isfortran(X_train):
        X_train = np.asfortranarray(X_train)
        X_test = np.asfortranarray(X_test)

    # Compute weights
    if balanced:
        uniq_classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=uniq_classes, y=y_train)
        sample_weights = np.empty(len(y_train), dtype=np.float64)
        for i in range(len(uniq_classes)):
            sample_weights[y_train == uniq_classes[i]] = class_weights[i]
        sample_weights = sample_weights / np.sum(sample_weights)
    else:
        sample_weights = None

    # fit model
    glm_class = ad.glm.binomial(y_train, 
                            weights=sample_weights,
                            dtype=np.float64)
    cv_res = ad.cv_grpnet(
        X=X_train,
        glm=glm_class,
        groups=groups,
        min_ratio=1e-3,
        seed=seed,
        intercept=True,
        progress_bar=False,
        alpha=alpha,
    )
    state = cv_res.fit(
        X=X_train,
        glm=glm_class,
        groups=groups,
        min_ratio=1e-3,
        intercept=True,
        progress_bar=False,
        alpha=alpha
    )
    coef = state.betas[-1]
    intercept = np.array([state.intercepts[-1]])
    lam = state.lmdas[-1]
    print('lambda', lam)
    # predict_proba 
    linear_pred = ad.diagnostic.predict(X_test, coef, intercept)
    proba = scipy.special.expit(linear_pred)
    proba = np.stack((1 - proba, proba), axis=-1).squeeze()
    print(proba.shape)
    y_pred = np.argmax(proba, axis=-1).squeeze()

    print(balanced_accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred))

    return coef