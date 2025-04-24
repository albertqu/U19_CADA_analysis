import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from utils_rr.configs import RAND_STATE
import joblib
import os
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


""" 
This Function uses PCA and SVC to find trials where mice's movements are not indicative 
of future decision, prior to lateralized action
"""

def tracks_fit_pca_idR(pivot_tracks, restaurant_id, 
                       variance_threshold=0.85, n_comp=None,
                       plot_var=False):
    """
    Fit PCA to data from a specific restaurant and return transformed data with principal components
    that explain at least variance_threshold of the variance.
    This is because due to experimental setup imperfection, animal behavior tracks are restaurant dependent

    Parameters:
    ----------
    pivot_tracks : pd.DataFrame
        DataFrame containing the pivot_tracks data
    restaurant_id : int
        ID of the restaurant to process
    variance_threshold : float, default=0.8
        Threshold for explained variance

    Returns:
    -------
    pca_df : pd.DataFrame
        DataFrame with transformed data
    """
    aux_cols = ['track_id', 'slp_accept', 'slp_decision', 'restaurant']
    track_cols = [c for c in pivot_tracks.columns if c not in aux_cols]
    
    # Select data for this restaurant
    r_selection = pivot_tracks['restaurant'] == restaurant_id
    Xdf = pivot_tracks.loc[r_selection, track_cols]
    X = Xdf.values
    
    print(f"Restaurant {restaurant_id} - Data shape: {X.shape}")
    
    # Initial PCA to determine number of components needed
    n_init = min(X.shape[1] // 2, X.shape[0] - 1)  # Ensure n_components doesn't exceed samples-1
    pca_init = PCA(n_components=n_init)
    pca_init.fit(X)
    vratio = np.cumsum(pca_init.explained_variance_ratio_)
    
    if plot_var:
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(1, len(vratio)+1), vratio, 'k-')
        plt.axhline(0.9, color='r', linestyle='--')
        sns.despine()
    
    # Find minimum components needed to explain variance_threshold
    min_comp = np.where(vratio >= variance_threshold)[0]
    if len(min_comp) == 0:
        min_comp = n_init
        print(f"Warning: Could not reach {variance_threshold*100}% explained variance with {n_init} components.")
        print(f"Maximum explained variance: {vratio[-1]:.4f}")
    else:
        min_comp = min_comp[0] + 1
        print(f"Components needed for {variance_threshold*100}% variance: {min_comp}")
        print(f"Explained variance ratios: {vratio[:min_comp]}")
    print('explained:', vratio[:min_comp])
    
    # Take at least 4 components if available, but not more than needed for variance threshold
    if n_comp is None:
        n_components = min_comp #max(min(4, X.shape[1]), min_comp)
    else:
        n_components = n_comp
    print(f"Using {n_components} principal components")
    
    # Fit final PCA and transform data
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    # Add decision columns and index
    pca_df[aux_cols] = pivot_tracks.loc[Xdf.index, aux_cols].values
    # pca_df['original_index'] = Xdf.index
    # Return dataframe and PCA model
    return pca_df, pca


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from utils_rr.glasso import GroupElasticLogisticCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, make_scorer
import xgboost as xgb
import os
import joblib
from time import time
import warnings
warnings.filterwarnings('ignore')

def model_combined_grid_search(X, y, model_folder=None, restaurant_id=None):
    """
    Run grid search for all models simultaneously and compare results
    
    Parameters:
    -----------
    X : array-like
        Training features
    y : array-like
        Target values
    model_folder : str, optional
        Folder to save models and results
    restaurant_id : int, optional
        ID of the restaurant (for saving models)
        
    Returns:
    --------
    results_df : DataFrame
        DataFrame with grid search results for all models
    best_model : estimator
        Best model from grid search
    best_model_name : str
        Name of the best model
    """
    # Define models to test
    models = {
        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=42,
            eval_metric='logloss'
        ),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        ),
        'GroupLassoAdelie': GroupElasticLogisticCV(balanced=True, group=False, seed=RAND_STATE)
    }
    
    # Define parameter grids for each model
    param_grids = {
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'RandomForest': {
            'n_estimators': [100, 300, 600, 1000],
            'max_depth': [3, 5, 8], 
            'min_samples_leaf': [20, 0.005, 0.01],
        },
        'GroupLassoAdelie': {
            'alpha': [0.5, 1.0]
        }
    }
    
    # Define scoring metrics
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score)
    }
    
    # Run grid search for each model
    results = []
    best_scores = {}
    best_models = {}
    
    for model_name, model in models.items():
        print(f"\nRunning grid search for {model_name}...")
        start_time = time()
        
        # Set up grid search
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            n_jobs=-1,
            scoring=scoring,
            refit='balanced_accuracy',
            return_train_score=True,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Record time taken
        elapsed_time = time() - start_time
        print(f"{model_name} grid search completed in {elapsed_time:.2f} seconds")
        
        # Store best model
        best_models[model_name] = grid_search.best_estimator_
        
        # Extract best score
        best_score = grid_search.best_score_
        best_scores[model_name] = best_score
        
        # Print best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best balanced accuracy: {best_score:.4f}")
        
        # Save model if requested
        if model_folder and restaurant_id is not None:
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_path = os.path.join(model_folder, f"trackProp_grid_{restaurant_id}_{model_name}.pkl")
            joblib.dump(grid_search.best_estimator_, model_path)
        
        relevant_cols = [c for c in grid_search.cv_results_ if c.startswith('mean_test_') or c.startswith('param_')]
        result = pd.DataFrame({c: grid_search.cv_results_[c] for c in relevant_cols})
        if model_folder is not None:
            result_path = os.path.join(model_folder, f"trackProp_grid_{restaurant_id}_{model_name}_results.csv")
            result.to_csv(result_path, index=False)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model overall
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]
    
    print(f"\nBest model overall: {best_model_name}")
    print(f"Best balanced accuracy: {best_scores[best_model_name]:.4f}")
    
    return results_df, best_model, best_model_name

def tracks_model_select_idR(pca_df, restaurant_id, model_folder=None):
    """
    Train all models for a specific restaurant's PCA data using combined grid search
    
    Parameters:
    -----------
    pca_df : DataFrame
        DataFrame with PCA components and decision data
    restaurant_id : int
        ID of the restaurant to process
    model_folder : str, optional
        Folder to save models and results
        
    Returns:
    --------
    best_model : estimator
        Best model from grid search
    results_df : DataFrame
        DataFrame with grid search results
    filtered_df : DataFrame
        Data used for training
    """
    # Filter out rows with 'quit' decisions
    filtered_df = pca_df[pca_df['slp_decision'].isin(['accept', 'reject'])].copy()
    
    if filtered_df.empty:
        print(f"No valid decisions for restaurant {restaurant_id}")
        return None, None, filtered_df
    
    # Get PC columns
    pc_cols = [col for col in filtered_df.columns if col.startswith('PC')]
    
    # Prepare X and y
    X = filtered_df[pc_cols].values
    
    # Convert decision to binary format
    if filtered_df['slp_decision'].dtype == 'object':
        y = (filtered_df['slp_decision'] == 'accept').astype(int).values
    else:
        y = filtered_df['slp_decision'].values
    
    print(f"\n{'-'*50}")
    print(f"Running combined grid search for Restaurant {restaurant_id}")
    print(f"{'-'*50}")
    
    # Run combined grid search
    results_df, best_model, best_model_name = model_combined_grid_search(
        X, y, model_folder, restaurant_id
    )
    
    print(f"\nBest model for Restaurant {restaurant_id}: {best_model_name}")
    
    return best_model, results_df, filtered_df

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
def tracks_train_qda_idR(pca_df):
    filtered_df = pca_df[pca_df['slp_decision'].isin(['accept', 'reject'])].copy()

    # Get PC columns
    pc_cols = [col for col in filtered_df.columns if col.startswith('PC')]

    # Prepare X and y
    # X = filtered_df[pc_cols].drop(columns=['PC1']).values
    X = filtered_df[pc_cols].values

    # Convert decision to binary format
    if filtered_df['slp_decision'].dtype == 'object':
        y = (filtered_df['slp_decision'] == 'accept').astype(int).values
    else:
        y = filtered_df['slp_decision'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RAND_STATE)

    clf = Pipeline([('scaler', StandardScaler()), ('qda', QuadraticDiscriminantAnalysis(reg_param=0.0))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('test', balanced_accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred))

    clf.fit(X, y)
    return clf, filtered_df

def tracks_train_svc_idR(pca_df, restaurant_id, model_folder=None):
    """Train SVC classifier with grid search for a specific restaurant's PCA data"""
    # Filter out rows with 'quit' decisions as they might not be relevant
    filtered_df = pca_df[pca_df['slp_decision'].isin(['accept', 'reject'])].copy()
    
    if filtered_df.empty:
        print(f"No valid decisions for restaurant {restaurant_id}")
        return None, filtered_df
    
    # Get PC columns
    pc_cols = [col for col in filtered_df.columns if col.startswith('PC')]
    
    # Prepare X and y
    X = filtered_df[pc_cols].values
    y = filtered_df['slp_decision'].values
    
    # Define SVC parameters for grid search
    svc_params = {
        'kernel': ('linear', 'rbf'), 
        'C': [0.1, 1, 10, 50, 100, 500]
    }
    
    # Grid search settings
    gs_params = {
        'cv': 5, 
        'n_jobs': os.cpu_count()-1,
        'scoring': ['balanced_accuracy', 'accuracy', 'f1'],
        'refit': 'balanced_accuracy'
    }
    
    # Create and fit grid search
    grid_search = GridSearchCV(SVC(class_weight='balanced', probability=True), svc_params, **gs_params)
    grid_search.fit(X, y)
    
    # Save results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    if model_folder is not None:
        cv_results.to_csv(os.path.join(model_folder, f'restaurant_{restaurant_id}_svc_cv_results.csv'))
        joblib.dump(grid_search.best_estimator_, os.path.join(model_folder, f'restaurant_{restaurant_id}_svc_best.pkl'))
        
    # Print best results
    print(f"Restaurant {restaurant_id} - Best parameters: {grid_search.best_params_}")
    print(f"Restaurant {restaurant_id} - Best score: {grid_search.best_score_:.4f}")
    
    # Return best model and data used
    return grid_search.best_estimator_, filtered_df

def tracks_calc_decision_uncertainty(model, pca_df):
    """Get prediction probabilities and identify points with uncertain predictions"""
    if model is None:
        return None
    
    # Get PC columns
    pc_cols = [col for col in pca_df.columns if col.startswith('PC')]
    
    # Get probabilities
    if len(pc_cols) > 0:
        probs = model.predict_proba(pca_df[pc_cols].values)
        
        # Add probabilities to dataframe
        if probs.shape[1] >= 2:  # Binary classification
            pca_df['prob_accept'] = probs[:, 1] if model.classes_[1] == 'accept' else probs[:, 0]
            pca_df['prob_reject'] = probs[:, 0] if model.classes_[0] == 'reject' else probs[:, 1]
            
            # Calculate uncertainty (closer to 0.5 means more uncertain)
            pca_df['decision_uncertainty'] = (0.5 - abs(pca_df['prob_accept'] - 0.5)) / 0.5
            
            return pca_df
    
    return None

def tracks_find_independent_set(all_pca_results, uncertainty_threshold=0.8):
    """
    Find set S where X_s is likely independent of slp_decision
    based on the uncertainty threshold from SVC predictions
    """
    # Combine all restaurant results
    combined_df = pd.concat(all_pca_results, ignore_index=True)
    
    # Identify points where the model is uncertain (close to random prediction)
    # These points likely form set S where X_s is independent of slp_decision
    set_S = combined_df[combined_df['decision_uncertainty'] >= uncertainty_threshold].copy()
    
    print(f"Found {len(set_S)} points in set S with uncertainty threshold {uncertainty_threshold}")
    
    return set_S

def tracks_evaluate_independence(set_S, pivot_tracks):
    """
    Evaluate whether X_s is independent of slp_decision for the identified set S
    using chi-square test of independence
    """
    if set_S.empty:
        print("Set S is empty, cannot evaluate independence")
        return
    
    # Get original indices
    original_indices = set_S['original_index'].values
    
    # Create contingency table for chi-square test
    # We'll use PC1 binned into quartiles as a representative of X_s
    set_S['PC1_bin'] = pd.qcut(set_S['PC1'], 4, labels=False)
    
    # Only use accept and reject decisions
    test_df = set_S[set_S['slp_decision'].isin(['accept', 'reject'])].copy()
    
    if len(test_df) < 20:
        print("Not enough data points for reliable chi-square test")
        return
    
    # Create contingency table
    contingency = pd.crosstab(test_df['PC1_bin'], test_df['slp_decision'])
    
    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    print("\nChi-square test for independence:")
    print(f"Chi2 statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if p > 0.05:
        print("The test suggests X_s is independent of slp_decision in set S (p > 0.05)")
    else:
        print("The test suggests X_s is not independent of slp_decision in set S (p <= 0.05)")
    return chi2, p


# Main execution
def main():
    # Placeholder - assuming pivot_tracks is loaded
    # In real implementation, you would load your data here
    
    all_pca_results = []
    
    # Process each restaurant
    for restaurant_id in range(1, 5):  # Restaurants 1, 2, 3, 4
        print(f"\n{'='*50}")
        print(f"Processing Restaurant {restaurant_id}")
        print(f"{'='*50}\n")
        
        # Fit PCA
        pca_df, pca_model = fit_pca_for_restaurant(pivot_tracks, restaurant_id)
        
        # Plot PCA results
        plot_pca_results(pca_df, restaurant_id)
        
        # Train SVC with grid search
        svc_model, filtered_df = train_svc_for_restaurant(pca_df, restaurant_id)
        
        # Get prediction probabilities
        if svc_model is not None:
            result_df = get_model_probabilities(svc_model, pca_df, restaurant_id)
            if result_df is not None:
                all_pca_results.append(result_df)
    
    # Find set S where X_s is likely independent of slp_decision
    # Try different thresholds for uncertainty
    for threshold in [0.10, 0.15, 0.20]:
        print(f"\n{'='*50}")
        print(f"Evaluating with uncertainty threshold: {threshold}")
        print(f"{'='*50}\n")
        
        set_S = find_independent_set(all_pca_results, uncertainty_threshold=threshold)
        
        # Evaluate independence
        evaluate_independence(set_S, pivot_tracks)
    
    # Save the final set S
    set_S = find_independent_set(all_pca_results, uncertainty_threshold=0.15)  # Middle threshold
    set_S.to_csv(os.path.join(cache_folder, 'independent_set_S.csv'), index=False)
    
    # Extract original indices for easy reference
    original_indices = set_S['original_index'].values
    pd.DataFrame({'original_index': original_indices}).to_csv(
        os.path.join(cache_folder, 'set_S_indices.csv'), index=False
    )
    
    print("\nAnalysis complete. Results saved to cache folder.")
