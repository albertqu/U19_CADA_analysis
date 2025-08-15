import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, make_scorer
from typing import Dict, List, Union, Callable, Optional, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import re
from copy import deepcopy
import time


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from typing import Dict, List, Union, Any, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns


def cross_validate_models(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    metrics: Dict[str, Callable] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    plot_results: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform k-fold cross-validation on multiple models and return their performance metrics.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of models to evaluate with {model_name: model_instance}
    X : pd.DataFrame
        Features data
    y : pd.Series
        Target data
    metrics : Dict[str, Callable], default=None
        Dictionary of metrics to evaluate {metric_name: metric_function}
        If None, uses balanced_accuracy, accuracy and f1_score for classification
    cv : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducible results
    verbose : bool, default=True
        Whether to print progress and fold-wise scores
    plot_results : bool, default=True
        Whether to plot comparison of model performance
        
    Returns:
    --------
    Dict[str, Dict[str, List[float]]]
        Dictionary with model names as keys and dictionaries of metric scores as values
    """
    # Default metrics (same as in original code)
    if metrics is None:
        metrics = {
            'bac': balanced_accuracy_score,
            'acc': accuracy_score,
            'f1': f1_score
        }
    
    # Initialize KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Dictionary to store results for all models
    all_results = {}
    
    # Loop through each model
    for model_name, model in models.items():
        if verbose:
            print(f"\nEvaluating {model_name}:")
        
        # Initialize scores for this model
        model_scores = {metric_name: [] for metric_name in metrics.keys()}
        
        # Perform cross-validation
        fold = 1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            
            # Handle different y types (Series, DataFrame, numpy array)
            # ensuring right format
            if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                y_train = y.iloc[train_index].squeeze()
                y_test = y.iloc[test_index].squeeze()
            else:  # Assume numpy array
                y_train = y[train_index].squeeze()
                y_test = y[test_index].squeeze()
            
            
            # Create a fresh copy of the model to avoid data leakage
            model_copy = clone_model(model)
            
            # Fit the model
            model_copy.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_test)
            
            # Calculate metrics
            fold_scores = {}
            for metric_name, metric_func in metrics.items():
                score = metric_func(y_test, y_pred)
                model_scores[metric_name].append(score)
                fold_scores[metric_name] = score
            
            if verbose:
                print(f"  Fold {fold}:", end=" ")
                for metric_name, score in fold_scores.items():
                    print(f"{metric_name}: {score:.4f}", end=" ")
                print()
            
            fold += 1
        
        # Store results for this model
        all_results[model_name] = model_scores
        
        # Print average scores for this model
        if verbose:
            print(f"  Mean scores for {model_name}:", end=" ")
            for metric_name, scores in model_scores.items():
                print(f"{metric_name}: {np.mean(scores):.4f}", end=" ")
            print()
    
    # Plot results if requested
    if plot_results:
        plot_cv_comparison(all_results)
    
    return all_results





def plot_cv_comparison(results: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot comparison of model performance across different metrics.
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, List[float]]]
        Dictionary with model names as keys and dictionaries of metric scores as values
    """
    # Get all metric names
    metric_names = list(next(iter(results.values())).keys())
    
    # Create figure with one subplot per metric
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6), squeeze=False)
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[0, i]
        
        # Prepare data for plotting
        plot_data = []
        for model_name, model_results in results.items():
            for score in model_results[metric_name]:
                plot_data.append({
                    'Model': model_name,
                    'Score': score,
                    'Metric': metric_name
                })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot
        sns.boxplot(x='Model', y='Score', data=plot_df, ax=ax, color='gray')
        
        # Add mean values as text
        for j, model_name in enumerate(results.keys()):
            mean_val = np.mean(results[model_name][metric_name])
            ax.text(j, mean_val, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Set title and labels
        ax.set_title(f'{metric_name}')
        ax.set_ylabel('Score')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    sns.despine()


def get_summary_df(results: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Create a summary DataFrame with mean and std of all metrics for all models.
    
    Parameters:
    -----------
    results : Dict[str, Dict[str, List[float]]]
        Dictionary with model names as keys and dictionaries of metric scores as values
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with mean and std for each model and metric
    """
    summary_data = []
    
    for model_name, model_results in results.items():
        model_summary = {'Model': model_name}
        
        for metric_name, scores in model_results.items():
            model_summary[f'{metric_name}_mean'] = np.mean(scores)
            model_summary[f'{metric_name}_std'] = np.std(scores)
        
        summary_data.append(model_summary)
    
    return pd.DataFrame(summary_data)


def feature_group_cross_validation(
    models: Dict[str, any],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    mode: str = 'group_elimination',
    group_pattern: str = r'([^_]+)_b\d+',
    lag_pattern: str = r'[^_]+_b(\d+)',
    cv: int = 5,
    scoring: Dict[str, Callable] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    max_lags_to_keep: int = 5,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Perform cross-validation with systematic feature group elimination or lag selection.
    
    Parameters:
    -----------
    models : Dict[str, any]
        Dictionary of models to evaluate {model_name: model_instance}
    X : pd.DataFrame
        Features dataframe with column names following pattern '[group]_b[lag]'
    y : Union[pd.Series, np.ndarray]
        Target variable
    mode : str, default='group_elimination'
        'group_elimination': Remove one feature group at a time
        'lag_selection': Try different maximum lag values
        'combined': Test both group elimination and lag selection
    group_pattern : str, default=r'([^_]+)_b\d+'
        Regex pattern to extract group name from feature names
    lag_pattern : str, default=r'[^_]+_b(\d+)'
        Regex pattern to extract lag number from feature names
    cv : int, default=5
        Number of folds for cross-validation
    scoring : Dict[str, Callable], default=None
        Dictionary of scoring metrics {metric_name: metric_function}
    random_state : int, default=42
        Random seed for reproducible results
    n_jobs : int, default=-1
        Number of CPU cores to use
    max_lags_to_keep : int, default=5
        Maximum number of lags to try when mode='lag_selection'
    verbose : bool, default=True
        Whether to print progress information
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing results for each experiment:
        - 'group_results': Results when eliminating each group
        - 'lag_results': Results when using different max lag values
        - 'best_configurations': Summary of best configurations
    """
    # Default scoring metrics if none provided
    if scoring is None:
        scoring = {
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score)
        }
    
    # Initialize KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Extract all feature groups
    feature_groups = set()
    for col in X.columns:
        match = re.match(group_pattern, col)
        if match:
            feature_groups.add(match.group(1))
    
    # Extract all lag values
    lag_values = set()
    for col in X.columns:
        match = re.match(lag_pattern, col)
        if match:
            lag_values.add(int(match.group(1)))
    
    # Set the maximum lag value
    max_lag = max(lag_values) if lag_values else 0
    
    # Dictionary to store all results
    all_results = {
        'group_elimination': pd.DataFrame(),
        'lag_selection': pd.DataFrame(),
        'best_configurations': pd.DataFrame()
    }
    
    # 1. Group Elimination: Test removing one group at a time
    if mode in ['group_elimination', 'combined']:
        if verbose:
            print(f"\n{'='*50}\nTesting feature group elimination\n{'='*50}")
            
        group_results = []
        
        # Baseline: use all features
        baseline_result = _evaluate_feature_set(
            models=models,
            X=X,
            y=y,
            kf=kf,
            scoring=scoring,
            feature_subset=X.columns,
            experiment="All Features (Baseline)",
            n_jobs=n_jobs,
            verbose=verbose
        )
        group_results.append(baseline_result)
        
        # Test removing each group
        for group in sorted(feature_groups):
            if verbose:
                print(f"\nEvaluating models without '{group}' features")
            
            # Select columns that don't belong to the current group
            cols_to_keep = [col for col in X.columns if not re.match(f"{group}_b\\d+", col)]
            
            # Skip if removing this group would remove all features
            if not cols_to_keep:
                if verbose:
                    print(f"  Skipping '{group}' - would remove all features")
                continue
            
            result = _evaluate_feature_set(
                models=models,
                X=X,
                y=y,
                kf=kf,
                scoring=scoring,
                feature_subset=cols_to_keep,
                experiment=f"Without {group}",
                n_jobs=n_jobs,
                verbose=verbose
            )
            group_results.append(result)
        
        # Combine results
        all_results['group_elimination'] = pd.concat(group_results)
    
    # 2. Lag Selection: Test different maximum lag values
    if mode in ['lag_selection', 'combined']:
        if verbose:
            print(f"\n{'='*50}\nTesting different maximum lag values\n{'='*50}")
            
        lag_results = []
        
        # Try different maximum lag values
        lag_values_to_try = sorted(lag_values)
        if max_lags_to_keep and len(lag_values_to_try) > max_lags_to_keep:
            # Select a subset of lags to try
            lag_values_to_try = sorted(lag_values)[-max_lags_to_keep:]
        
        for max_lag_to_keep in lag_values_to_try:
            if verbose:
                print(f"\nEvaluating models with maximum lag {max_lag_to_keep}")
            
            # Select columns with lag <= max_lag_to_keep
            cols_to_keep = []
            for col in X.columns:
                match = re.match(lag_pattern, col)
                if match and int(match.group(1)) <= max_lag_to_keep:
                    cols_to_keep.append(col)
            
            # Skip if this would remove all features
            if not cols_to_keep:
                if verbose:
                    print(f"  Skipping max_lag={max_lag_to_keep} - would remove all features")
                continue
            
            result = _evaluate_feature_set(
                models=models,
                X=X,
                y=y,
                kf=kf,
                scoring=scoring,
                feature_subset=cols_to_keep,
                experiment=f"Max Lag {max_lag_to_keep}",
                n_jobs=n_jobs,
                verbose=verbose
            )
            lag_results.append(result)
        
        # Combine results
        all_results['lag_selection'] = pd.concat(lag_results)
    
    # 3. Find best configurations
    best_configs = _find_best_configurations(all_results, scoring)
    all_results['best_configurations'] = best_configs
    
    # 4. Plot results
    _plot_results(all_results)
    
    return all_results


def _evaluate_feature_set(
    models: Dict[str, any],
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    kf: KFold,
    scoring: Dict[str, Callable],
    feature_subset: List[str],
    experiment: str,
    n_jobs: int,
    verbose: bool
) -> pd.DataFrame:
    """
    Evaluate all models using the specified feature subset.
    """
    results = []
    
    # Use only the selected features
    X_subset = X[feature_subset]
    
    for model_name, model in models.items():
        if verbose:
            print(f"  Evaluating {model_name} with {len(feature_subset)} features...")
            start_time = time.time()
        
        # Create a fresh copy of the model
        model_copy = deepcopy(model)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model_copy,
            X_subset,
            y,
            cv=kf,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True
        )
        
        # Process results
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            
            result = {
                'Model': model_name,
                'Experiment': experiment,
                'Metric': metric,
                'Mean Score': mean_score,
                'Std Score': std_score,
                'Num Features': len(feature_subset)
            }
            results.append(result)
        
        if verbose:
            elapsed = time.time() - start_time
            mean_scores = {k.replace('test_', ''): np.mean(v) for k, v in cv_results.items() if k.startswith('test_')}
            scores_str = ' '.join([f"{k}: {v:.4f}" for k, v in mean_scores.items()])
            print(f"    Completed in {elapsed:.2f}s - {scores_str}")
    
    return pd.DataFrame(results)


def _find_best_configurations(results: Dict[str, pd.DataFrame], scoring: Dict[str, Callable]) -> pd.DataFrame:
    """
    Find the best configurations for each model and metric.
    """
    best_configs = []
    
    # Process group elimination results if available
    if not results['group_elimination'].empty:
        for model in results['group_elimination']['Model'].unique():
            for metric in scoring.keys():
                model_metric_df = results['group_elimination'][
                    (results['group_elimination']['Model'] == model) & 
                    (results['group_elimination']['Metric'] == metric)
                ]
                
                # Find best experiment
                best_idx = model_metric_df['Mean Score'].idxmax()
                best_row = model_metric_df.loc[best_idx].copy()
                best_row['Category'] = 'Group Elimination'
                best_configs.append(best_row)
    
    # Process lag selection results if available
    if not results['lag_selection'].empty:
        for model in results['lag_selection']['Model'].unique():
            for metric in scoring.keys():
                model_metric_df = results['lag_selection'][
                    (results['lag_selection']['Model'] == model) & 
                    (results['lag_selection']['Metric'] == metric)
                ]
                
                # Find best experiment
                best_idx = model_metric_df['Mean Score'].idxmax()
                best_row = model_metric_df.loc[best_idx].copy()
                best_row['Category'] = 'Lag Selection'
                best_configs.append(best_row)
    
    return pd.DataFrame(best_configs)


def _plot_results(results: Dict[str, pd.DataFrame]) -> None:
    """
    Plot the results of feature group elimination and lag selection.
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Group Elimination Plot
    if not results['group_elimination'].empty:
        # Get unique models and metrics
        models = results['group_elimination']['Model'].unique()
        metrics = results['group_elimination']['Metric'].unique()
        
        # Create subplots - one row per metric, one column per model
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), squeeze=False)
        fig.suptitle('Feature Group Elimination Results', fontsize=16)
        
        for i, metric in enumerate(metrics):
            # Filter data for this metric
            metric_df = results['group_elimination'][results['group_elimination']['Metric'] == metric]
            
            # Create barplot
            ax = axes[i, 0]
            sns.barplot(
                data=metric_df,
                x='Experiment',
                y='Mean Score',
                hue='Model',
                ax=ax
            )
            
            # Add error bars
            for j, model in enumerate(models):
                model_df = metric_df[metric_df['Model'] == model]
                x_positions = np.arange(len(model_df))
                if len(models) > 1:
                    # Adjust x positions for grouped bars
                    width = 0.8 / len(models)
                    offset = j * width - 0.4 + width/2
                    x_positions = x_positions + offset
                
                # Get position of bars for this model
                if j == 0:
                    # For first model, get the actual x positions
                    container = ax.containers[j]
                    x_positions = [rect.get_x() + rect.get_width()/2 for rect in container]
                
                # Add error bars
                ax.errorbar(
                    x=x_positions,
                    y=model_df['Mean Score'].values,
                    yerr=model_df['Std Score'].values,
                    fmt='none',
                    color='black',
                    capsize=4
                )
            
            # Customize plot
            ax.set_title(f'{metric} by Feature Group Elimination')
            ax.set_ylabel(f'{metric} Score')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # 2. Lag Selection Plot
    if not results['lag_selection'].empty:
        # Get unique models and metrics
        models = results['lag_selection']['Model'].unique()
        metrics = results['lag_selection']['Metric'].unique()
        
        # Create subplots - one row per metric, one column per model
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)), squeeze=False)
        fig.suptitle('Maximum Lag Selection Results', fontsize=16)
        
        for i, metric in enumerate(metrics):
            # Filter data for this metric
            metric_df = results['lag_selection'][results['lag_selection']['Metric'] == metric]
            
            # Create barplot
            ax = axes[i, 0]
            sns.barplot(
                data=metric_df,
                x='Experiment',
                y='Mean Score',
                hue='Model',
                ax=ax
            )
            
            # Add error bars (same logic as above)
            for j, model in enumerate(models):
                model_df = metric_df[metric_df['Model'] == model]
                if j == 0:
                    container = ax.containers[j]
                    x_positions = [rect.get_x() + rect.get_width()/2 for rect in container]
                
                # Add error bars
                ax.errorbar(
                    x=x_positions,
                    y=model_df['Mean Score'].values,
                    yerr=model_df['Std Score'].values,
                    fmt='none',
                    color='black',
                    capsize=4
                )
            
            # Customize plot
            ax.set_title(f'{metric} by Maximum Lag')
            ax.set_ylabel(f'{metric} Score')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # 3. Best Configurations Plot
    if not results['best_configurations'].empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot data for heatmap
        best_df = results['best_configurations'][['Model', 'Metric', 'Mean Score', 'Experiment', 'Category']]
        pivot_df = best_df.pivot_table(
            index=['Model', 'Category'],
            columns='Metric',
            values='Mean Score',
            aggfunc='max'
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.4f',
            cmap='YlGnBu',
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Best Configurations')
        plt.tight_layout()
        plt.show()



###########################################
######### Claude workflow #################
###########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

def example():
    # correlation_feature_clustering(X, correlation_threshold=0.3, plot=True)
    # pca_feature_clustering(X, n_clusters=3, explained_variance_threshold=0.95, plot=True)
    # Load a dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Apply feature clustering
    selected_features, clusters = feature_clustering_workflow(
        X, y, 
        method='mutual_info',  # 'correlation', 'pca', or 'mutual_info'
        correlation_threshold=0.7,
        is_classification=True,
        n_clusters=5, 
        selection_method='mi',  # 'mi', 'variance', or 'central'
        plot=True
    )

    # Use selected features for model building
    X_selected = X[selected_features]
    print(f"Reduced features from {X.shape[1]} to {X_selected.shape[1]}")

def correlation_feature_clustering(X, correlation_threshold=0.7, plot=True):
    """
    Cluster features based on their correlation coefficient.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame containing the features
    correlation_threshold : float, default=0.7
        Threshold to determine clusters
    plot : bool, default=True
        Whether to plot the correlation heatmap and dendrogram
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to feature names
    """
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Convert correlation matrix to distance matrix
    distance_matrix = 1 - corr_matrix
    
    # Plot correlation heatmap
    if plot:
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()
    
    # Perform hierarchical clustering
    Z = linkage(squareform(distance_matrix), method='ward')
    
    # Plot dendrogram
    if plot:
        plt.figure(figsize=(14, 8))
        dendrogram(Z, labels=X.columns, leaf_rotation=90)
        plt.title('Feature Clustering Dendrogram')
        plt.xlabel('Features')
        plt.ylabel('Distance')
        plt.axhline(y=correlation_threshold, c='k', linestyle='--', label=f'Threshold = {correlation_threshold}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Form clusters
    max_dist = 1 - correlation_threshold
    clusters = fcluster(Z, max_dist, criterion='distance')
    
    # Create dictionary of clusters
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(X.columns[i])
    
    return cluster_dict


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def pca_feature_clustering(X, n_clusters=None, explained_variance_threshold=0.95, plot=True):
    """
    Cluster features using PCA loadings.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame containing the features
    n_clusters : int or None, default=None
        Number of clusters to form. If None, determined by explained variance
    explained_variance_threshold : float, default=0.95
        Threshold for cumulative explained variance if n_clusters is None
    plot : bool, default=True
        Whether to plot the PCA results
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to feature names
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of components
    if n_clusters is None:
        pca_full = PCA()
        pca_full.fit(X_scaled)
        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components that explain desired variance
        n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    else:
        n_components = n_clusters
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Get loadings
    loadings = pca.components_.T
    
    # Plot explained variance
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=explained_variance_threshold, c='r', linestyle='--')
        plt.grid()
        plt.show()
    
    # Use the loadings to cluster features
    if n_clusters is None:
        n_clusters = n_components
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(loadings)
    
    # Create dictionary of clusters
    cluster_dict = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(X.columns[i])
    
    return cluster_dict


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def mutual_info_clustering(X, y, is_classification=True, n_clusters=5, plot=True):
    """
    Cluster features based on mutual information with target 
    and correlation between features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame containing the features
    y : array-like
        Target variable
    is_classification : bool, default=True
        Whether this is a classification problem
    n_clusters : int, default=5
        Number of clusters to form
    plot : bool, default=True
        Whether to plot results
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to feature names
    """
    # Calculate mutual information with target
    if is_classification:
        mi_scores = mutual_info_classif(X, y)
    else:
        mi_scores = mutual_info_regression(X, y)
    
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Create feature similarity matrix combining MI and correlation
    n_features = X.shape[1]
    feature_similarity = np.zeros((n_features, n_features))
    
    mi_score_dict = dict(zip(mi_df['Feature'], mi_df['MI_Score']))
    
    for i, feat1 in enumerate(X.columns):
        for j, feat2 in enumerate(X.columns):
            # Similar MI score with target + low correlation = different clusters
            mi_similarity = 1 - abs(mi_score_dict[feat1] - mi_score_dict[feat2]) / max(mi_df['MI_Score'])
            corr_similarity = corr_matrix.loc[feat1, feat2]
            
            # Weight MI similarity more than correlation
            feature_similarity[i, j] = 0.7 * mi_similarity + 0.3 * corr_similarity
    
    # Cluster features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_similarity)
    
    # Create dictionary of clusters
    cluster_dict = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(X.columns[i])
    
    # Plot MI scores
    if plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='MI_Score', y='Feature', data=mi_df)
        plt.title('Feature Mutual Information with Target')
        plt.tight_layout()
        plt.show()
    
    return cluster_dict


def select_representatives(X, y, cluster_dict, method='mi', is_classification=True):
    """
    Select representative features from each cluster.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame containing the features
    y : array-like
        Target variable
    cluster_dict : dict
        Dictionary mapping cluster IDs to feature names
    method : str, default='mi'
        Method to select representatives:
        - 'mi': highest mutual information
        - 'variance': highest variance
        - 'central': most correlated with other features in cluster
    is_classification : bool, default=True
        Whether this is a classification problem
        
    Returns:
    --------
    list
        List of selected feature names
    """
    selected_features = []
    
    if method == 'mi':
        # Calculate mutual information
        if is_classification:
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        mi_dict = dict(zip(X.columns, mi_scores))
        
        # Select feature with highest MI from each cluster
        for cluster_id, features in cluster_dict.items():
            if features:  # Ensure cluster is not empty
                best_feature = max(features, key=lambda f: mi_dict[f])
                selected_features.append(best_feature)
    
    elif method == 'variance':
        # Calculate variance
        var_dict = dict(zip(X.columns, X.var().values))
        
        # Select feature with highest variance from each cluster
        for cluster_id, features in cluster_dict.items():
            if features:
                best_feature = max(features, key=lambda f: var_dict[f])
                selected_features.append(best_feature)
    
    elif method == 'central':
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select most central feature from each cluster
        for cluster_id, features in cluster_dict.items():
            if len(features) == 1:
                selected_features.append(features[0])
            elif features:
                # For each feature, calculate average correlation with other features in cluster
                avg_corr = {}
                for feat in features:
                    corrs = [corr_matrix.loc[feat, other] for other in features if other != feat]
                    avg_corr[feat] = np.mean(corrs) if corrs else 0
                
                # Select feature with highest average correlation
                best_feature = max(avg_corr.items(), key=lambda x: x[1])[0]
                selected_features.append(best_feature)
    
    return selected_features


def feature_clustering_workflow(X, y, method='correlation', n_clusters=None, 
                               correlation_threshold=0.7, is_classification=True,
                               selection_method='mi', plot=True):
    """
    Complete workflow for feature clustering and selection.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame containing the features
    y : array-like
        Target variable
    method : str, default='correlation'
        Clustering method: 'correlation', 'pca', or 'mutual_info'
    n_clusters : int or None, default=None
        Number of clusters to form
    correlation_threshold : float, default=0.7
        Threshold for correlation clustering
    is_classification : bool, default=True
        Whether this is a classification problem
    selection_method : str, default='mi'
        Method to select representatives: 'mi', 'variance', 'central'
    plot : bool, default=True
        Whether to plot results
        
    Returns:
    --------
    tuple
        (selected features list, cluster dictionary)
    """
    # Step 1: Cluster features
    if method == 'correlation':
        cluster_dict = correlation_feature_clustering(X, correlation_threshold, plot)
    elif method == 'pca':
        cluster_dict = pca_feature_clustering(X, n_clusters, plot=plot)
    elif method == 'mutual_info':
        cluster_dict = mutual_info_clustering(X, y, is_classification, n_clusters, plot)
    else:
        raise ValueError("Method must be one of: 'correlation', 'pca', 'mutual_info'")
    
    # Step 2: Select representatives from each cluster
    selected_features = select_representatives(X, y, cluster_dict, selection_method, is_classification)
    
    # Print results
    print(f"Identified {len(cluster_dict)} feature clusters")
    for cluster_id, features in cluster_dict.items():
        selected = [f for f in features if f in selected_features]
        print(f"Cluster {cluster_id}: {len(features)} features")
        print(f"  - Selected: {selected}")
        print(f"  - All features: {features}")
    
    print(f"\nFinal selection: {len(selected_features)} features")
    print(selected_features)
    
    return selected_features, cluster_dict