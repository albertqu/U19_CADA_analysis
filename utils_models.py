# system
import os, re
from numbers import Number
# data
import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster, preprocessing
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, explained_variance_score, r2_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import KFold
import logging

try:
    from umap.umap_ import UMAP
except:
    logging.warning('UMAP not installed, some functions might be unsupported')
try:
    from xgboost import XGBClassifier, XGBRegressor
except:
    print("Warning! XGBoost cannot be loaded!")

# plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
try:
    import plotly.express as px
except:
    print("Warning: Plotly cannot be loaded!")

RAND_STATE = 230


# PRINCIPLE: CODE: 0 INDEX, PLOTTING: 1 INDEX
""" ##########################################
################ Preprocessing ###############
########################################## """


def load_session(root, opt='all'):
    file_map = {'MAS': "WilbrechtLab_4ChoiceRevDataALL_MASTERFILE_081513.csv",
                'MAS2': 'WilbrechtLab_4ChoiceRevDataALL_MASTERFILE_051820.csv',
                '4CR': "4CR_Merge_052920.csv",
                'FIP': "FIP Digging Summary Master File.csv",
                'FIP_RL': "FIP_RL_061120.csv"}
    if isinstance(opt, str):
        if opt == 'all':
            opts = file_map.keys()
        else:
            opts = []
            assert opt in file_map, 'dataset not found'
            return pd.read_csv(os.path.join(root, file_map[opt]))
    else:
        opts = opt
    pdfs = [pd.read_csv(os.path.join(root, file_map[o])) for o in opts]
    return pd.concat(pdfs, axis=0, join='outer', ignore_index=True)


def label_feature(pdf, criteria='num', verbose=False):
    # OBSOLETE
    results = []
    for c in pdf.columns:
        try:
            cdata = pdf[c]
            if criteria == 'num':
                l = int(np.issubdtype(cdata.dtype, np.number) or np.all(cdata.dropna().str.isnumeric()))
            elif criteria == 'full':
                l = int(not np.any(cdata.isnull()))
            else:
                raise NotImplementedError('Unknown criteria: ', criteria)
        except:
            l = -1
        results.append(l)
        if verbose:
            print(c, l)
    return np.array(results)


def dataset_split_feature_classes(pdf):
    classes = {'other': []}
    for c in pdf.columns:
        if '.' in c:
            cnames = c.split('.')
            classe = ".".join(cnames[:-1])
            if classe in classes:
                classes[classe].append(c)
            else:
                classes[classe] = [c]
        else:
            classes['other'].append(c)
    return {cl: pdf[classes[cl]] for cl in classes}


def dataset_feature_cleaning(pdf, spec_map):
    # TODO: implement merger
    # TODO: maybe at some point drop rows instead of slicing them to rid the warnings?
    # TODO: move function outside to form a new function with a new table
    # Search for specific selectors
    for s in spec_map:
        # TODO: add function to allow multiplexing handling
        if spec_map[s] == 'selD':
            # Select rows with nonnull values
            pdf = pdf.loc[~pdf[s].isnull()]
        elif spec_map[s] == 'selF':
            pdf[s].values[pdf[s].isnull()]= -1
            # pdf[s] = pdf[s].astype(np.int)
        elif isinstance(spec_map[s], dict):
            # Select
            assert 'sel' in spec_map[s], print('Bad Usage of Dictionary Coding, no drop nor sel')
            sel_list = spec_map[s]['sel']
            pdf = pdf.loc[pdf[s].isin(sel_list)]
    return pdf


def vanilla_vectorizer(pdf):
    keywords = pdf.columns[label_feature(pdf) == 1][:-2]
    X0 = pdf[keywords].astype(np.float64)
    X_nonan = X0.dropna()
    return X_nonan


def MAS_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
    """

    :param pdf:
    :param inputRange: list (denoting the column indices) or int (first inputRange columns are inputs)
    :param NAN_policy:
    :return:
    """
    spec_map = {
        'exp1_label_FI_AL_M': 'selF',
        'exp2_Angel': 'selF',
        'Treat': 'cat',
        'Age At Treat': 'cat',
        'Genotype': 'cat',
        'Experimenter': 'cat'
    }
    # test.weight
    testweight = pdf['test.weight']
    for i in range(testweight.shape[0]):
        target = testweight.iloc[i]
        if ' at ' in str(target):
            dp = float(target.split(" ")[0])
            pdf['test.weight'].values[i] = dp

    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    D = pdf.shape[-1]
    for irg, rg in enumerate(inputRange):
        if rg < 0:
            inputRange[irg] = D+rg

    outputRange = np.setdiff1d(np.arange(D), inputRange)
    pdf = dataset_feature_cleaning(pdf, spec_map)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, NAN_policy, spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def FOURCR_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
    """

    :param pdf:
    :param inputRange: list (denoting the column indices) or int (first inputRange columns are inputs)
    :param NAN_policy:
    :return:
    """
    spec_map = {
        'exp1_label_FI_AL_M': 'selF',
        'exp2_Angel': 'selF',
        'Treat': 'cat',
        'Genotype': 'cat',
        'Experimenter': 'cat',
        'ID': 'cat' # Figure out better way to encode this
    }
    # test.weight
    testweight = pdf['test.weight']
    for i in range(testweight.shape[0]):
        target = testweight.iloc[i]
        if ' at ' in str(target):
            dp = float(target.split(" ")[0])
            pdf['test.weight'].values[i] = dp

    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    D = pdf.shape[-1]
    for irg, rg in enumerate(inputRange):
        if rg < 0:
            inputRange[irg] = D+rg

    outputRange = np.setdiff1d(np.arange(D), inputRange)
    pdf = dataset_feature_cleaning(pdf, spec_map)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, 'ignore', spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def FIP_data_vectorizer(pdf, inputRange, NAN_policy='drop'):
    """
    :param pdf:
    :param inputRange: list (denoting the column indices) or int (first inputRange columns are inputs)
    :param NAN_policy:
    :return:
    """
    spec_map = {
        'Experimental Cohort': 'cat',
        'FIP Duration': 'cat',
        'Behavior Duration': 'cat',
        'exp_treat_sex': 'cat'
    }
    if not isinstance(inputRange, list):
        inputRange = np.arange(inputRange)
    outputRange = np.setdiff1d(np.arange(pdf.shape[-1]), inputRange)
    inPDF, outPDF = pdf.iloc[:, inputRange], pdf.iloc[:, outputRange]
    tPDF, tLabels, tFRows = default_vectorizer(inPDF, 'ignore', spec_map, True)
    xPDF, xLabels, xFRows = default_vectorizer(outPDF, NAN_policy, spec_map, True)
    mRows = tFRows & xFRows
    return (tPDF.loc[mRows], tLabels.loc[mRows]), (xPDF.loc[mRows], xLabels.loc[mRows])


def default_vectorizer(pdf, NAN_policy='drop', spec_map=None, finalROW=False):
    """ Taking in pdf with raw data, returns tuple(PDF_feature, encoding PDF)
    Every modification to rows must be recorded if the features are to be used in a (Input, Output) model
    and finalRow must be `True`.
    # TODO: when setting data values, use pdf.values[slice, slice] = value
    :param pdf:
    :param spec_map:
        cat: for categorical label
        selF for setting blank label values as -1; selD for deleting blank label values
    :return:
    """
    # NAN_policy: fill additional value (minimum for instance)
    if spec_map is None:
        spec_map = {}

    # Check drop
    droplist = [c for c in pdf.columns if c in spec_map and spec_map[c] == 'drop']
    pdf.drop(columns=droplist, inplace=True)

    class_table = classify_features(pdf)
    # class 2, special treatment out front
    SPECIAL_MAP = class_table['class'] == 2
    special_features = class_table['feature'][SPECIAL_MAP]
    # ONLY special features that got converted to other types will be kept in the final data frame
    for spe in special_features:
        sdata = pdf[spe]
        null_flags = sdata.isnull()
        nonnull = sdata[~null_flags]
        m = re.search("(\d+)/(\d+)/(\d+)", nonnull.iloc[0])
        if m:
            # TODO: if some relationship were to be found, we could split date code to three column year,
            #  month, date to probe seasonality
            class_table['class'][class_table['feature'] == spe] = 1
            for i in range(sdata.shape[0]):
                if not null_flags[i]:
                    m = re.search("(\d+)/(\d+)/(\d+)", sdata.iloc[i])
                    g3 = m.group(3)
                    if len(g3) == 4:
                        g3 = g3[-2:]
                    datecode = int(f"{int(g3):02d}{int(m.group(1)):02d}{int(m.group(2)):02d}")
                    pdf[spe].values[i] = datecode
                else:
                    pdf[spe].values[i] = np.nan

        elif spe in spec_map:
            class_table['class'][class_table['feature'] == spe] = 0
            if spec_map[spe] == 'cat':
                del spec_map[spe]
        else:
            for i, s in enumerate(nonnull):
                try:
                    if str(s).startswith("#"):
                        try:
                            pdf[spe].replace({s: np.nan}, inplace=True)
                            pdf[spe] = pdf[spe].astype(np.float)
                            class_table['class'][class_table['feature'] == spe] = 1
                        except:
                            print(f'feature {spe} needs special handling')
                        break
                except:
                    print(spe, i, 'nonnull')
                    raise RuntimeError()

    # class 0, encode, use hot map
    ALPHAS_MAP = class_table['class'] == 0
    alpha_features = class_table['feature'][ALPHAS_MAP]
    pdf_ENC = pdf.loc[:, alpha_features].copy(deep=True)
    for alp in alpha_features:
        if alp in spec_map:
            # spec_map features demand special encoding
            assert isinstance(spec_map[alp], dict), 'special map must be a dictionary'
            pdf[alp].replace(spec_map[alp], inplace=True)
        else:
            null_alp = pdf[alp].isnull()
            unique_vals = pdf[alp].dropna().unique()
            if len(unique_vals) < 3:
                # Default binary coding
                pdf[alp]=pdf[alp].astype('category')
                pdf_ENC[alp] = pdf[alp]
                pdf.loc[:, alp] = pdf[alp].cat.codes.astype(np.float)
                # TODO: clean the setter algorithms by converting everything to normal
                pdf[alp].values[null_alp] = np.nan
            else:
                # ONE HOT ENCODING if more than 2 possible values
                # NULL value be sure to mark back to null
                pdf = pd.get_dummies(pdf, prefix_sep='__', columns=[alp], dtype=np.float)
                pdf[[c for c in pdf.columns if c.startswith(alp+'__')]].values[null_alp] = np.nan
            # if NAN greater than 50%, apply NAN_policy first

    NUM_MAP = class_table['class'] == 1
    num_features = class_table['feature'][NUM_MAP]
    pdf.drop(columns=np.setdiff1d(class_table['feature'],
                                        np.concatenate((alpha_features, num_features))), inplace=True)

    pdf = pdf.astype(np.float)
    # DISPOSE OF columns that have too many nans TODO: IMPLEMENT IGNORE POLICY OF NANS
    if NAN_policy == 'drop':
        THRES = 0.3
        dropped_cols = class_table['null_ratio'] >= THRES
        dropped_features = class_table['feature'][dropped_cols]
        dropped_alpha_features = class_table['feature'][ALPHAS_MAP & dropped_cols]
        pdf_ENC.drop(columns=dropped_alpha_features, inplace=True)
        droppingFs = []
        for feat in dropped_features:
            if feat in pdf.columns:
                droppingFs.append(feat)
            else:
                droppingFs = droppingFs + [c for c in pdf.columns if c.startswith(feat+'__')]
        pdf.drop(columns=droppingFs, inplace=True)
        nonullrows = ~pdf.isnull().any(axis=1)
    elif NAN_policy == 'ignore':
        pdf_ENC.values[pdf_ENC.isnull()] = np.nan
        pdf.values[pdf.isnull()] = np.nan
        nonullrows = np.full(pdf.shape[0], 1, dtype=bool)
    else:
        raise NotImplementedError(f'unknown policy {NAN_policy}')
    if finalROW:
        return pdf, pdf_ENC, nonullrows
    else:
        pdf = pdf.loc[nonullrows]
        pdf_ENC = pdf_ENC.loc[nonullrows]
        return pdf, pdf_ENC


def categorical_feature_num_to_label(pdf, pdfENC, column):
    orig, val = column.split('__')
    return pdfENC[orig]


def classify_features(pdf, out=None):
    # cat: 0, num: 1, special: 2
    # out: tuple(folder, saveopt)
    class_table = {}
    class_table['feature'] = []
    class_table['class'] = []
    class_table['null_ratio'] = []
    for c in pdf.columns:
        class_table['feature'].append(c)
        cdata = pdf[c]
        null_sels = cdata.isnull()
        null_ratio = null_sels.sum() / len(cdata)
        nonnulls = cdata[~null_sels]
        if np.issubdtype(cdata.dtype, np.number) or np.all(nonnulls.str.isnumeric()):
            class_table['class'].append(1)
        elif np.all(nonnulls.str.isalpha()):
            class_table['class'].append(0)
        else:
            class_table['class'].append(2)
        class_table['null_ratio'].append(null_ratio)
    for k in class_table:
        class_table[k] = np.array(class_table[k])
    if out is not None:
        outfolder, saveopt = out
        pd.DataFrame(class_table).to_csv(os.path.join(outfolder, f'{saveopt}_classTable.csv'), index=False)
    return class_table


def get_data_label(tPDF, tLabels, label=None, STRIFY=True):
    # TODO: add merge label function if needed
    # Guarantee tLabels to be series
    if label is None:
        allLabels = tLabels.columns
        return set(np.concatenate([allLabels, [c for c in tPDF.columns if '__' not in c]]))
    else:
        targetDF = None
        if label in tLabels.columns:
            targetDF = tLabels
        elif label in tPDF.columns:
            targetDF = tPDF
        else:
            raise RuntimeError(f'Unknown Label: {label}!')
        LDLabels = targetDF[label]
        if STRIFY:
            LDLabels = LDLabels.astype('str')
        return LDLabels


""" ########################################
################ Validation ################
######################################## """


def auc_roc_2dist(d1, d2, method='QDA', k=5, return_dist=False):
    X = np.concatenate((d1, d2)).reshape((-1, 1))
    y = np.repeat([0, 1], len(d1))
    if method == 'QDA':
        clf = QDA()
    elif method == 'LR':
        clf = LogisticRegression(solver="liblinear", random_state=RAND_STATE)
    if k == 1:
        clf.fit(X, y)
        return roc_auc_score(y, clf.predict_proba(X)[:, 1])
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=RAND_STATE)
        roc_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            test_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            # for NaNs we replace it with the opposite of test label!
            nantp = np.isnan(test_probs)
            if np.sum(nantp) > 0:
                logging.warning("NaNs encountered in QDA prediction, interpret AUCROC with caution!")
                test_probs[nantp] = 1-y_test[nantp]
            roc_scores.append(roc_auc_score(y_test, test_probs))
        if return_dist:
            return roc_scores
        else:
            return np.mean(roc_scores)


def make_k_arange_segs(n, k):
    # alternative method making split more even
    inds = np.arange(n)
    # block_f = n / k
    # if block_f % 1 > 0.5:
    #     block = int(np.ceil(block_f))
    # else:
    #     block = int(np.floor(block_f))
    block = n // k
    for i in range(k):
        if i == k-1:
            yield inds[i*block:]
        else:
            yield inds[i*block:(i+1)*block]


def ksplit_X_y(X, y, k):
    for inds in make_k_arange_segs(len(X), k):
        yield X[inds], y[inds]


def simple_metric(y, y_pred):
    region = np.abs(y-np.mean(y)) <= np.std(y)
    return 1 - (np.std((y[region]-y_pred[region])) / np.std(y[region])) ** 2
    # return np.std((y[region]-y_pred[region]))


def fp_corrected_metric(y, y_pred, method):
    region = np.abs(y-np.mean(y)) <= np.std(y)
    metric_list = {'explained_variance': explained_variance_score,
                   'r2': r2_score}
    return metric_list[method](y[region], y_pred[region])


""" ##########################################
################ Visualization ###############
########################################## """


def visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=False):
    xs = np.arange(1, len(pca_full.singular_values_)+1)
    ys = np.cumsum(pca_full.explained_variance_ratio_)
    titre = dataset_name+' PCA dimension plot'
    if JNOTEBOOK_MODE:
        fig = px.line(pd.DataFrame({'components': xs, 'variance ratio': ys}),
                x='components', y='variance ratio', title=titre)
        fig.show()
    else:
        plt.plot(xs, ys, 'o-')
        plt.xlabel('# Components')
        plt.ylabel("cumulative % of variance")
        plt.title(titre)
        plt.show()


def dataset_dimensionality_probing(X_nonan_dm, dataset_name, visualize=True, verbose=True,
                                   JNOTEBOOK_MODE=False):
    # DIM Probing
    pca_full = PCA().fit(X_nonan_dm) # N x D -> N x K * K x D
    if visualize:
        visualize_dimensionality_ratios(pca_full, dataset_name, JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    #automatic_dimension probing
    # TODO: IMPLEMENT CROSS-VALIDATION BASED SELECTION
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ClusteringDim = np.where(cumsum >= 0.9)[0][0]+1 # Determine with the ratio plots MAS: 9
    variance_ratio = np.sum(pca_full.explained_variance_ratio_[:ClusteringDim])
    if verbose:
        print(f"Capturing {100*variance_ratio:.4f}% variance with {ClusteringDim} components")
    return pca_full, ClusteringDim


def visualize_loading_weights(pca, keywords, nth_comp, show=True):
    xs = np.arange(keywords.shape[0])
    loadings = pca.components_[nth_comp]
    pos = loadings >= 0
    plt.bar(xs[pos], loadings[pos], color='b', label='+weights')
    plt.bar(xs[~pos], -loadings[~pos], color='r', label='-weights')
    plt.xticks(xs, keywords, rotation=90)
    plt.legend()
    plt.subplots_adjust(bottom=0.3)
    if show:
        plt.show()


def save_loading_plots(decomp, keywords, model_name, plots):
    outpath = os.path.join(plots, f'{model_name}_numerical_features')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(decomp.components_.shape[0]):
        plt.figure(figsize=(15, 8))
        visualize_loading_weights(decomp, keywords, i, show=False)
        fname = os.path.join(outpath, f'{model_name}_numF_D{i+1}_loading')
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.eps')
        plt.close()


def visualize_feature_importance(values, names, thres=20, tag=''):
    feat_imps, feat_imps_stds = values
    sort_inds = np.argsort(feat_imps)[::-1]
    if thres is not None and len(feat_imps) > thres:
        feat_imps_slice = feat_imps[sort_inds[:thres]]
        feat_imps_stds_slice = feat_imps_stds[sort_inds[:thres]]
        names_slice = names[sort_inds[:thres]]
    else:
        feat_imps_slice, feat_imps_stds_slice = feat_imps[sort_inds], feat_imps_stds[sort_inds]
        names_slice = names[sort_inds]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    axes[0].hist(feat_imps, weights=np.full_like(feat_imps, 1/len(feat_imps)))
    axes[0].set_xlabel('feature importance')
    axes[0].set_ylabel('percent')

    xs = range(len(feat_imps_slice))
    axes[1].bar(xs, feat_imps_slice, yerr=feat_imps_stds_slice, align="center")
    axes[1].set(xticks=xs, xlim=[-1, len(xs)])
    axes[1].set_xticklabels(names_slice, rotation=60)
    fig.suptitle(tag)
    #plt.show()


def visualize_2D(X_HD, labels, dims, tag, show=True, out=None, label_alias=None, axis_alias=None,
                 vis_ignore=None, JNOTEBOOK_MODE=False):
    # TODO: implement saving for JNOTEBOOK_MODE
    """dims: 0 indexed"""
    # change clustering
    if vis_ignore is None:
        vis_ignore = []
    if not hasattr(dims, '__iter__'):
        dims = [dims, dims + 1]
    X_2D = X_HD[:, dims]
    c0n = f'Comp {dims[0]+1}' if axis_alias is None else axis_alias[0]
    c1n = f'Comp {dims[1]+1}' if axis_alias is None else axis_alias[1]
    titre = f'{tag}_2D_{dims[0]}-{dims[1]}_vis'
    if JNOTEBOOK_MODE:
        # NO need to ignore in interactive mode
        nlabels = len(labels.unique())
        # TODO: blend in real labels of treatments
        if labels.dtype == 'object':
            cseq = sns.color_palette("coolwarm", nlabels).as_hex()
            cmaps = {ic: cseq[i] for i, ic in enumerate(sorted(labels.unique()))}
            if label_alias is not None:
                newLabels = labels.copy(deep=True)
                newCMAPs = {}
                for l in labels.unique():
                    al = label_alias[l]
                    newLabels[labels == l] = al
                    newCMAPs[al] = cmaps[l]
                labels = newLabels
                cmaps = newCMAPs
        else:
            cmaps = None
        fig = px.scatter(pd.DataFrame({c0n: X_2D[:, 0], c1n: X_2D[:, 1],
                                 labels.name: labels.values}), x=c0n, y=c1n, color=labels.name,
                                 color_discrete_map=cmaps, title=titre)
        if show:
            fig.show()
    else:
        uniqlabels = np.setdiff1d(np.unique(labels), vis_ignore)
        nlabels = len(uniqlabels)
        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(15, 15))
            ax = plt.gca()
            sns.set_style(style='black')
            if labels.dtype == 'object':
                cseq = sns.color_palette('coolwarm', nlabels)
                for i, l in enumerate(sorted(uniqlabels)):
                    ax.scatter(X_2D[labels == l, 0], X_2D[labels == l, 1], color=cseq[i], label=l)
                ax.legend()
            else:
                cmap = sns.cubehelix_palette(as_cmap=True)
                uniqsel = np.isin(labels, uniqlabels)
                points = ax.scatter(X_2D[uniqsel, 0], X_2D[uniqsel, 1], c=labels[uniqsel], cmap=cmap)
                fig.colorbar(points)
            plt.title(f'{tag} 2D {dims} visualization')
            plt.xlabel(c0n)
            plt.ylabel(c1n)

            if show:
                plt.show()
            if out is not None:
                fname = os.path.join(out, titre)
                plt.savefig(fname+'.png')
                plt.savefig(fname+'.eps')
                plt.close()


def save_LD_plots(X_LD, labels, tag, out, show=True):
    # labels: pd.Series containing labels for different sample points
    outpath = os.path.join(out, tag+"_2D", 'labelGroup_'+labels.name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(X_LD.shape[1] - 1):
        visualize_2D(X_LD, labels, i, tag, show=show, out=outpath)


def visualize_3D(X_HD, labels, dims, tag, show=True, out=None, label_alias=None,
                 vis_ignore=None, axis_alias=None, JNOTEBOOK_MODE=False):
    # TODO: implement saving for JNOTEBOOK_MODE  ADD DATASET NAME MAYBE?
    """dims: 0 indexed"""
    if vis_ignore is None:
        vis_ignore = []
    # change clustering
    if not hasattr(dims, '__iter__'):
        dims = np.arange(dims, dims+3)
    assert X_HD.shape[1] >= 3, 'not enough dimensions try 2d instead'
    X_3D = X_HD[:, dims]
    c0n = f'Comp {dims[0]+1}' if axis_alias is None else axis_alias[0]
    c1n = f'Comp {dims[1]+1}' if axis_alias is None else axis_alias[1]
    c2n = f'Comp {dims[2]+1}' if axis_alias is None else axis_alias[2]
    titre = f'{tag}_3D_{dims[0]}-{dims[1]}-{dims[2]}_vis'

    if JNOTEBOOK_MODE:
        nlabels = len(labels.unique())
        # TODO: blend in real labels of treatments
        if labels.dtype == 'object':
            cseq = sns.color_palette("coolwarm", nlabels).as_hex()
            cmaps = {ic: cseq[i] for i, ic in enumerate(sorted(labels.unique()))}
            if label_alias is not None:
                newLabels = labels.copy(deep=True)
                newCMAPs = {}
                for l in labels.unique():
                    al = label_alias[l]
                    newLabels[labels == l] = al
                    newCMAPs[al] = cmaps[l]
                labels = newLabels
                cmaps = newCMAPs
        else:
            cmaps = None
        fig = px.scatter_3d(pd.DataFrame({c0n: X_3D[:, 0], c1n: X_3D[:, 1], c2n: X_3D[:, 2],
                                 labels.name: labels.values}), x=c0n, y=c1n, z=c2n,
                            color=labels.name, color_discrete_map=cmaps, title=titre)
        if show:
            fig.show()
    else:
        uniqlabels = np.setdiff1d(np.unique(labels), vis_ignore)
        nlabels = len(uniqlabels)
        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            sns.set_style(style='dark')
            if labels.dtype == 'object':
                cseq = sns.color_palette('coolwarm', nlabels).as_hex()
                for i, l in enumerate(sorted(uniqlabels)):
                    ax.scatter(X_3D[labels == l, 0], X_3D[labels == l, 1], X_3D[labels == l, 2], s=100,
                               color=cseq[i], label=l)
                ax.legend()
            else:
                cmap = sns.cubehelix_palette(as_cmap=True)
                uniqsel = np.isin(labels, uniqlabels)
                points = ax.scatter(X_3D[uniqsel, 0], X_3D[uniqsel, 1], X_3D[uniqsel, 2], s=100,
                                    c=labels[uniqsel], cmap=cmap)
                fig.colorbar(points)
            plt.title(f'{tag} 3D {dims} visualization')
            plt.xlabel(c0n)
            plt.ylabel(c1n)

            if show:
                plt.show()
            if out is not None:
                fname = os.path.join(out, titre)
                plt.savefig(fname+'.png')
                plt.savefig(fname+'.eps')
                plt.close()


def visualize_3d_multiple_surface_umap():
    import plotly.graph_objects as go
    for m in umap_accus:
        if len(umap_accus[m].shape) == 1:
            umap_accus[m] = umap_accus[m].reshape(umap_xs.shape)
    fig = go.Figure(data=[go.Surface(x=umap_min_dists_seqs, y=umap_neighbor_seqs, z=umap_accus[m], showscale=False)
                          for m in umap_accus])
    fig.update_layout({'title': 'UMAP', 'xaxis_title': 'min_dist', 'yaxis_title': 'neighbor'})
    fig.show()


def visualize_3d_multiple_surface(data, title='3D surface'):
    import plotly.graph_objects as go
    fig = go.Figure(
        data=[go.Surface(x=data['x']['data'], y=data['y']['data'], z=data['z']['data'], showscale=False)])
    fig.update_layout({'title': title, 'xaxis_title': data['x']['name'],
                       'yaxis_title': data['y']['name'],
                       'zaxis_title': data['z']['name']})
    fig.show()


def visualize_LD_multimodels(models, labels, dims, ND=3, show=True, out=None, label_alias=None,
                             vis_ignore=None, axis_alias=None, JNOTEBOOK_MODE=False):

    for m in models:
        model, X_LD = models[m]
        vfunc = visualize_3D if min(X_LD.shape[1], ND) == 3 else visualize_2D
        vfunc(X_LD, labels, dims, m, show=show, out=out, label_alias=label_alias,
              vis_ignore=vis_ignore, axis_alias=axis_alias,
              JNOTEBOOK_MODE=JNOTEBOOK_MODE)


def visualize_conn_matrix(mats, uniqLabels, affinity='euclidean', tag=None, cluster_param=3, label_alias=None,
                          confusion_mode=None, save=None):
    """
    cite: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    :param mats:
    :param uniqLabels:
    :param affinity:
    :param tag:
    :param cluster_param:
    :param confusion_mode: None if for other connectivity matrices, for confusion matrices: raw, normalize
    by true, pred, or total
    :return:
    """
    # HANDLE dissimality more graceful
    NM = len(list(mats.values())[0])
    lxs = np.arange(len(uniqLabels))
    # visualize confusing matrix, dissimilarity matrices and connectivity matrices
    fig, axes = plt.subplots(nrows=NM, ncols=len(mats), sharex=True, sharey=True,
                             figsize=(len(mats) * 7, NM * 5))
    for k, m in enumerate(mats):
        for l, met in enumerate(mats[m]):
            if NM * len(mats) == 1:
                ax = axes
            elif NM == 1:
                ax = axes[k]
            elif len(mats) == 1:
                ax = axes[l]
            else:
                ax = axes[l][k]
            # modify linkage
            # TODO: modify method to accommodate 2D rows
            mat2vis = mats[m][met]
            if confusion_mode is not None:
                if confusion_mode == 'true':
                    mat2vis = mat2vis / np.sum(mat2vis, keepdims=True, axis=1)
                elif confusion_mode == 'pred':
                    mat2vis = mat2vis / np.sum(mat2vis, keepdims=True, axis=0)
                elif confusion_mode == 'all':
                    mat2vis = mat2vis / np.sum(mat2vis)
            hsort_mat, hsort_labels, hsort = conn_matrix_hierarchical_sort(mat2vis, uniqLabels,
                                                                           affinity=affinity,
                                                                           cluster_param=cluster_param)
            if label_alias is not None:
                hsort_labels = [label_alias[hl] for hl in hsort_labels]
            clabels, ccounts = np.unique(hsort.labels_, return_counts=True)
            cluster_edges = np.cumsum(ccounts) - 0.5
            if affinity == 'precomputed':
                hsort_mat = 1 / (hsort_mat + 1e-12)

            if confusion_mode is not None:
                # For confusion matrix, visualize accuracy
                thresh = hsort_mat.max() / 1.5 if confusion_mode != 'raw' else hsort_mat.max() / 2
                for i in range(hsort_mat.shape[0]):
                    for j in range(hsort_mat.shape[1]):
                        if confusion_mode != 'raw':
                            fm = "{:0.2f}"
                        else:
                            fm = "{:,}"
                        ax.text(j, i, fm.format(hsort_mat[i, j]),
                                 horizontalalignment="center",
                                 color="white" if hsort_mat[i, j] > thresh else "black")
                fig.suptitle("Confusion Matrix For Classifiers")
            elif affinity == 'precomputed':
                thresh = hsort_mat.max() / 1.5 if confusion_mode != 'raw' else hsort_mat.max() / 2
                for i in range(hsort_mat.shape[0]):
                    if confusion_mode != 'raw':
                        fm = "{:0.2f}"
                    else:
                        fm = "{:,}"
                    ax.text(i, i, fm.format(hsort_mat[i, i]),
                            horizontalalignment="center",
                            color="white" if hsort_mat[i, i] > thresh else "black")
                fig.suptitle("Similarity Matrix Across classes")
            else:
                fig.suptitle("Raw Data Matrix Sorted With Dissimilarity")
            pmp = ax.imshow(hsort_mat, cmap=plt.cm.get_cmap('Blues'))
            ax.hlines(cluster_edges, -0.5, hsort_mat.shape[0] - 0.5, colors='k')
            ax.vlines(cluster_edges, -0.5, hsort_mat.shape[0] - 0.5, colors='k')
            ax.set_title(f"{m}_{met}" + f"_{tag}" if tag is not None else "")
            ax.set_xticks(lxs)
            ax.set_xticklabels(hsort_labels, rotation=45, horizontalalignment="right")
            ax.set_ylabel('truth')
            ax.set_yticks(lxs)
            ax.set_yticklabels(hsort_labels)
            ax.set_ylabel(f'{met}')
            plt.colorbar(pmp, ax=ax)
    if save is not None:
        fig.savefig(os.path.join(save, f"{tag}_conn_matrix.eps"))


def visualize_F_measure_multi_clf(mats, uniqLabels, label_alias=None):
    """
    cite: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    :param mats:
    :param uniqLabels:
    :param affinity:
    :param tag:
    :param cluster_param:
    :param confusion_mode: None if for other connectivity matrices, for confusion matrices: raw, normalize
    by true, pred, or total
    :return:
    """
    # HANDLE dissimality more graceful
    NM = len(list(mats.values())[0])
    # visualize confusing matrix, dissimilarity matrices and connectivity matrices
    all_fmeas = []
    if label_alias is not None:
        label_show = [label_alias[hl] for hl in uniqLabels]
    else:
        label_show = uniqLabels
    all_labels = np.tile(label_show, NM * len(mats))
    all_dims = []
    all_clfs = []
    for m in mats:
        all_dims.append((len(uniqLabels) * len(mats[m])) * [m])
        for met in mats[m]:
            all_clfs.append(len(uniqLabels) * [met])
            f_meas = class_wise_F_measure(mats[m][met])
            all_fmeas.append(f_meas)
    F_pdf = pd.DataFrame({'DimReduction': np.concatenate(all_dims), 'Classifier': np.concatenate(all_clfs),
                          'Group': all_labels, 'F-measure': np.concatenate(all_fmeas)})
    fig = px.bar(F_pdf, x="Group", y="F-measure", facet_row="Classifier", facet_col="DimReduction",
                 title=f"Class-wise F-measure For Classification (Chance: {1 / len(uniqLabels)})")
    fig.show()


def feature_clustering_visualize(X, keywords, nclusters, show=True):
    agglo = cluster.FeatureAgglomeration(n_clusters=nclusters)
    agglo.fit(X)
    ys = np.arange(keywords.shape[0])
    labels = agglo.labels_
    sorts = np.argsort(labels)
    plt.barh(ys, labels[sorts])
    plt.yticks(ys, keywords[sorts])
    plt.subplots_adjust(left=0.3)
    if show:
        plt.show()
    return agglo


#deprecated
def save_isomap_2D(iso, X_iso, plots, clustering=None):
    if not os.path.exists(plots):
        os.makedirs(plots)
    for i in range(X_iso.shape[1]-1):
        plt.figure(figsize=(15, 15))
        if clustering is not None:
            for l in np.unique(clustering.labels_):
                plt.scatter(X_iso[clustering.labels_ == l, i], X_iso[clustering.labels_ == l, i+1], label=l)
            plt.legend()
        else:
            plt.scatter(X_iso[:, i], X_iso[:, i+1])
        plt.xlabel(f'Comp {i+1}')
        plt.ylabel(f'Comp {i+2}')
        fname = os.path.join(plots, f'iso_2D{i+1}-{i+2}_loading' + '_cluster' if clustering else '')
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.eps')
        plt.close()


def silhouette_plot(X, labels, cluster_arg=None, ax=None):
    # Visualize Silhouette Plot
    # Cite: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    if ax is None:
        ax = plt.gca()

    n_clusters = len(np.unique(labels))
    n_unique_labels = np.sort(np.unique(labels))

    cpalette = sns.color_palette('Spectral', n_colors=n_clusters)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, labels)
    if cluster_arg is None:
        cluster_arg = f'n_clusters={n_clusters}'
    print(f'For {cluster_arg}, "The average silhouette_score is : {silhouette_avg}')
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == n_unique_labels[i]]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cpalette[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title(f"The silhouette plot for {cluster_arg}.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


""" ##########################################
################# Analysis ###################
########################################## """


# Basic
def feature_preprocess(tPDF, xPDF, method='skscale'):
    # TODO: implement more detailed feature preprocessing based on feature relationships
    # in fact tPDF does not need much unit variance scaling, but is implemented here for consistency
    keywordsT, keywordsX = tPDF.columns, xPDF.columns
    # Takes in demeaned data matrix (np.ndarray), and starts analysis, TODO: obtain X_cat_map, and X_cat labels
    if method == 'skscale':
        X_nonan_dm = preprocessing.scale(xPDF.values, axis=0)
        T_nonan_dm = preprocessing.scale(tPDF.values, axis=0)
    else:
        X_nonan_dm = xPDF.sub(xPDF.mean()).values
        T_nonan_dm = tPDF.sub(tPDF.mean()).values
    return T_nonan_dm, keywordsT, X_nonan_dm, keywordsX


def get_corr_matrix(T_nonan_dm, keywordsT, X_nonan_dm, keywordsX, dataset_name, dataOut,
                    JNOTEBOOK_MODE=False):
    if T_nonan_dm.shape[1] < X_nonan_dm.shape[1]:
        fLen = X_nonan_dm.shape[1]
        sLen = T_nonan_dm.shape[1]
        fDM, sDM = X_nonan_dm, T_nonan_dm
        fKWs, sKWs = keywordsX, keywordsT
    else:
        fLen = T_nonan_dm.shape[1]
        sLen = X_nonan_dm.shape[1]
        fDM, sDM = T_nonan_dm, X_nonan_dm
        fKWs, sKWs = keywordsT, keywordsX
    corrMat = np.zeros((fLen, sLen))
    for i in range(fLen):
        for j in range(sLen):
            corrMat[i, j] = np.corrcoef(fDM[:, i], sDM[:, j])[0, 1]
    plt.figure(figsize=(15, 15))
    plt.imshow(corrMat ** 2)
    plt.colorbar()
    plt.yticks(np.arange(len(fKWs)), fKWs)
    plt.xticks(np.arange(len(sKWs)), sKWs, rotation=90)
    plt.subplots_adjust(left=0.3, right=0.7, top=0.99, bottom=0.35)
    plt.show()
    corrPDF = pd.DataFrame(data=corrMat, index=fKWs, columns=sKWs)
    if JNOTEBOOK_MODE:
        corrPDF.head()
    else:
        corrPDF.to_csv(os.path.join(dataOut, dataset_name + 'corrMatrix.csv'), index=True)


# Dim Reduction
def dim_reduction(X_nonan_dm, ClusteringDim, models='all', params=None):
    """
    Performs dim reduction on data
    :param models: str or list-like
    kernelPCA_params: tuple (dimMUL, kernel)
        if kernelPCA is one of the models in `models`, kernelPCA_params must not be None
    :return: dict {model: (fit_model, transformed_data)} or just one tuple
    """
    # TODO: MDS, T-SNE
    DRs = {}
    EXTRACT = False
    if X_nonan_dm.shape[1] <= 3:
        DRs = {'raw': (None, X_nonan_dm)}
    else:
        if models=='all':
            models = ['PCA', 'ICA', 'ISOMAP', 'Kernel_PCA', 'MDS', 'NMDS', 'UMAP', 'tSNE']
        elif isinstance(models, str):
            models = [models]
            EXTRACT = False
        if params is None:
            params = {}
        # Maybe Use tSNE
        for m in models:
            DRModel = None
            if m == 'PCA':
                DRModel = PCA(n_components=ClusteringDim)
            elif m == 'ICA':
                DRModel = FastICA(n_components=ClusteringDim, random_state=RAND_STATE)
            elif m == 'ISOMAP':
                DRModel = Isomap(n_components=ClusteringDim)
            elif m == 'Kernel_PCA':
                if m in params:
                    dimMUL, kernel = params['Kernel_PCA']
                else:
                    dimMUL, kernel = 2, 'poly'
                DRModel = KernelPCA(n_components=ClusteringDim*dimMUL, kernel=kernel)
            elif m == 'MDS':
                if 'MDS' in params:
                    eps = params['MDS']
                else:
                    eps = 1e-9
                DRModel = MDS(n_components=ClusteringDim, max_iter=3000, eps=eps,
                          random_state=RAND_STATE, dissimilarity="euclidean")
            elif m == 'NMDS':
                if 'NMDS' in params:
                    eps = params['NMDS']
                else:
                    eps = 1e-12
                DRModel = MDS(n_components=ClusteringDim, metric=False, max_iter=3000, eps=eps,
                           dissimilarity="euclidean", random_state=RAND_STATE)
            elif m == 'UMAP':
                defaults = {'n_neighbors': 30, 'min_dist': 0.2, # 'n_neighbors': 10, 'min_dist': 0.8,
                            # TODO: formal testing
                            'n_components': 3, 'metric': 'euclidean'}
                if 'UMAP' in params:
                    defaults.update(params['UMAP'])
                DRModel = UMAP(**defaults)
            elif m == 'tSNE':
                defaults = {'perplexity': 30, 'init': 'pca',
                            'n_components': 2}
                if 'tSNE' in params:
                    defaults.update(params['tSNE'])
                DRModel = TSNE(**defaults)
            else:
                raise NotImplementedError(f'model {m} not implemented')
            DRs[m] = (DRModel, DRModel.fit_transform(X_nonan_dm))
    if EXTRACT:
        return list(DRs.values())[0]
    return DRs


def classifier_LD_multimodels(models, labels, LD_dim=None, N_iters=100, mode='true',
                              ignore_labels=None, clf_models='all', clf_params=None,
                              cluster_param=3, label_alias=None, show=True, save=None):
    # default return raw confusion matrix and visualize normalized by true labels
    if clf_models == 'all':
        clf_models = ['QDA', 'SVC', 'RandomForests', 'XGBoost'] # XGBoost
    elif isinstance(clf_models, str):
        clf_models = [clf_models]
    if clf_params is None:
        clf_params = {}
    clfs = {}
    confs = {}
    if ignore_labels is None:
        ignore_labels = []
    uniqLabels = np.setdiff1d(labels.unique(), ignore_labels)
    # Plot a score find best score ?
    for m in models:
        model, X_LD = models[m]
        if LD_dim is None:
            LD_dim = X_LD.shape[-1]
        X_LD = X_LD[:, :LD_dim]
        # TODO: add stratified k fold later
        neg_selectors = labels.isin(uniqLabels)
        X_LDF, labelsF = X_LD[neg_selectors], labels[neg_selectors]
        # TODO: Generalize to models other than QDA
        confs[m] = {}
        clfs[m] = {}
        for cm in clf_models:
            if cm == 'QDA':
                clfF = QuadraticDiscriminantAnalysis()
            elif cm == 'SVC':
                defaults = {"gamma": 2, "C": 1}
                if 'SVC' in clf_params:
                    defaults.update(clf_params['SVC'])
                clfF = SVC(kernel='rbf', **defaults)
                # TODO default SVC rbf
            elif cm == 'RandomForests':
                defaults = {"n_estimators": 50, "n_jobs": 4}
                if 'RandomForests' in clf_params:
                    defaults.update(clf_params['RandomForests'])
                clfF = RandomForestClassifier(**defaults)
            elif cm == 'XGBoost':
                defaults = {"learning_rate": 0.3, "gamma": 0, "reg_lambda": 1, # minimum loss reduction
                            "min_child_weight": 1, # larger, less likely to overfit, more conservative
                            "max_depth": 6, "n_jobs": 4}
                if 'XGBoost' in clf_params:
                    defaults.update(clf_params['XGBoost'])
                clfF = XGBClassifier(**defaults)
            else:
                raise NotImplementedError(f"Unknown Classifier {cm}")

            feat_sum, feat_square = 0, 0
            confmats = np.zeros((len(uniqLabels), len(uniqLabels)))
            for i in range(N_iters):  # TODO: replace with shuffle splits
                X_train, X_test, y_train, y_test = train_test_split(
                    X_LDF, labelsF, test_size=0.3, shuffle=True, stratify=labelsF)
                # QDA
                clfF.fit(X_train, y_train)
                preds = clfF.predict(X_test)
                conf_mat = confusion_matrix(y_test, preds, labels=uniqLabels)
                confmats += conf_mat
                if hasattr(clfF, 'feature_importances_'):
                    feat_imp = clfF.feature_importances_
                    feat_sum += feat_imp
                    feat_square += np.square(feat_imp)
            # TODO: maybe add confidence interval for confmats later
            confs[m][cm] = confmats / N_iters
            clfF.fit(X_LDF, labelsF)
            res = {'model': clfF}
            if isinstance(feat_sum, np.ndarray) and isinstance(feat_square, np.ndarray):
                mfs = feat_sum / N_iters
                sfs = feat_square / N_iters - mfs ** 2
                res["f_importance"] = (mfs, sfs)
            clfs[m][cm] = res # get clf for all data
    # TODO: Handle multiple models
    if show:
        visualize_conn_matrix(confs, uniqLabels, tag=f"normalize_{mode}", cluster_param=cluster_param,
                              confusion_mode=mode, label_alias=label_alias, save=save)
        visualize_F_measure_multi_clf(confs, uniqLabels, label_alias=label_alias)
    # TODO: return the labels
    return clfs, confs


def diff_evaluation_LD_multimodels(models, labels, LD_dim=None, method=('silhouette', 'average'),
                                   metric='euclidean', cluster_param=3, label_alias=None, ignore_labels=None,
                                   show=True, save=None):
    # TODO: make method tunable, not necessary so far because silhouette is way better
    mats = {}
    if ignore_labels is None:
        ignore_labels = []
    uniqLabels = np.setdiff1d(labels.unique(), ignore_labels)
    #uniqLabels = labels.unique()
    if isinstance(method, str):
        method = [method]
    # Plot a score find best score ?
    for m in models:
        if m == 'tSNE':
            continue
        model, X_LD = models[m]
        if LD_dim is None:
            LD_dim = X_LD.shape[-1]
        else:
            LD_dim = min(LD_dim, X_LD.shape[-1])
        X_LD = X_LD[:, :LD_dim]
        # TODO: fix unique labels
        mats[m] = {}
        for met in method:
            mat, ulabels = calculate_class_dissimilarity(X_LD, labels=labels, categories=uniqLabels,
                                                         method=met, metric=metric)
            # Get similarity as inverse of dissimilarity
            mats[m][met] = mat

    # TODO: Handle multiple models
    if len(mats) != 0:
        # TODO: more robust handling of TSNE
        if show:
            visualize_conn_matrix(mats, uniqLabels, affinity='precomputed', tag=f"{metric}_similarity",
                                  cluster_param=cluster_param, label_alias=label_alias, save=save)
    return mats


def calculate_class_dissimilarity(X, Y=None, labels=None, intraDistPair=None, categories=None,
                                  method='silhouette', metric='euclidean'):
    # TODO handle singletons
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        if categories is not None:
            uniq_labels = categories
        else:
            uniq_labels = np.unique(labels)
        N = uniq_labels.shape[0]
        dissim = np.empty((N, N), dtype=np.float)
        groups = [X[labels==iul, :] for iul in uniq_labels]
        if method == 'silhouette':
            intraDists = [np.sum(np.triu(
                pairwise_distances(
                    iX, metric=metric), 1)) * 2 / (iX.shape[0] ** 2 - iX.shape[0]) for iX in groups]
        else:
            intraDists = [None] * N
        for i, iX in enumerate(groups):
            for j, jY in enumerate(groups):
                dissim[i, j] = calculate_class_dissimilarity(iX, Y=jY,
                                                             intraDistPair=(intraDists[i], intraDists[j]),
                                                             method=method, metric=metric)
        return dissim, uniq_labels
    elif Y is not None:
        dist_XY = pairwise_distances(X, Y, metric=metric)
        if method == 'average':
            return np.mean(dist_XY)
        elif method == 'max':
            return np.max(dist_XY)
        elif method == 'min':
            return np.min(dist_XY)
        elif method == 'silhouette':
            distX, distY = intraDistPair
            dia = np.sum(np.diag(dist_XY))
            if dia == 0.:
                amean = (np.sum(dist_XY) - np.sum(dia)) / (dist_XY.shape[0] ** 2 - dist_XY.shape[0])
                return amean / (distX + distY)
            else:
                return np.mean(dist_XY) / (distX + distY)
        elif method == 'avgmin':
            # TODO: normalized version
            x_y = np.min(dist_XY, axis=1)
            y_x = np.min(dist_XY, axis=0)
            return np.mean(np.concatenate((x_y, y_x)))
        else:
            raise NotImplementedError(f'Unknown Dissimilarity Method {method}')
        # average linkage
        # max linkage
        # min
        # Silhouette dij / (sig_i + sig_j)
    else:
        raise RuntimeError('At least one of Y and lables must be not None')


def clustering_LD_multimodels(models, labels, ClusteringDim=None, show=True, out=None):
    # TODO: finish clutsering method
    clusterings = {}
    NClusters = len(labels.unique())

    for m in models:
        model, X_LD = models[m]
        if ClusteringDim is None:
            ClusteringDim = X_LD.shape[-1]
        clustering[m] = {}
        # Default: Spectral
        X_LD = X_LD[:, :ClusteringDim]
        clustering = SpectralClustering(n_clusters=NClusters,
                                        assign_labels="discretize",
                                        random_state=RAND_STATE).fit(X_LD)
        clustering[m]['spectral'] = clustering


def conn_matrix_hierarchical_sort(mat, labels, affinity='euclidean', method='ward', cluster_param=3):
    # TODO: fine tune sorting method number of clusters
    if affinity != 'euclidean' and method == 'ward':
        method = 'single'
    if isinstance(cluster_param, int):
        aggs = cluster.AgglomerativeClustering(n_clusters=cluster_param, affinity=affinity,
                                               linkage=method).fit(mat)
    else:
        aggs = cluster.AgglomerativeClustering(n_clusters=None, affinity=affinity, linkage=method,
                                               distance_threshold=cluster_param).fit(mat)
    neworder = np.argsort(aggs.labels_)
    mat_sorted = mat[:, neworder][neworder, :]
    return mat_sorted, labels[neworder], aggs


# Regression
def regression_multi_models(models, Y, method='linear', N_iters=100, raw_features_names=None, reg_params=None,
                            feature_importance=True, confidence_level=0.95, show=True):
    # implement feature importance
    # dataset_name, dataOut
    # TODO: different dim models for regression
    metrics = ['R2', 'MAEp', 'MSEp']
    if len(Y.shape) == 1:
        Y = pd.DataFrame({Y.name: Y})
    if 'raw' in models:
        assert raw_features_names is not None
    # TODO: implement method supporting multidim Y

    if reg_params is None:
        reg_params = {}

    if method == 'all':
        method = ['linear', 'DT', 'SVR_poly', 'SVR_linear', 'SVR_rbf', 'SVR_sigmoid', 'XGBoost',
                  'RandomForests']
    elif isinstance(method, str):
        method = [method]

    if feature_importance and ('DT' not in method):
        method.append('DT')

    reg_results = {}
    y_inds = Y.columns
    y_inds_double = np.tile(y_inds, 2)
    reg_pdf = {'model': [], 'method': [], 'score_type': [], 'label': []}
    reg_pdf.update({imet: [] for imet in metrics})
    D_feats = len(y_inds)

    for k in models:
        reg_results[k] = {}
        for m in method:
            model, X_LD = models[k]
            reg_results[k][m] = {f'train_{imet}': 0. for imet in metrics}
            reg_results[k][m].update({f'test_{imet}': 0. for imet in metrics})
            feat_sum, feat_square = 0., 0.
            for _ in range(N_iters):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_LD, Y.values, test_size=0.3)  # TODO: do multiple iterations
                #print(y_train.shape, y_test.shape)
                if m == 'linear':
                    # Linear Regression
                    reg = LinearRegression().fit(X_train, y_train)
                elif m == 'DT':
                    # Decision Tree Regression
                    reg = DecisionTreeRegressor(random_state=RAND_STATE).fit(X_train, y_train)
                # TODO: ADD adaboost regressor
                elif re.search(r'SVR_(\w+)', m):
                    kernelType = m.split('_')[1] #['linear', 'poly', 'rbf', 'sigmoid']
                    reg = SVR(C=1, epsilon=0.5, kernel=kernelType).fit(X_train, y_train)

                elif m == 'RandomForests':
                    defaults = {"n_estimators": 50, "n_jobs": 4}
                    if 'RandomForests' in reg_params:
                        defaults.update(reg_params['RandomForests'])
                    reg = RandomForestRegressor(**defaults).fit(X_train, y_train)
                elif m == 'XGBoost':
                    defaults = {"learning_rate": 0.3, "gamma": 0, "reg_lambda": 1,  # minimum loss reduction
                                "min_child_weight": 1,  # larger, less likely to overfit, more conservative
                                "max_depth": 6, "n_jobs": 4}
                    if 'XGBoost' in reg_params:
                        defaults.update(reg_params['XGBoost'])
                    reg = XGBRegressor(**defaults).fit(X_train, y_train)
                else:
                    raise NotImplementedError(f"Unknown Regression Method {m}")
                if feature_importance and hasattr(reg, 'feature_importances_'):
                    feat_imp = reg.feature_importances_
                    feat_sum += feat_imp
                    feat_square += np.square(feat_imp)
                for imet in metrics:
                    mtrains, mtests = test_regression_score(reg, X_train, y_train, X_test, y_test, imet)
                    reg_results[k][m][f"train_{imet}"] += mtrains
                    reg_results[k][m][f"test_{imet}"] += mtests
            if isinstance(feat_sum, np.ndarray) and isinstance(feat_square, np.ndarray):
                feat_mean = feat_sum / N_iters
                feat_std = np.sqrt(feat_square / N_iters - feat_mean ** 2)
                reg_results[k][m]['f_importance'] = (feat_mean, feat_std)

            for imet in metrics:
                itrain = reg_results[k][m][f"train_{imet}"]
                itest = reg_results[k][m][f"test_{imet}"]
                reg_results[k][m][f"train_{imet}"] = itrain / N_iters
                reg_results[k][m][f"test_{imet}"] = itest / N_iters
                reg_pdf[imet].append(np.concatenate((itrain, itest)))
            # TODO: save regression models
            reg_pdf['model'].append([k] * 2 * D_feats)
            reg_pdf['method'].append([m] * 2 * D_feats)
            reg_pdf['score_type'].append(['train'] * D_feats + ['test'] * D_feats)
            reg_pdf['label'].append(y_inds_double)
    for rp in reg_pdf:
        reg_pdf[rp] = np.concatenate(reg_pdf[rp])

    if show:
        # fig, axes = plt.subplots(nrows=len(method), ncols=len(models), sharey=True, sharex=True)
        # for i, met in enumerate(method):
        #     for j, mod in enumerate(models):
        for imet in metrics:
            fig = px.bar(reg_pdf, x="label", y=imet, color="score_type", barmode="group",
                         facet_row="method", facet_col="model")
            fig.show()
    return reg_results


def regression_with_norm_input(tPDF, xPDF, dataset_name, dataOut, method='linear', feature_importance=True):
    # TODO: use grid search to build a great regression model
    # Deprecated
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)
    X_train, X_test, y_train, y_test = train_test_split(
        T_nonan_dm, X_nonan_dm, test_size=0.3) # TODO: do multiple iterations
    if method == 'all':
        method = ['linear', 'DT', 'SVR_poly', 'SVR_linear', 'SVR_rbf', 'SVR_sigmoid']
    elif isinstance(method, str):
        method = [method]

    if feature_importance and ('DT' not in method):
        method.append('DT')

    for m in method:
        if m == 'linear':
            # Linear Regression
            reg = LinearRegression().fit(X_train, y_train)
            composite_linreg = reg.score(X_test, y_test)
            train_accus, test_accus = test_model_efficacy(reg, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus},
                                  index=keywordsX)
        elif m == 'DT':
            # Decision Tree Regression
            regressor = DecisionTreeRegressor(random_state=RAND_STATE)
            regressor.fit(X_train, y_train)
            composite_treereg = regressor.score(X_test, y_test)
            plt.bar(np.arange(len(keywordsT)), regressor.feature_importances_)
            plt.xticks(np.arange(len(keywordsT)), keywordsT, rotation=90)
            plt.subplots_adjust(left=0.3, right=0.7, top=0.99, bottom=0.35)
            plt.show()
            train_accu, test_accus = test_model_efficacy(regressor, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_accuracy': train_accus, 'test_accuracy': test_accus},
                                  index=keywordsX)
        elif re.search(r'SVR_(\w+)', m):
            kernelType = m.split('_')[1] #['linear', 'poly', 'rbf', 'sigmoid']
            clf = SVR(C=1, epsilon=0.5, kernel=kernelType)
            train_errors, test_errors = test_model_efficacy(clf, X_train, y_train, X_test, y_test)
            errPDF = pd.DataFrame({'train_err': train_errors, 'test_err': test_errors}, index=keywordsX)
        errPDF.to_csv(os.path.join(dataOut, dataset_name + f'_{m}_regressor_err.csv'), index=True)


# Model Testing
def test_model_efficacy(model, X_train, y_train, X_test, y_test):
    train_accus, test_accus = np.empty(y_train.shape[1]), np.empty(y_train.shape[1])
    for i in range(y_train.shape[1]):
        # model.fit(X_train, y_train[:, i])
        train_accu = model.score(X_train, y_train[:, i])
        test_accu = model.score(X_test, y_test[:,i])
        train_accus[i] = train_accu
        test_accus[i] = test_accu
    return train_accus, test_accus


def test_regression_score(reg, X_train, y_train, X_test, y_test, method='MAEp'):
    if method == 'MAEp':
        devs = y_train - np.mean(y_train, keepdims=True, axis=0)
        trains = mean_absolute_error(y_train, reg.predict(X_train)) / np.mean(np.abs(devs), axis=0)
        tests = mean_absolute_error(y_test, reg.predict(X_test)) / np.mean(np.abs(devs), axis=0)
    elif method == 'MSEp':
        devs = y_train - np.mean(y_train, keepdims=True, axis=0)
        trains = mean_squared_error(y_train, reg.predict(X_train)) / np.mean(np.square(devs), axis=0)
        tests = mean_squared_error(y_test, reg.predict(X_test)) / np.mean(np.square(devs), axis=0)
    elif method == 'R2':
        trains, tests = test_model_efficacy(reg, X_train, y_train, X_test, y_test)
    else:
        raise NotImplementedError(f"Unknown Method {method}")
    return trains, tests


def adjusted_BAC_confusion_matrix(conf):
    norm_conf = np.diag(conf) / np.sum(conf, axis=1)
    rlevel = 1 / conf.shape[0]
    return (np.mean(norm_conf) - rlevel) / (1 - rlevel)


def class_wise_F_measure(conf):
    tps = np.diag(conf)
    prec = tps / np.sum(conf, axis=0)
    recall = tps / np.sum(conf, axis=1)
    return 2 * prec * recall / (prec + recall)


# Model Selection
def get_accu(models_behaviors, LDLabels, test_p, CLUSTER_PARAM=4, LDDIM=3):
    _, ignores, _ = get_test_specific_options(test_p, BLIND=False)
    clfs, confs = classifier_LD_multimodels(models_behaviors, LDLabels, LD_dim=LDDIM, mode='true',
                                            ignore_labels=ignores, cluster_param=CLUSTER_PARAM,
                                            show=False)  # ['1.0', '0.0', '-1.0']
    k = diff_evaluation_LD_multimodels(models_behaviors, LDLabels, cluster_param=CLUSTER_PARAM, show=False)
    accu_dict = {}
    for m in confs:
        accus = np.diag(confs[m]['QDA'])
        accu_dict[m] = {'avg': np.mean(accus),
                        'max': np.max(accus),
                        'top_half': np.mean(accus[accus > np.percentile(accus, 50)])}

    return accu_dict


def extract_pdf(accus, xs, xtag, ys=None, ytag=None):
    if ys is not None:
        ys = np.ravel(ys, order='C')
        xs = np.ravel(xs, order='C')
        for m in accus:
            accus[m] = np.ravel(accus[m], order='C')
        all_ys = []
    labels = []
    all_accus = []
    all_xs = []
    for m in accus:
        labels.append(np.full(len(xs), m))
        all_accus.append(accus[m])
        all_xs.append(xs)
        if ys is not None:
            all_ys.append(ys)
    labels = np.concatenate(labels)
    all_accus = np.concatenate(all_accus)
    all_xs = np.concatenate(all_xs)
    if ys is not None:
        pdf = pd.DataFrame({xtag: all_xs, ytag: np.concatenate(all_ys), 'TPs': all_accus, 'measure': labels})
    else:
        pdf = pd.DataFrame({xtag: all_xs, 'TPs': all_accus, 'measure': labels})
    return pdf


def run_procedures():
    # TOP half
    # maximum
    # average
    CLUSTER_PARAM = 4
    LDDIM = 3
    umap_neighbor_seqs = np.array([10 * i for i in range(1, 9)])
    umap_min_dists_seqs = np.array([0.1 * i for i in range(1, 9)])
    tSNE_perplexity = np.array([10 * i for i in range(1, 9)])
    umap_accus = {m: np.zeros((len(umap_neighbor_seqs), len(umap_min_dists_seqs))) for m in
                  ['avg', 'max', 'top_half']}
    umap_xs, umap_ys = np.meshgrid(umap_neighbor_seqs, umap_min_dists_seqs)
    tSNE_accus = {m: np.zeros(len(tSNE_perplexity)) for m in ['avg', 'max', 'top_half']}
    for ij, jseq in enumerate(umap_neighbor_seqs):
        DIM_PARAMS_tSNE = {'tSNE': {'perplexity': 40, 'init': 'pca',
                                    'n_components': 2}}
        models_behaviors, LDLabels, BXpdf, BTpdf = behavior_analysis_pipeline(ROOT,
                                                                              dataRoot, test_p,
                                                                              behavior=behavior_p,
                                                                              dim_models=['tSNE'],
                                                                              ND=plot_dimension_p,
                                                                              NAN_POLICY=NAN_POLICY,
                                                                              BLIND=False,
                                                                              dim_params=DIM_PARAMS,
                                                                              cluster_param=CLUSTER_PARAM,
                                                                              show=False,
                                                                              JNOTEBOOK_MODE=True)
        accu_dict_tSNE = \
        get_accu(models_behaviors, LDLabels, test_p, CLUSTER_PARAM=CLUSTER_PARAM, LDDIM=LDDIM)['tSNE']
        for m in tSNE_accus:
            tSNE_accus[m][ij] = accu_dict_tSNE[m]
        for ik, kseq in enumerate(umap_min_dists_seqs):
            DIM_PARAMS = {'UMAP': {'n_neighbors': jseq, 'min_dist': kseq,
                                   'n_components': 3, 'metric': 'euclidean'}}
            models_behaviors, LDLabels, BXpdf, BTpdf = behavior_analysis_pipeline(ROOT,
                                                                                  dataRoot, test_p,
                                                                                  behavior=behavior_p,
                                                                                  dim_models=['UMAP'],
                                                                                  ND=plot_dimension_p,
                                                                                  NAN_POLICY=NAN_POLICY,
                                                                                  BLIND=False,
                                                                                  dim_params=DIM_PARAMS,
                                                                                  cluster_param=CLUSTER_PARAM,
                                                                                  show=False,
                                                                                  JNOTEBOOK_MODE=True)
            accu_dict_UMAP = \
            get_accu(models_behaviors, LDLabels, test_p, CLUSTER_PARAM=CLUSTER_PARAM, LDDIM=LDDIM)['UMAP']
            for m in umap_accus:
                umap_accus[m][ij][ik] = accu_dict_UMAP[m]

    px.line_3d(pd.DataFrame(extract_pdf(umap_accus, umap_xs, 'umap_neighbor', umap_ys, 'umap_min_dist')),
               x='umap_neighbor', y='umap_min_dist', z='TPs', color='measure', title='UMAP')
    px.line(pd.DataFrame(extract_pdf(tSNE_accus, tSNE_perplexity, 'tSNE_perplexity')), x='tSNE_perplexity',
            y='TPs', color='measure', title='tSNE')
    accuv = umap_accus['avg']
    xmax, ymax = np.where(accuv == np.max(accuv))
    print(umap_neighbor_seqs[xmax], umap_min_dists_seqs[ymax])
    print(np.max(accuv), accuv[xmax, ymax])


""" ##########################################
########### DATA-SPECIFIC MODULE #############
########################################## """


def get_test_specific_options(test, BLIND=False, ignore_others=True, ignore_labels=None):
    """ Function to use for determining test specific options, currently this is tuned to 4CR dataset;
    Need to be replaced for specific dataset.
    """
    # Need to change when the total test options changed
    # Color key Experiment 1: (Groups 5-8 are unusually flexible; group 1,6-8 are different strains than 2-5)
    exp1 = { #"0.0": r'Controls',
            "1.0": r"Controls WT/SAL male P60-90 Bl/6J/CR",
            "2.0": r"FI male P60 Taconic",
            "3.0": r"FR male P60 Taconic",
            "4.0": r"ALG male P60 Taconic",
            "5.0": r"ALS male P60 Taconic",
            "6.0": r"5d COC test at P90 Bl/6CR",
            "7.0": r"BDNF met/met Ron tested at P60",
            "8.0": r"P26 males WT Bl/6CR"}
    # Color Key Experiment 2 data (focusing on angel's mice and bdnf/trkb manipulations) P40-60 ages
    exp2 = {"1.0": r"Controls VEH/SAL/WT",
            "2.0": r"acute NMPP1pump",
            "3.0": r"chronic NMPP1pump",
            "4.0": r"BDNF Val/Val Ron",
            "5.0": r"P1-23 NMPP1H20",
            "6.0": r"P1-40 NMPP1H20",
            "7.0": r"BDNF Met/Met Ron"}
    if not ignore_others:
        exp1["-1.0"] = r"OTHERS"
        exp2["-1.0"] = r"OTHERS"

    exp1_params = {'UMAP': {'n_neighbors': 10,
                              'min_dist': 0.8,
                              'n_components': 3,
                              'metric': 'euclidean'}}
    exp2_params = {}

    TEST_LABEL_ALIAS = {
        'exp1_label_FI_AL_M': None if BLIND else exp1,
        'exp2_Angel': None if BLIND else exp2,
        'age': None,
        'RL_age': None,
        'RL_treat_sex': None,
        'RL_treat': None,
        'RL_sex': None
    }

    IGNORE_LABELS = {
        'exp1_label_FI_AL_M': ['-1.0', '0.0', '1.0', '3.0', '4.0', '6.0', '7.0'],
        'exp2_Angel': ['-1.0', '1.0'],
        'age': None,
        'RL_age': None,
        'RL_treat_sex': ['ALS_F', 'FI_F', 'FR_F'],
        'RL_treat': None,
        'RL_sex': None
    }

    DIM_PARAMS = {
        'exp1_label_FI_AL_M': exp1_params,
        'exp2_Angel': exp2_params,
        'age': {},
        'RL_age': {},
        'RL_treat_sex': {},
        'RL_treat': {},
        'RL_sex': {}
    }

    return TEST_LABEL_ALIAS[test], IGNORE_LABELS[test] if ignore_labels is None else ignore_labels,\
           DIM_PARAMS[test]


def behavior_analysis_pipeline(ROOT, dataRoot, test, behavior='both', dim_models='all', reg_models='linear',
                               clf_models='all', reg_feature_models=None, ND=3, LD_dim=3, NAN_POLICY='drop',
                               BLIND=False, normalize_mode='true', dim_params=None, clf_params=None,
                               cluster_param=4, ignore_others=True, vis_ignore=None, show=True, save=True,
                               JNOTEBOOK_MODE=True):
    # TODO: GET RAW MODEL
    """
    :param ROOT: This is the root folder
    :param dataRoot: folder to find behavior data file
    :param NAN_POLICY: How to handle/encode NAN data
    :param test: Input to get_test_specific_options function, test option
    :param BLIND: Flag to indicate whether explicit labels or numerical labels (data blind visualization)
           show up
    :param ignore_others: Flag whether NULL group for label encoding (dataset specific)
    :param show: Switch for visualization
    :param JNOTEBOOK_MODE: Switch for analysis mode, True if using jupyter notebook like platform
    :param behavior: tag to distinguish whether to use only REV, SD task data or both
    :param dim_models: list to specify what dimensionality reduction model to use, 'all' for using all models
           implemented
    :param dim_params:
    :param LD_dim: number of dimensions to use for postprocessing (classification e.g.)
    :param reg_models: specify what regression model to use, 'all' for using all models
           implemented
    :param clf_models: specify what classification model to use, 'all' for using all models
           implemented
    :param clf_params: parameters for classification
    :param reg_feature_models: specify what features to use for regression (by default raw, PCA, ISOMAP,
    if available)
    :param ND: number of dimensions to use for visualization (2 or 3)
    :param normalize_mode: method to normalize confusion matrix for classification
    :param cluster_param: number of clusters to find in the sorting of confusion matrix

    :return:
    """
    # SPECIFICATION (TODO: DATA SPECIFIC)
    if test.startswith('RL'):
        dataset_name = 'FIP_RL'
    else:
        dataset_name = '4CR'  # 'MAS'
    dataOut = os.path.join(dataRoot, dataset_name)
    plots = os.path.join(ROOT, 'plots', dataset_name, test)

    if not os.path.exists(plots):
        os.makedirs(plots)

    if not os.path.exists(dataOut):
        os.makedirs(dataOut)
    pdf = load_session(dataRoot, dataset_name)

    # test speciifc option (TODO: DATA Specific)
    test_label_alias, ignore_labels, dim_default_params = get_test_specific_options(test, BLIND=BLIND,
                                                                                    ignore_others=ignore_others,
                                                                                    ignore_labels=vis_ignore)
    # preprocessing (TODO: DATA SPECIFIC)
    RANGES = {'FIP': 14,  # Behavior only
              'MAS': list(range(6)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              'MAS2': list(range(8)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              'FIP_RL': 15,
              '4CR': 8}  # fix for all
    vectorize_func = {
        'FIP': FIP_data_vectorizer,
        'FIP_RL': FIP_data_vectorizer,
        'MAS': MAS_data_vectorizer,
        'MAS2': MAS_data_vectorizer,
        '4CR': FOURCR_data_vectorizer
    }

    # -----------------------------------------

    # FEATURE CODING
    # todo: set by slice problem did not show up with step by step run
    (tPDF, tLabels), (xPDF, xLabels) = vectorize_func[dataset_name](pdf, RANGES[dataset_name], NAN_POLICY)
    tLEN, xLEN = tPDF.shape[1], xPDF.shape[1]
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)

    # DATA SELECTION: (TODO: DATA SPECIFIC)
    feature_classes_X = dataset_split_feature_classes(xPDF)
    if behavior == 'SD':
        SDX = feature_classes_X['SD']
        _, _, SDX_nonan_dm, SDkeywordsX = feature_preprocess(tPDF, SDX)
        BXpdf = SDX
        BX_nonan_dm, BkeywordsX = SDX_nonan_dm, SDkeywordsX
    elif behavior == 'REV':
        REVX = feature_classes_X['REV']
        _, _, REVX_nonan_dm, REVkeywordsX = feature_preprocess(tPDF, REVX)
        BXpdf = REVX
        BX_nonan_dm, BkeywordsX = REVX_nonan_dm, REVkeywordsX
    else:
        BXpdf = xPDF
        BX_nonan_dm, BkeywordsX = X_nonan_dm, keywordsX
    # TODO: add BT_nonan_dm

    if BX_nonan_dm.shape[1] <= 3:
        axis_alias = BkeywordsX
    else:
        axis_alias = None

    STRIFY = True
    if test[:3] == 'exp':
        LABEL = test  # 'exp1_label_FI_AL_M', 'exp2_Angel'
          # Set this as true if we want the color of the points to be discrete (for discrete classes of vars)
        LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=STRIFY)
        if ignore_others:
            selectors = LDLabels.astype(np.float) != -1
            BX_nonan_dm = BX_nonan_dm[selectors]
            LDLabels = LDLabels[selectors]
            BTpdf = tPDF[selectors]
        else:
            BTpdf = tPDF
    elif test == 'age':
        exp1_labels = get_data_label(tPDF, tLabels, 'exp1_label_FI_AL_M', STRIFY=False)
        angel_labels = get_data_label(tPDF, tLabels, 'exp2_Angel', STRIFY=False)
        LABEL = 'Age At Test'
        LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=False)
        #LDLabels = ((LDLabels // 10) * 10).astype(str)
        # FILTER OUT Treatment Groups
        selectors = (angel_labels.astype(np.float) == -1) & (exp1_labels.astype(np.float) == -1)
        BX_nonan_dm = BX_nonan_dm[selectors]
        LDLabels = LDLabels[selectors]
        BTpdf = tPDF[selectors]
    elif test.startswith('RL'):
        tag = test[3:]
        if tag == 'age':
            LABEL = f"Age At Test"
            LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=False)
        else:
            label_map = {'treat_sex': f"exp_{tag}", 'treat': 'Treat', 'sex': 'Sex'}
            LABEL = label_map[tag]
            LDLabels = get_data_label(tPDF, tLabels, LABEL, STRIFY=STRIFY)
        BTpdf = tPDF
    else:
        raise NotImplementedError(f'Unknown test: {test}')

    # DIM Reduction
    if dim_params is None:
        dim_params = dim_default_params
    pca_full, ClusteringDim = dataset_dimensionality_probing(BX_nonan_dm, dataset_name, visualize=show,
                                                             JNOTEBOOK_MODE=JNOTEBOOK_MODE)
    models_behaviors = dim_reduction(BX_nonan_dm, ClusteringDim, models=dim_models, params=dim_params)
    visualize_LD_multimodels(models_behaviors, LDLabels, 0, ND=ND, show=show,
                             label_alias=test_label_alias, vis_ignore=ignore_labels,
                             axis_alias=axis_alias, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # Quantify Difference
    # Train Classifiers on X_LDs and quantify cross validation area
    if 'age' in test:
        if reg_feature_models is None:
            reg_FModels = [m for m in ['ISOMAP', 'PCA'] if m in models_behaviors]
        else:
            reg_FModels = reg_feature_models
        models4regression = {m: models_behaviors[m] for m in reg_FModels}
        if 'raw' not in models_behaviors:
            models4regression.update({'raw': (None, BX_nonan_dm)})
        LDLabels = LDLabels.astype(np.float)
        reg_results = regression_multi_models(models4regression, LDLabels, method=reg_models, N_iters=100,
                                              raw_features_names=BkeywordsX,
                                    feature_importance=True, confidence_level=0.95, show=True)
    else:
        clfs, confs = classifier_LD_multimodels(models_behaviors, LDLabels, LD_dim=LD_dim,
                                                mode=normalize_mode, ignore_labels=ignore_labels,
                                                clf_models=clf_models, clf_params=clf_params,
                                                cluster_param=cluster_param, label_alias=test_label_alias,
                                                show=show, save=plots if save else None)
        dissim_mats = diff_evaluation_LD_multimodels(models_behaviors, LDLabels, ignore_labels=ignore_labels,
                                                     cluster_param=cluster_param,
                                                     label_alias=test_label_alias, show=show,
                                                     save=plots if save else None)

    return models_behaviors, LDLabels, BXpdf, BTpdf # TODO: Returns tLabels also depending on utility


""" ##########################################
#################### MAIN ####################
########################################## """


def main():
    # TODO: add function to handle merged datasets
    # paths
    JNOTEBOOK_MODE = False
    CAT_ENCODE = None
    NAN_POLICY = 'drop'
    CLUSTER_PARAM = 4
    IGNORE_OTHERS = True # Change this to show "Others" in clustering plots
    DIM_MODELS = ['PCA', 'ISOMAP', 'UMAP'] #['PCA', 'ISOMAP', 'UMAP', 'tSNE']
    CLF_MODELS = ['QDA'] #['QDA', 'SVC', 'RandomForests', 'XGBoost']
    REG_MODELS = ['linear', 'RandomForests']
    TEST_OPTIONS = ['exp1_label_FI_AL_M', 'exp2_Angel', 'age', 'RL_treat_sex', 'RL_treat', 'RL_sex', 'RL_age']
    BEHAVIOR_OPTS = ['both', 'SD', 'REV']
    if JNOTEBOOK_MODE:
        ROOT = "/content/drive/My Drive/WilbrechtLab/adversity_4CR/"
    else:
        ROOT = "/Users/albertqu/Documents/7.Research/Wilbrecht_Lab/PCA_adversity/"
    dataRoot = os.path.join(ROOT, 'data')

    test_p, behavior_p, plot_dimension_p = 'exp1_label_FI_AL_M', 'both', 3
    dim_params = {'UMAP': {'n_neighbors': 10,
                              'min_dist': 0.8,
                              'n_components': 3,
                              'metric': 'euclidean'}},
    models_behaviors, LDLabels, BXpdf, BTpdf = behavior_analysis_pipeline(ROOT, dataRoot, test_p,
                                                                          behavior=behavior_p,
                                                                          dim_models=DIM_MODELS,
                                                                          reg_models=REG_MODELS,
                                                                          clf_models=CLF_MODELS,
                                                                          reg_feature_models=None,
                                                                          ND=plot_dimension_p, LD_dim=3,
                                                                          NAN_POLICY=NAN_POLICY, BLIND=False,
                                                                          cluster_param=CLUSTER_PARAM,
                                                                          ignore_others=IGNORE_OTHERS,
                                                                          JNOTEBOOK_MODE=True)

    dataset_name = '4CR' # 'MAS'
    dataOut = os.path.join(dataRoot, dataset_name)
    plots = os.path.join(ROOT, 'plots', dataset_name, test_p)

    if not os.path.exists(plots):
        os.makedirs(plots)

    if not os.path.exists(dataOut):
        os.makedirs(dataOut)
    pdf = load_session(dataRoot, dataset_name)

    # preprocessing
    RANGES = {'FIP': 14,  # Behavior only
              'MAS': list(range(6)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              'MAS2': list(range(8)) + list(range(pdf.shape[1] - 8, pdf.shape[1])),
              '4CR': 8}  # fix for all
    vectorize_func = {
        'FIP': FIP_data_vectorizer,
        'MAS': MAS_data_vectorizer,
        'MAS2': MAS_data_vectorizer,
        '4CR': FOURCR_data_vectorizer
    }

    # todo: set by slice problem did not show up with step by step run
    (tPDF, tLabels), (xPDF, xLabels) = vectorize_func[dataset_name](pdf, RANGES[dataset_name], NAN_POLICY)
    tLEN, xLEN = tPDF.shape[1], xPDF.shape[1]
    T_nonan_dm, keywordsT, X_nonan_dm, keywordsX = feature_preprocess(tPDF, xPDF)
    pca_full, ClusteringDim = dataset_dimensionality_probing(X_nonan_dm, dataset_name,
                                                             JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # Correlation Matrix
    get_corr_matrix(T_nonan_dm, keywordsT, X_nonan_dm, keywordsX, dataset_name,
                    dataOut, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # DATA labeling
    # MAS: ['Genotype', 'Treat']
    # df['Name'] = df['First'].str.cat(df['Last'],sep=" ")
    LABEL = 'Age At Test'  # 'exp1_label_FI_AL_M', 'exp2_Angel'
    STRIFY = True
    LDLabels=get_data_label(tPDF, tLabels, LABEL, STRIFY=STRIFY)

    # DIM Reduction
    models = dim_reduction(X_nonan_dm, ClusteringDim, params={'Kernel_PCA': (2, 'poly')})
    visualize_LD_multimodels(models, LDLabels, 0, ND=3, show=True, JNOTEBOOK_MODE=JNOTEBOOK_MODE)

    # ------------------------------- LOOP -------------------------------
    # TODO: fix loading methods
    for l in tLabels.columns:
        LDLabels = tLabels[l]
        # PCA
        save_loading_plots(models['PCA'][0], keywordsX, 'PCA', plots)
        save_LD_plots(models['PCA'][1], LDLabels, 'PCA', plots, show=False)

        # ICA
        fastICA = FastICA(n_components=ClusteringDim, random_state=RAND_STATE)
        X_ICAHD = fastICA.fit_transform(X_nonan_dm)
        save_loading_plots(fastICA, keywordsX, 'ICA', plots)
        save_LD_plots(X_ICAHD, LDLabels, 'ICA', plots, show=False)

        # ISOMAP
        embedding = Isomap(n_components=ClusteringDim)
        X_isomap = embedding.fit_transform(X_nonan_dm)
        save_LD_plots(X_isomap, LDLabels, 'ISOMAP', plots, show=False)

        # Kernel PCA
        kernelPCA = KernelPCA(n_components=ClusteringDim * 2, kernel='poly')
        X_kernel = kernelPCA.fit_transform(X_nonan_dm)
        save_LD_plots(X_kernel, LDLabels, 'Kernel_PCA', plots, show=False)

    # ------------------------------------------------------------------------







