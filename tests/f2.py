def supT_test(groupA:np.array, groupB=None, nperm=10000, alternative='greater', njobs=1):
    """Supremum T-test for comparing two groups"""
    def _permute_supT(groupNull, lenA, alternative):
        i_inds = np.arange(groupNull.shape[0])
        np.random.shuffle(i_inds)
        iA_inds = i_inds[:lenA]
        iB_inds = i_inds[lenA:]
        ires = stats.ttest_ind(groupNull[iA_inds], groupNull[iB_inds], axis=0, equal_var=False, alternative=alternative, nan_policy='omit')
        istat = np.nanmax(ires.statistic) if alternative == 'greater' else np.nanmin(ires.statistic)
        return istat
    
    if groupB is None:
        # now we are comparing A against 0, so we could simply do contrastive test with -A
        groupB = -groupA 
    x_res = stats.ttest_ind(groupA, groupB, axis=0, equal_var=False, alternative=alternative, nan_policy='omit')
    x_stat = np.nanmax(x_res.statistic) if alternative == 'greater' else np.nanmin(x_res.statistic)

    groupNull = np.vstack([groupA, groupB])

    lenA = groupA.shape[0]
    if njobs != 1:
        from joblib import Parallel, delayed
        all_stats = Parallel(n_jobs=njobs)(
            delayed(_permute_supT)(groupNull, lenA, alternative)
            for _ in range(nperm)
        )
    else:
        all_stats = []
        for i in range(nperm):
            istat = _permute_supT(groupNull, lenA, alternative)
            all_stats.append(istat)
        all_stats = np.array(all_stats)
    pval = np.mean(all_stats >= x_stat) if alternative == 'greater' else np.mean(all_stats<= x_stat)
    return x_stat, pval