from collections import defaultdict
from numpy.random import randint
def select_cross_paradigm(target_paradigm, all_paradigms, p_dist_func):
    """
    Args:
        target_paradigm (Paradigm): Paradigm to randomly select from.
        all_paradigms ([Paradigm]): Iterable of all paradigms in dataset.
        p_dist_func ((Paradigm, Paradigm) => float): function for comput
    """
    pass
    # else:
    #     # TODO: implement this
    #     # TODO: consider using a precomputed distance matrix, instead?
    #     assert False

def select_rand_cross_paradigm(target_msd, target_paradigm_i, all_paradigms_w_msd):
    """Randomly select a cross-table paradigm.

    Args:
        target_msd (str) 
        target_form (str) 
        all_paradigms_w_msd ([Paradigm]): Iterable of all paradigms in dataset *that have the target_msd filled*.

    Returns:
        (str): {target_msd} form from a randomly selected paradigm.
    """
    num_paradigms = len(all_paradigms_w_msd)
    assert num_paradigms > 0, f"there are 0 paradigms with {target_msd}"
    # assert num_paradigms != 1, f"the target msd {target_msd} only has a single filled cell."
    rand_p_ind = randint(0, num_paradigms)
    selected_p = all_paradigms_w_msd[rand_p_ind]
    # NOTE: see discussion in 2021-10-08 report for why this is commented out. 
    # while selected_p.paradigm_index == target_paradigm_i:
    #     rand_p_ind = randint(0, num_paradigms)
    #     selected_p = all_paradigms_w_msd[rand_p_ind]
    return selected_p[target_msd], selected_p.paradigm_index

def create_cross_table_reinflection_frame(reinflection_frame, all_paradigms):
    """Create reinflection frame with sources from a randomly selected cross-table.

    Args:
        reinflection_frame (pd.DataFrame): DataFrame with |source_form|source_tag|target_form|target_tag|paradigm|.
        all_paradigms ([Paradigm]): List of all paradigms in the data.
    """
    def _map_msd_to_pdgms_with_inds(msds):
        msd_to_inds = defaultdict(list)
        for msd in msds:
            for i in range(len(all_paradigms)):
                if msd in all_paradigms[i]:
                    msd_to_inds[msd].append(i)
        return msd_to_inds
    
    def _convert_rf_tag_format_to_orig_format(msd):
        return msd.replace(";", "-")

    msds = all_paradigms[0].get_all_msds()
    msd_to_all_pd_inds = _map_msd_to_pdgms_with_inds(msds) 
    reinflection_frame[['cross_table_src', 'cross_table_i']] = reinflection_frame[['paradigm', 'target_tag']].apply(lambda row: select_rand_cross_paradigm(_convert_rf_tag_format_to_orig_format(row.target_tag), row.paradigm, [all_paradigms[ind] for ind in msd_to_all_pd_inds[_convert_rf_tag_format_to_orig_format(row.target_tag)]]), axis=1, result_type='expand')
    # reinflection_frame[['cross_table_src', 'cross_table_i']] = reinflection_frame[['paradigm', 'target_tag']].apply(lambda row: print(row), axis=1)
    return reinflection_frame