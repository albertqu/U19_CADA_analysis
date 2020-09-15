tos = get_trial_outcomes(mat, trial_av=True)
tos = get_trial_outcomes(mat, as_array=True)
tos.shape
cued_port_side = access_mat_with_path(mat, 'glml/value/cue_port_side', ravel=True)
np.unique(cued_port_side)
np.unique(tos)
outcomes = np.array(tos == 1.2, dtype=np.int)
outcomes[:10]
outcomes = np.array(tos == 1.2, dtype=np.float)
stays = np.zeros_like(cued_port_side)
stays[1:] = (cued_port_side[1:] == cued_port_side[:-1])
stays.shape
stays[:10]
plt.plot(outcomes, stays)
plt.plot(outcomes, stays)
plt.scatter(outcomes, stays)
plt.plot(outcomes);plt.plot(stays);plt.legend(['outcome', 'stays'])
plt.plot(outcomes);plt.plot(stays);plt.legend(['outcome', 'stays'])
plt.plot(outcomes[:-1]);plt.plot(stays[1:]);plt.legend(['outcome', 'stays'])
actions = get_trial_outcome_laterality(mat, as_array=True)
actions
actions.shape
np.sum(cued_port_side == actions)
plt.plot(actions)
stay_actions = np.zeros_like(actions)
stay_actions[1:] = (actions[1:] == actions[:-1])
np.sum(stay_actions == stay)
np.sum(stay_actions == stays)
plt.plot(outcomes[:-1]);plt.plot(stays[1:]);plt.plot(stay_actions[1:]);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1]);plt.plot(stays[1:]-1);plt.plot(stay_actions[1:]+1);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1]);plt.plot(stays[1:]-1.5);plt.plot(stay_actions[1:]+1.5);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1], markersize=1.5);plt.plot(stays[1:]-1.5, markersize=1.5);plt.plot(stay_actions[1:]+1.5,markersize=1.5);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1], markersize=10);plt.plot(stays[1:]-1.5, markersize=10);plt.plot(stay_actions[1:]+1.5,markersize=10);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1],'o-', markersize=1.5);plt.plot(stays[1:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[1:]+1.5,'o-', markersize=1.5);plt.legend(['outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1], markersize=10);plt.plot(stays[1:]-1.5, markersize=10);plt.plot(stay_actions[1:]+1.5,markersize=10);plt.legend(['past outcome', 'cue_stay', 'stay_action'])
plt.plot(outcomes[:-1],'o-', markersize=1.5);plt.plot(stays[1:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[1:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome', 'cue_stay', 'stay_action'])
outcomes_past2 = outcomes[:-2] * outcomes[1:-1]
plt.plot(outcomes_past2,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2rew', 'cue_stay', 'stay_action'])
plt.plot(outcomes_past2,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2rew', 'cue_stay', 'stay_action'])
outcomes_past2un = (outcomes[:-2] + outcomes[1:-1] > 0)
plt.plot(outcomes_past2un,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2unrew', 'cue_stay', 'stay_action'])
plt.plot(outcomes_past2,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2rew', 'cue_stay', 'stay_action'])
plt.plot(outcomes_past2un,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2unrew', 'cue_stay', 'stay_action'])
plt.plot(outcomes_past2un,'o-', markersize=1.5);plt.plot(stays[2:]-1.5, 'o-', markersize=1.5);plt.plot(stay_actions[2:]+1.5,'o-', markersize=1.5);plt.legend(['past outcome2unrew', 'cue_stay', 'stay_action'])
pred_error = (stay_actions-outcome_past2un)
pred_error = (stay_actions-outcomes_past2un)
pred_error = (stay_actions[2:]-outcomes_past2un)
plt.plot(pre_error)
plt.plot(pred_error)
smoothen = [pred_error[i:i+30] for i in range(len(pred_error)-29)]
len(smoothen)
len(pred_error)
plt.plot(smoothen)
smoothen = [np.mean(pred_error[i:i+30]) for i in range(len(pred_error)-29)]
plt.plot(smoothen)
smoothen = [np.mean(pred_error[i:i+30]) for i in range(len(pred_error)-29)]
plt.plot(smoothen)
plt.plot(smoothen)
def moving_average(s, window=30, non_overlap=False, pad=False):
    # pad in front
    if non_overlap:
        smoothen = [np.mean(s[i:i + window]) for i in range(0, len(s) - window + 1, window)]
    else:
        smoothen = [np.mean(s[i:i + window]) for i in range(len(s) - window + 1)]
    if pad:
        return np.concatenate((np.full(window-1, smoothen[0]), smoothen))
    else:
        return smoothen
plt.plot(moving_average(pred_error, window=50, non_overlap=True))
np.mean(pred_error)
all_errors = pred_error[pred_error != 0]
plt.plot(moving_average(all_errors, window=50, non_overlap=True))
plt.plot(moving_average(all_errors, window=50, non_overlap=True))
plt.plot(all_errors)
plt.plot(moving_average(all_errors, window=50, non_overlap=True))
plt.plot(moving_average(all_errors, window=50, non_overlap=True))
plt.plot(all_errors)
all_errors.shape
plt.plot(moving_average(all_errors, window=10, non_overlap=True))
np.mean(all_errors)
plt.hist(all_errors)
plt.plot(moving_average(all_errors, window=10, non_overlap=True))
plt.plot(moving_average(all_errors, window=10, non_overlap=True))
plt.plot(np.abs(pred_errors))
plt.plot(np.abs(pred_error))
plt.plot(moving_average(np.abs(pred_error), window=50, non_overlap=True))
plt.plot(moving_average(np.abs(pred_error), window=50, non_overlap=True))
np.mean(pred_error)
np.mean(np.abs(pred_error))
plt.plot(moving_average(pred_error, window=50, non_overlap=True))
peristimulus_time_trial_heatmap_plot(aligned[0], time_window, center_in_trials,
                                                               ("", "time (ms)", ""))