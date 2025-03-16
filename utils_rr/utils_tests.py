import numpy as np

def calculate_restaurant_sequence_index(restaurants):
    """Given a sequence of restaurants, infer trial number, assuming no skips.
    Example:
    >>> calculate_structured_trial_index([1, 3, 4, 1, 2, 3, 1])
    [0, 2, 3, 4, 5, 6, 8]
    """
    assert len(restaurants) > 0, "empty data"
    trial_ids = [-1] * len(restaurants)
    curr_i = 0
    r = restaurants[0]
    if r != 1:
        diff = r - 1
        curr_i += diff
    trial_ids[0] = curr_i
    prev_r = r
    for i in range(1, len(restaurants)):
        r = restaurants[i]
        if r == prev_r:
            diff = 4
        else:
            diff = (r - prev_r) % 4
        curr_i += diff
        trial_ids[i] = curr_i
        prev_r = r
    return trial_ids

def test_calculate_restaurant_sequence_index():
    test_cases = [
        [1, 3, 4, 1, 2, 3, 1],
        [1, 2, 4, 2, 4, 4, 3, 1, 2, 3, 4, 1, 2, 3, 4, 2]
    ]
    total = 0
    for test_arg in test_cases:
        result = calculate_restaurant_sequence_index(test_arg)
        correct = np.allclose(np.array(result) % 4 + 1, test_arg)
        total += int(correct)
    print('Accuracy rate: ', total / len(test_cases))

