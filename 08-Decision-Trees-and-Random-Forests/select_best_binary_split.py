import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    min_samples_leaf : int
        Minimum number of samples allowed at any leaf.

    Returns
    -------
    feat_id : int or None, one of {0, 1, 2, .... F-1}
        Indicates which feature in provided x array is used for best split.
        If None, a binary split that improves the cost is not possible.
    thresh_val : float or None
        Value of x[feat_id] at which we threshold.
        If None, a binary split that improves the cost is not possible.
    x_LF : 2D array, shape (L, F)
        Training data features assigned to left child using best split.
    y_L : 1D array, shape (L,)
        Training labels assigned to left child using best split.
    x_RF : 2D array, shape (R, F)
        Training data features assigned to right child using best split.
    y_R : 1D array, shape (R,)
        Training labels assigned to right child using best split.

    Examples
    --------
    # Example 1a: Simple example with F=1 and sorted features input
    >>> N = 6
    >>> F = 1
    >>> x_NF = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 1b: Same as 1a but just scramble the order of x
    # Should give same results as 1a
    >>> x_NF = np.asarray([2.0, 1.0, 0.0, 3.0, 5.0, 4.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 2: Advanced example with F=12 total features
    # Fill the features such that middle column is same as 1a above,
    # but the first 6 columns with random features
    # and the last 6 columns with all zeros
    >>> N = 6
    >>> F = 13
    >>> prng = np.random.RandomState(0)
    >>> x_N1 = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6,1))
    >>> x_NF = np.hstack([prng.randn(N, F//2), x_N1, np.zeros((N, F//2))])
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    6
    >>> thresh_val
    2.5

    # Example 3: binary split isn't possible (because all x same)
    >>> N = 5
    >>> F = 1
    >>> x_NF = np.asarray([3.0, 3.0, 3.0, 3.0, 3.0]).reshape((5,1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True

    # Example 4: binary split isn't possible (because all y same)
    >>> N = 5
    >>> F = 3
    >>> prng = np.random.RandomState(0)
    >>> x_NF = prng.rand(N, F)
    >>> y_N  = 1.2345 * np.ones(N)
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True
    '''
    N, F = x_NF.shape

    # Allocate space to store the cost and threshold of each feat's best split
    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)
    for f in range(F):

        # Compute all possible x threshold values for current feature
        # possib_xthresh_V : 1D array of size V
        #    Each entry is a float
        #    Entries are in sorted order from smallest to largest
        #    Represents one possible distinct threshold for provided N examples
        xunique_U = np.unique(x_NF[:,f])
        possib_xthresh_V = 0.5 * (xunique_U[:-1] + xunique_U[1:])
        V = possib_xthresh_V.size
        if MIN_SAMPLES_LEAF > 1:
            m = MIN_SAMPLES_LEAF - 1
            possib_xthresh_V = possib_xthresh_V[m:(V-m)]
        V = possib_xthresh_V.size
        if V == 0:
            # If all the x values for this feature are same, we can't split.
            # Keep cost as "infinite" and continue to next feature
            cost_F[f] = np.inf
            continue

        # TODO Compute total cost at each possible threshold
        # Your goal is *correctness*, don't prioritize speed or efficiency yet.
        # Hint: You may need several lines of code, maybe a for loop.
        # Hint 2: look below at how we assemble the left and right child 
        # Array to store the total sum of squared residuals for each of the V splits
        total_cost_V = np.inf * np.ones(V) 
        
        # Arrays to store the predicted mean for the left/right child, used for checking termination
        left_yhat_V = np.zeros(V) 
        right_yhat_V = np.zeros(V) 
        
        for v in range(V):
            thresh = possib_xthresh_V[v]
            
            # 1. Split data based on current threshold
            left_mask = x_NF[:, f] < thresh
            right_mask = np.logical_not(left_mask)
            
            y_L = y_N[left_mask]
            y_R = y_N[right_mask]
            
            N_L = y_L.size
            N_R = y_R.size
            
            # Apply MIN_SAMPLES_LEAF check (already done above for edge cases, but good to re-check)
            if N_L < MIN_SAMPLES_LEAF or N_R < MIN_SAMPLES_LEAF:
                continue

            # 2. Calculate the optimal prediction (mean) for each child
            mean_L = np.mean(y_L)
            mean_R = np.mean(y_R)
            
            # Store yhat for termination check
            left_yhat_V[v] = mean_L
            right_yhat_V[v] = mean_R

            # 3. Calculate cost (Sum of Squared Errors - SSE) for each child
            # Cost = Sum(y - y_hat)^2
            cost_L = np.sum(np.square(y_L - mean_L))
            cost_R = np.sum(np.square(y_R - mean_R))
            
            # 4. Total Cost of Split = Cost_L + Cost_R
            total_cost_V[v] = cost_L + cost_R

        # Check if there is any split that improves our cost or predictions.
        # This check is crucial for preventing infinite loops in training.
        costs_all_the_same = np.allclose(total_cost_V, total_cost_V[0], equal_nan=True)
        yhat_all_the_same = np.allclose(left_yhat_V, right_yhat_V)
        
        # We must also check if the best cost is actually an improvement over no split.
        # The cost of no split is the SSE of the entire node: np.sum(np.square(y_N - np.mean(y_N)))
        cost_no_split = np.sum(np.square(y_N - np.mean(y_N)))
        
        # If all costs are the same (no improvement) OR 
        # the best split cost is NOT strictly better than the no-split cost, 
        # we mark this feature as invalid (inf cost).
        if costs_all_the_same or np.nanmin(total_cost_V) >= cost_no_split:
            cost_F[f] = np.inf
            continue
        
        # Pick out the split candidate that has best cost
        chosen_v_id = np.argmin(total_cost_V)
        
        cost_F[f] = total_cost_V[chosen_v_id]
        thresh_val_F[f] = possib_xthresh_V[chosen_v_id]

    # Determine single best feature to use
    best_feat_id = np.argmin(cost_F)
    best_thresh_val = thresh_val_F[best_feat_id]
    
    if not np.isfinite(cost_F[best_feat_id]):
        # Edge case: not possible to split further, because
        # either all x values the same, or all y values are the same
        return (None, None, None, None, None, None)

    ## Assemble the left and right child datasets
    left_mask_N = x_NF[:, best_feat_id] < best_thresh_val
    right_mask_N = np.logical_not(left_mask_N)
    x_LF, y_L = x_NF[left_mask_N], y_N[left_mask_N]
    x_RF, y_R = x_NF[right_mask_N], y_N[right_mask_N]

    # TODO uncomment below to verify your cost computation
    left_cost = np.sum(np.square(y_L - np.mean(y_L)))
    right_cost = np.sum(np.square(y_R - np.mean(y_R)))
    assert np.allclose(cost_F[best_feat_id], left_cost + right_cost)

    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)
