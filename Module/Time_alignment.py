### This function for time alignment ###
def shift_and_remove_bottom_rows(arr, seq_len):
    """
    Time Adjustment
    Shift the `[:,:,-2]` column of the input array up by `seq_len` rows, then delete the bottom `seq_len` rows, while keeping the third dimension unchanged. 
    
    Parameters:
    arr (numpy.ndarray): The input 3D array with shape (batch_size, seq_len, input_dim).
    seq_len (int): The number of rows to shift up and the number of rows to delete. 

    Return:
    numpy.ndarray: The processed three-dimensional array.
    """
    last_but_one_dim = arr[:, :, -2]
    shifted_dim = np.roll(last_but_one_dim, -seq_len, axis=0)
    arr[:, :, -2] = shifted_dim
    arr = arr[:-seq_len]

    return arr

### This function creates a sliding window ###
def slide_y(y_for_AT, window_size):
    y_ = []
    for i in range(window_size,len(y_for_AT)-window_size):
        y_slided = y_for_AT[i:(i+window_size)]
        y_.append(y_slided)
    y_ = np.array(y_)
    print("y in previous step(slided):", y_.shape)
    return y_

def slide_y_pre(y_pre, window_size):
    y_ = []
    for i in range(window_size,len(y_pre)-window_size):
        y_slided = y_pre[(i-window_size):i]
        y_.append(y_slided)
    y_ = np.array(y_)
    print("y in previous step(slided):", y_.shape)
    return y_