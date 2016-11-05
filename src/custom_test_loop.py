import numpy as np

def custom_test_loop(net, X, output_layer='score', batch_size=64):
    input_dim = X.shape[-1]
    y_hat_size = net.blobs['score'].data.shape[1]
    input_len = X.shape[0]

    zero_pad = np.zeros(( batch_size - (input_len % batch_size), 1, input_dim, input_dim))
    X = np.append(X, zero_pad, axis=0)
    num_batches = X.shape[0] // batch_size

    y_hat = np.zeros((X.shape[0], y_hat_size) )

    for it in range(num_batches):
        it_range = range(it*batch_size,it*batch_size+batch_size)
        net.blobs['data'].data[...] = X[it_range]
        
        out_test = net.forward(start='conv1')
        
        out = net.blobs['score'].data
        y_hat[it_range, :] = out
        
    #     print net.blobs['fc1'].data.shape
    #     print net.blobs['score'].data.shape
    #     print out_test

    y_hat = y_hat[0:input_len]

    return y_hat
    
