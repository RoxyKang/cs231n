from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_cls = W.shape[1]

    scores = X.dot(W)

    # sum over the loss value for each training example
    for i in range(num_train):
        correct_cls = y[i]
        scores_i = scores[i] - np.max(scores[i])
        prob = np.exp(scores_i)/np.sum(np.exp(scores_i))
        loss += -np.log(prob[correct_cls])
        for j in range(num_cls):
            dW[:,j] += prob[j] * X[i]
        dW[:,correct_cls] -= X[i]
    
    # Get avg
    loss /= num_train
    dW /= num_train

    # Add regularization term, L2
    loss += reg * np.sum(W*W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]

    # Loss
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    exp_sum = np.sum(np.exp(scores), axis=1)
    
    train_num_index = np.arange(num_train)
    exp_correct_class_score = exp_scores[train_num_index, y]
    loss = np.sum(-np.log(exp_correct_class_score/exp_sum))

    # Gradient
    ind_score_div_sum = exp_scores / np.reshape(exp_sum, (exp_sum.shape[0], 1))
    ind_score_div_sum[train_num_index, y] -=1
    dW = X.T.dot(ind_score_div_sum)
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W*W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
