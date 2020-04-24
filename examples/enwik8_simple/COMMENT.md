###torch.arange
    torch.arange(5)
    >>> tensor([ 0,  1,  2,  3,  4])

###Weight-Tie 
Weight Tying : Sharing the weight matrix between input-to-embedding layer and output-to-softmax layer; That is, instead of using two weight matrices, we just use only one weight matrix. The intuition behind doing so is to combat the problem of overfitting. Thus, weight tying can be considered as a form of regularization.

f_embed: |T| x dim(Embed) 
g_decode: { dim(Embed) or dim_last_layer } x |T|
