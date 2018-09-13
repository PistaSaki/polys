import numpy as np
import itertools as itt
import tensorflow as tf

from scipy.special import factorial, binom
import numpy.linalg as la
import matplotlib.pyplot as pl
import numbers


from pitf import ptf
from pitf.ptf import is_tf_object
from pitf import nptf
from pitf.tf_runnables import TF_Runnable

from piplot import misc as pipl

from . import pi_tensor as pit
from . import pi_batches as pib
from .pi_batches import Batched_Object


        
##################################
## Auxiliary functions operating directly on np.arrays

def get_monomials(x, degs):
    """
    `x` is tensor (numpy or tensorflow). 
    All indices are batch-indices except the last one which is the index of the variable. 
    E.g. `x` of shape [10, 30, 2] means 10 x 30 batch of two-dimensional vectors [x0, x1].
    If in addition `degs` = [3, 5], then the resulting shape is [10, 30, 3, 5].
    """
    def monomials_1var(x, deg):
        return x[..., None] ** np.arange(deg)
    
    assert x.shape[-1] == len(degs)
    
    # sometimes degs are tf.Dimension. Convert to ints:
    degs = [int(d) for d in degs]
    
    ret = 1
    for i in range(x.shape[-1]):
        xi = x[..., i]
        deg = degs[i]
        pows_xi = monomials_1var(xi, deg)
        selector = [None]*len(degs)
        selector[i] = slice(None)
        selector = [Ellipsis] + selector
        ret = ret * pows_xi[tuple(selector)]
        
    return ret

### Test get_monomials
#run_in_session(
#    get_monomials(
#        x = tf.constant([
#                [1, 0],
#                [0, 2],
#                [2, 3]
#            ]), 
#        degs = [2, 3]
#    )
#)

def eval_poly(coef, x, batch_ndim = None, var_ndim = None, val_ndim = None, ):
    def check_or_infer(n, n_infered, name):
        exception_string = (
                "Problem with shapes ({}): ".format(name) +
                " coef.shape=" + str(coef.shape) +
                " x.shape=" + str(x.shape) +
                " batch_ndim=" + str(batch_ndim) +
                " var_ndim="+ str(var_ndim) +
                " val_ndim=" + str(val_ndim)
            )
        
        assert n_infered >= 0, exception_string + "; n_infer < 0" 
        if n is None:
            return n_infered
        else:
            assert n == n_infered, (
                exception_string + "; given {} != infered {}".format(n, n_infered)
            )
            return n
        
#    if is_tf_object(x):
#        assert all(k is not None for k in [batch_ndim, var_ndim, val_ndim])
#    else:
    var_ndim = check_or_infer(var_ndim, int(x.shape[-1]), "var_ndim")
    batch_ndim = check_or_infer(batch_ndim, len(x.shape) - 1, "batch_ndim")
    val_ndim = check_or_infer(val_ndim, len(coef.shape) - batch_ndim - var_ndim, "val_ndim")

    degs = coef.shape[batch_ndim: batch_ndim + var_ndim]
    
    monoms = get_monomials(x, degs = degs)
    monoms = monoms[(Ellipsis, ) + (None, )*val_ndim]
    return nptf.reduce_sum(
        coef * monoms, 
        axis = tuple(np.arange(batch_ndim, batch_ndim + var_ndim))
    )

#run_in_session(
#    eval_poly(
#        coef = tf.constant([2, 0, 1]),
#        x = tf.constant([3])
#    )
#)
#
### Test `eval_poly`: broadcasting works for both `coef` and `x`  
#coef = tf.constant(rnd.randint(5, size=(1, 3)))
#x = tf.range(10)[:,None]
#pib.display_tensors_along_batch(
#    tensors =[
#        coef,
#        x,
#        eval_poly(
#            coef ,
#            x 
#        )
#    ],
#    batch_ndim = nptf.ndim(x) - 1
#)

def get_1D_Taylor_matrix(a, deg, trunc = None):
    """
    Taking (truncated) Taylor expansion at $a\in R$ defines a linear map $R[x] \to R[x]$.
    We return the matrix of this map.
    Our convention will be: $ deg(a_0 + a_1 x + ... + a_{n-1} x^{n-1}) = n $.
    """
    M = np.array([
        [binom(n, k) * a**(n-k) for k in range(n + 1)] + [0]*(deg - n - 1)
        for n in range(deg)
    ]).T
    
    if trunc is not None:
        M = M[:trunc]
        
    return M


#get_1D_Taylor_matrix( a = 2, deg = 4, trunc = 2)

def get_1d_Taylor_coef_grid(coef, poly_index, new_index, control_times, trunc = None):
    """
    Returns a tensor with one new index of length = len(control_times)
    """
    A = np.array([
        get_1D_Taylor_matrix(a, deg = coef.shape[poly_index], trunc = trunc).T
        for a in control_times        
    ], dtype = nptf.np_dtype(coef))
    
    if poly_index >= new_index:
        poly_index += 1
        

    ##It should work like this the following commented code and it does in numpy. 
    ##However tensorflow can't broadcast matmul yet, so it fails in tensorflow.
    #taylors = right_apply_map_along_batch(
    #    X = nptf.expand_dims(coef, new_index),
    #    A = A,
    #    batch_inds = [new_index],
    #    contract_inds = [poly_index],
    #    added_inds = [poly_index]
    #)

    ## Also this way could be written more concisely if tensorflow had equivalent of np.repeat:
    coef_repeated = nptf.tile(
            nptf.expand_dims(coef, new_index),
            reps = [len(control_times) if i == new_index else 1 for i in range(nptf.ndim(coef) + 1)]
        )
    
    taylors = pit.right_apply_map_along_batch(
        X = coef_repeated,
        A = A,
        batch_inds = [new_index],
        contract_inds = [poly_index],
        added_inds = [poly_index]
    )
    return taylors

#get_1d_Taylor_coef_grid(
#    coef = np.eye(2),
#    poly_index = 1,
#    new_index = 1, 
#    control_times=[0, 1, 2]
#)

def get_1D_Taylors_to_spline_patch_matrix(a, b, deg):
    """
    `deg` is the degree of the Taylors. Thus the degree of the spline is 2 * deg.
    """
    Taylors_matrix = np.concatenate([
            get_1D_Taylor_matrix(a, deg = 2 * deg, trunc = deg),
            get_1D_Taylor_matrix(b, deg = 2 * deg, trunc = deg),
        ])
    
    #print(Taylors_matrix)
    return la.inv(Taylors_matrix)
                        
        
#get_1D_Taylors_to_spline_patch_matrix(0, 1, 2)
    

def get_Catmul_Rom_Taylors_1D(coef, control_index, control_times, added_index):
    assert len(control_times) == coef.shape[control_index]
         
    i0 = list(range(len(control_times)))
    c0 = coef
    
    # `t0` will be a tensor of the same ndim as `coef` but 
    # all dimenstions except at `control_index` are 1
    t_shape = np.ones(len(coef.shape), dtype=int)
    t_shape[control_index] = -1
    t0 = np.reshape(control_times, t_shape)
    
    i_minus = [0] + i0[:-1]
    i_plus = i0[1:] + [i0[-1]]
    
    t_minus = nptf.gather(  t0, indices = i_minus, axis = control_index)
    c_minus = nptf.gather(coef, indices = i_minus, axis = control_index)
    
    t_plus = nptf.gather(  t0, indices = i_plus, axis = control_index)
    c_plus = nptf.gather(coef, indices = i_plus, axis = control_index)
    
    der = (c_plus - c_minus) / (t_plus - t_minus)
    
    return nptf.stack([c0, der], axis = added_index )
    
    
    
    

def array_poly_prod(a, b, batch_ndim = 0, var_ndim = None, truncation = None):
    """
    `a`, `b` are np.arrays
    truncation can be None, number or an array, 
    specifying the maximal allowed degrees  in the product.
    """
    ## take `a` to be the polynomial with values of lower ndim
    if a.ndim > b.ndim:
        a, b = b, a
        
    ## if `var_ndim` is missing, infer from shapes
    if var_ndim is None:
        var_ndim = a.ndim - batch_ndim
        
    ## check whether `a` has scalar values
    assert (a.ndim == batch_ndim + var_ndim) or (a.ndim == b.ndim), (
        "You have two possibilities: " + 
        "either the values of the two polynomials have the same ndim, " +
        "or one of them has scalar values."        
    )
    
    ## shape of values of the resulting polynomial
    val_shape = b.shape[batch_ndim + var_ndim:]
    
    ## add some dimensions at the end of `a` so that 
    ## it has the same ndim as `b`
    a = a[(Ellipsis, ) + (None, ) * (b.ndim - a.ndim)]
    
    ## degrees of the polys
    degs_a, degs_b = [np.array(x.shape)[batch_ndim: batch_ndim + var_ndim] for x in [a,b] ]
    deg_c = degs_a + degs_b - 1
    
    if truncation is not None:
        truncation = truncation * np.ones_like(deg_c)
        deg_c = np.array([min(t, d) for t, d in zip(truncation, deg_c)])
        
    c_batch_shape = pib.get_common_broadcasted_shape([
            a.shape[:batch_ndim], b.shape[:batch_ndim]
    ])
    
    c = np.zeros( c_batch_shape + list(deg_c) + list(val_shape) , dtype = np.promote_types(a.dtype, b.dtype))
    
    
    for i in range(np.prod(degs_a)):
        mi = np.unravel_index(i, degs_a)

        for j in range(np.prod(degs_b)):
            mj = np.unravel_index(j, degs_b)
            
            #print(mi, mj, deg_c)

            if all(np.add(mi, mj) < deg_c):
                batches = (slice(None),)*batch_ndim
                a_index = batches + mi
                b_index = batches + mj
                c_index = batches + tuple(np.add(mi, mj))
                c[c_index] += a[a_index] * b[b_index]
 
    return c
    

def get_spline_from_taylors_1D_OLD(taylor_grid_coeffs, bin_axis, polynom_axis, control_times):
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    
    stacked_taylors = nptf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    A = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, b, deg = int(taylor_grid.shape[polynom_axis])
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = nptf.np_dtype(taylor_grid_coeffs)
    )

    coef = pit.right_apply_map_along_batch(
        X = stacked_taylors, A = A, 
        batch_inds = [bin_axis], contract_inds = [polynom_axis], added_inds = [polynom_axis]
    )
    
    
    return coef
    
def get_spline_from_taylors_1D(taylor_grid_coeffs, bin_axis, polynom_axis, control_times):
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    
    stacked_taylors = nptf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    dtype = nptf.np_dtype(taylor_grid_coeffs)
    deg = int(taylor_grid.shape[polynom_axis])
    ## reparametrization matrices for expressing the taylors in bin-scaled 
    ## Note that we have already stuck the two taylors together along the polynom-axis
    ## so our diagonal reparametrization matrices have the diagonal repeated twice
    ## ( thus the resulting matrix has dimension 2 * deg)
    RM = np.array(
        [
            np.diag([ (b- a)**k for k in range(deg) ] * 2)
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    SM = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, deg = deg
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )

    coef = pit.right_apply_map_along_batch(
        X = stacked_taylors, A = RM @ SM, 
        batch_inds = [bin_axis], contract_inds = [polynom_axis], added_inds = [polynom_axis]
    )
    
    
    return coef


def get_1D_integral_of_piecewise_poly(                             
        coef, bin_axis, polynom_axis, control_times,
        polys_are_in_bin_coords = True
    ):
    deg = int(coef.shape[polynom_axis])
    def integration_functional(a, b, deg):
        n = np.arange(deg)
        if polys_are_in_bin_coords:  
            return 1/(n + 1) * (b - a)#**(n+1)
        else:
            return 1/(n + 1) * (b**(n+1) - a**(n+1))
    
    IM = np.array([
            integration_functional(a, b, deg)
            for a, b in zip(control_times[:-1], control_times[1:])
        ], dtype = nptf.np_dtype(coef))
    
    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(
        X = coef, A = IM, 
        batch_inds = [], contract_inds = [bin_axis, polynom_axis], added_inds = []
    ) 
    
#get_1D_integral_of_piecewise_poly(
#    coef = np.array([
#        [0, 1, 0],
#        [1, 0, 0],
#    ]),
#    bin_axis = 0,
#    polynom_axis = 1,
#    control_times = [-1, 0, 1]
#)
    
def get_integral_of_spline_from_taylors_1D(
        taylor_grid_coeffs, bin_axis, polynom_axis, control_times,
        polys_are_in_bin_coords = True
    ):
    """
    This is basically a composition of 
    `get_spline_from_taylors_1D` and `get_1D_integral_of_piecewise_poly`
    I just believe that doing it in one step can be 2**n times faster
    where n is the number of dimensions, which is not too much.
    """
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    
    stacked_taylors = nptf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    dtype = nptf.np_dtype(taylor_grid_coeffs)
    deg = int(taylor_grid.shape[polynom_axis])
    ## reparametrization matrices for expressing the taylors in bin-scaled 
    ## Note that we have already stuck the two taylors together along the polynom-axis
    ## so our diagonal reparametrization matrices have the diagonal repeated twice
    ## ( thus the resulting matrix has dimension 2 * deg)
    RM = np.array(
        [
            np.diag([ (b- a)**k for k in range(deg) ] * 2)
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    SM = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, deg = deg
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
            
    

    ## finally we need to integrate 
    def integration_functional(a, b, deg):
        n = np.arange(deg)
        if polys_are_in_bin_coords:  
            return 1/(n + 1) * (b - a)#**(n+1)
        else:
            return 1/(n + 1) * (b**(n+1) - a**(n+1))
    
    IM = np.array([
            integration_functional(a, b, 2 * deg)
            for a, b in zip(control_times[:-1], control_times[1:])
        ], dtype = dtype)
    
    
    ## we multiply all the transformations 
    ## (we must make IM that is a batch of functionals into matrices)
    A = (RM @ SM @ IM[..., None])[..., 0]
    
    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(
        X = stacked_taylors, A = A, 
        batch_inds = [], contract_inds = [bin_axis, polynom_axis], added_inds = []
    ) 
    


    
#####################################
## Some other auxilliary funs

def plot_fun(f, start, end, **kwargs):
    
    if len(start) == 1:
        start = start if start is not None else 0
        end = end if end is not None else 1
        start, end = [x[0] if np.array(x).ndim > 0 else x for x in [start, end] ]            

        xxx = np.linspace(start, end, 100)
        yyy = [ f(np.array([x])) for x in xxx]
        pl.plot(xxx, yyy, **kwargs)

    elif len(start) == 2:
        start = start if start is not None else [0,0]
        end = end if end is not None else [1,1]

        pipl.plot_fun_heatmap(
            f = f,
            xxx = np.linspace(start[0], end[0], 30),
            yyy = np.linspace(start[1], end[1], 30),
        )
        
def replace_numbers_in_array(a, zeros):
    for i in range(a.size):
        if isinstance(a.flat[i], numbers.Number):
            a.flat[i] += zeros
            


def get_bin_indices(bins, tt):
    """
    Args:
        bins: float tensor with shape `[noof_bins + 1]`
        tt: float tensor with shape `batch_shape`
        
    Return:
        integer tensor with shape `batch_shape`
    """
    batch_ndim = nptf.ndim(tt)

    tt_reshaped = tt[..., None] 
    bins_reshaped = bins[(None,) * batch_ndim + (slice(None),)]
                         
    return nptf.reduce_sum(
        nptf.cast( tt_reshaped >= bins_reshaped , np.int64),
        axis = -1
    ) - 1
    
    
####################################
## Classes

class Val_Indexer:
    def __init__(self, obj):
        self.obj = obj
        
    def __getitem__(self, selector):
        return self.obj._val_getitem(selector)
        
        
class Val_Indexed_Object:
    @property
    def val(self):
        return Val_Indexer(obj = self)
    
    
            
class Poly(Batched_Object, Val_Indexed_Object, TF_Runnable):
    def __init__(self, coef, batch_ndim = 0, var_ndim = None, val_ndim = 0):
        self.coef = coef
        self.batch_ndim = batch_ndim
        if var_ndim is None:
            var_ndim = nptf.ndim(coef) - batch_ndim - val_ndim    
        self.var_ndim = var_ndim
        
        self.val_ndim = val_ndim
        
        try:
            self.degs = [int(d) for d in self.coef.shape[batch_ndim: batch_ndim + var_ndim]]
        except TypeError as err:
            raise Exception(
                err, 
                "Can not infer degrees from coef.shape = {} "
                "batch_ndim = {}, var_ndim = {}, val_ndim = {}".format(
                    coef.shape, self.batch_ndim, self.var_ndim, self.val_ndim
                )
            )
            
                     
        assert nptf.ndim(coef) == self.batch_ndim + self.var_ndim + self.val_ndim,(
            "The sum of batch_ndim = {}, var_ndim = {}, val_ndim = {} "
            "should be equal to coef_ndim = {}.".format(
                self.batch_ndim, self.var_ndim, self.val_ndim, nptf.ndim(coef)
            )
        )
        
    @staticmethod
    def from_tensor(a, batch_ndim = 0, var_ndim = None):
        """Generalization of "matrix of quadratic form" -> "polynomial of order 2".
        
        A tensor `a` with shape = [n]*deg defines 
        a polynomial of degree `deg` (wrt. each variable) in n-dimensional space.
        The array of coefficients of the result has shape [deg]*n.
        """
        deg = nptf.ndim(a) - batch_ndim
        if deg == 0:
            assert var_ndim is not None, "For a constant polynomial you must specify the number of variables."
            return Poly(
                coef = a[(...,) + (None,)*var_ndim],
                batch_ndim = batch_ndim,
                var_ndim = var_ndim
            )
        
        a_shape = a.shape
        n = a_shape[-1]
        if var_ndim is not None: 
            assert n == var_ndim
            
        assert all([dim == n for dim in a_shape[batch_ndim:]])
        
        if ptf.is_tf_object(a):
            a_np = ptf.unstack_to_array(a, start_index=batch_ndim)
            f_np = Poly.from_tensor(a_np, batch_ndim=0)
            ## there are some zeros among coeffs of f_np so replace by tf tensors of appropriate shape
            batch_shape = tf.shape(a)[:batch_ndim]
            replace_numbers_in_array(f_np.coef, tf.zeros(batch_shape, dtype = a.dtype))
            
            return f_np._stack_coef_to_tf(batch_ndim = batch_ndim)

        batch_shape = a.shape[:batch_ndim]

        coef = np.zeros(batch_shape + (deg +1,)* n , dtype = a.dtype)
        for iii in itt.product(*([range(n)]*deg)):
            degs = np.zeros(n, dtype = np.int)
            for i in iii:
                degs[i] += 1

            #print(iii, degs, a[iii])    
            bs = (slice(None),) * batch_ndim
            coef[bs + tuple(degs)] += a[bs + iii]

        return Poly(coef, var_ndim = n, batch_ndim=batch_ndim)
    
    def unit_like(p):
        """Return a polynomial representing 1 with the same batch_ndim and degrees as `p`.
        """
        shape = nptf.shape(p.coef)
        batch_ndim = p.batch_ndim
        batch_shape = shape[ : batch_ndim]
    
        dtype = nptf.np_dtype(p.coef)
    
        ## first construct the coefficients for one-element batch
        ## (i.e. batch_ndim = 0)
        one = np.zeros(np.prod(p.degs), dtype)
        one[0] = 1
        one = one.reshape([1]*batch_ndim + p.degs)
    
        ## add the batch-dimensions
        one = nptf.ones(batch_shape, dtype)[(Ellipsis,) + (None,) * p.var_ndim] * one
    
        return Poly(coef = one, batch_ndim= batch_ndim, var_ndim=p.var_ndim)
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return Poly(
                coef = data,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
        
        
    def _unstack_coef_to_array(self):
        assert ptf.is_tf_object(self.coef)
        return Poly(
            coef = ptf.unstack_to_array(
                x = self.coef,
                ndim = self.var_ndim,
                start_index = self.batch_ndim
            ),
            batch_ndim = 0,
            var_ndim = self.var_ndim,
            val_ndim = 0
        )
    
    def _stack_coef_to_tf(self, batch_ndim):
        coef = ptf.stack_from_array(
                a = self.coef,
                start_index = batch_ndim            
            )
        
        return Poly(
            coef = coef,
            batch_ndim = batch_ndim,
            var_ndim = self.var_ndim,
            val_ndim = len(coef.shape) - batch_ndim - self.var_ndim
        )
    
    def _put_coefs_to_tf_constant_if_np(self):
        if ptf.is_tf_object(self.coef):
            return self
        else:
            return Poly(
                coef = tf.constant(self.coef),
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        
    
            
    def __repr__(self):
        s = "Poly( " + str(self.coef) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += " )"
        
        return s
    
    def is_scalar(self):
        return self.val_ndim == 0
    
    def __call__(self, x):
        if not is_tf_object(x):
            x = np.array(x)
        return eval_poly(self.coef, x, batch_ndim = self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim )
    
    
    
    def __mul__(self, other, truncation = None):
            
        if isinstance(other, Poly):
            assert self.batch_ndim == other.batch_ndim
            assert self.var_ndim == other.var_ndim
            assert self.val_ndim == 0 or other.val_ndim == 0 or (self.val_ndim == other.val_ndim), (
                "At least one of the two polynomials should have scalar values or both should have the same values. " + 
                "This can be generalised but is not implemented. "
            )

            if any([ptf.is_tf_object(f.coef) for f in {self, other}]):
                f, g = [x._put_coefs_to_tf_constant_if_np() for x in [self, other]]
                
                ## It can happen that one of the polynomials 
                ## is not scalar-valued i.e. val_ndim > 0.
                ## In that case, we must add the corresponding number
                ## of dimesions to the values of the other one.
#                print(
#                        "f.val_ndim = " + str(f.val_ndim) + "; " +
#                        "g.val_ndim = " + str(g.val_ndim) + ". " 
#                    )


                if f.val_ndim == 0 or g.val_ndim == 0:
                    if f.val_ndim > g.val_ndim:
                        f, g = g, f
                    f = f.val[(None, ) * (g.val_ndim - f.val_ndim)]
                
                assert f.val_ndim == g.val_ndim, (
                    "f.val_ndim = " + str(f.val_ndim) + "; " +
                    "g.val_ndim = " + str(g.val_ndim) + ". " 
                )
                
                ## make f, g into polys with np.array coef
                f_np, g_np = [x._unstack_coef_to_array() for x in[f, g]]
#                print("shape of elements in coefficient fields of f_np, g_np is " +
#                      str(f_np.coef.flat[0].shape), str(g_np.coef.flat[0].shape)
#                )
                
                prod_np = f_np.__mul__(g_np, truncation = truncation)
                return prod_np._stack_coef_to_tf(self.batch_ndim)
            else:
                return Poly(
                    coef = array_poly_prod(
                        self.coef, other.coef, 
                        batch_ndim = self.batch_ndim, 
                        var_ndim = self.var_ndim,
                        truncation=truncation
                    ),
                    batch_ndim = self.batch_ndim,
                    var_ndim = self.var_ndim,
                    val_ndim = max(self.val_ndim, other.val_ndim)
                )
        else:
            if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
                other_ndim = nptf.ndim(other)
                assert other_ndim <= self.batch_ndim, (
                    "You are multiplying a polynomial with batch_dim = " + str(self.batch_ndim) +
                    " by a tensor of ndim = " + str(other_ndim) + "."
                )
                other = other[(...,) + (None,) * (self.ndim - other_ndim)]
            
            return Poly(
                coef = self.coef * other,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        
    def __rmul__(self, other):
        return self * other
    
    @property
    def batch_shape(self):
        return nptf.shape(self.coef)[:self.batch_ndim]
    
    @property
    def ndim(self):
        return self.batch_ndim + self.var_ndim + self.val_ndim
    
    @property
    def val_shape(self):
        return nptf.shape(self.coef)[self.batch_ndim + self.var_ndim:]
    
        
    def __add__(self, other):    
        if isinstance(other, Poly):
            assert self.batch_ndim == other.batch_ndim
            assert self.var_ndim == other.var_ndim
            assert self.val_ndim == other.val_ndim
            
            f, g = self, other
            f = f.pad_degs(g.degs)
            g = g.pad_degs(f.degs)

            return Poly(
                coef = f.coef + g.coef,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        else:
            return self + Poly.constant(other, self.batch_ndim, self.var_ndim, self.val_ndim)
        
    def __radd__(self, other):
        return self + other
    
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
    def __truediv__(self, other):
        #assert isinstance(other, numbers.Number), "Unsupported type {}.".format(type(other))
        return 1/other * self
        
        
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        assert isinstance(a, (np.ndarray, tf.Tensor, tf.Variable))
            
        f = self
        assert nptf.ndim(a) == f.val_ndim
        added_ndim = f.batch_ndim + f.var_ndim
        a = a[(None,) * added_ndim + (Ellipsis, )]
        #print(a.shape, f.coef.shape)
        
        return Poly(
            coef = f.coef * a, 
            batch_ndim = f.batch_ndim,
            var_ndim = f.var_ndim,
            val_ndim = f.val_ndim
        )
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `Poly` return `Poly` whose values are sums of the values of the original.
        """
        f = self
        if axis is None:
            axis = - np.arange(f.val_ndim) - 1
            axis = tuple(axis)
        else:
            ## make `axis` into a list
            try:
                axis = list(tuple(axis))
            except TypeError:
                axis = (axis, )
                
            ## force the `axis` to be positive
            axis = [i if i >= 0 else i + f.val_ndim for i in axis]
            assert all(0 <= i < f.val_ndim for i in axis )
            
            ##
            axis = np.array(axis) + f.batch_ndim + f.var_ndim
            axis = tuple(axis)
                
        #print("axis =", axis)
        return Poly(
            coef = nptf.reduce_sum(
                f.coef, 
                axis = axis
            ),
            batch_ndim = f.batch_ndim,
            var_ndim = f.var_ndim
        )
    
    def truncate_degs(self, limit_degs):
        degs = np.minimum(limit_degs, self.degs)
        selector = [slice(None)]* self.batch_ndim +  [slice(None, int(deg)) for deg in degs]
        return Poly(
            coef = self.coef[selector], 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim
        )
    
    
    def pad_degs(self, minimal_degs):
        ## In this first part we can work solely in numpy (i.e. with tangible numbers)
        paddings = np.concatenate([
                np.zeros(self.batch_ndim),
                np.maximum(self.degs, minimal_degs) - self.degs,
                np.zeros(self.val_ndim),
            ]).astype(np.int)
        
        paddings = np.stack([
                nptf.zeros_like(paddings), 
                paddings
        ]).T
        
        #print("paddings =", paddings)
        
        ## Only the padding must be done in appropriate module (np/tf)
        return Poly(
            coef = nptf.pad(self.coef, paddings=paddings), 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim
        )
        
        
    
    
    
    def plot(self, start = None, end = None, **kwargs):
        f = self
        assert f.batch_ndim == 0
        
        start = [0] * f.var_ndim if start is None else start
        end = [1] * f.var_ndim if end is None else end
        plot_fun(f, start, end, **kwargs)
        
    def Taylor_at(self, a):
        assert a.shape[-1] == self.var_ndim 
        if nptf.ndim(a) == 1:
            taylor_coef = pit.apply_tensor_product_of_maps(
                matrices = [
                    get_1D_Taylor_matrix(ai, deg = self.degs[i])
                    for i, ai in enumerate(a)
                ],
                x = self.coef,
                start_index = self.batch_ndim
            )
            return Poly(
                coef = taylor_coef, 
                batch_ndim = self.batch_ndim, 
                var_ndim = self.var_ndim, 
                val_ndim = self.val_ndim
            )
        else:
            #assert nptf.ndim(a) - 1 = self.batch_ndim
            #for i in range(self.var_ndim):
            #    get_1D_Taylor_matrix(a[..., i], deg = self.degs[i])
            
            raise NotImplementedError()
    
    def shift(self, shift):
        return self.Taylor_at(-np.array(shift))
    
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return Poly(
            coef = coef, 
            batch_ndim = batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim            
        )

    def _val_getitem(self, selector):
        if not isinstance(selector, tuple):
            selector = (selector,)
            
        coef = self.coef[
            (slice(None), )*(self.batch_ndim + self.var_ndim)
            + selector
        ]
        
        val_ndim = self.val_ndim - (len(self.coef.shape) - len(coef.shape))
        assert val_ndim >= 0, "val_ndim = " + str(val_ndim)
        return Poly(
            coef = coef, 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = val_ndim            
        )
        
    def truncated_exp(self, degs = None):
        """Return exponential in the local ring $R[x]/ x^degs$. 
        
        For example if degs = 4 then `truncated_exp(x) = 1 + x + 1/2 x^2`.
        """
        if degs is None:
            degs = self.degs
        
        g = self
        
        # We divide g into constant `a` and nilpotent part `b` (in the truncated ring)
        a = g.truncate_degs(1)
        b = (g - a).truncate_degs(degs)

        # We conpute exp(g) = exp(a + b) = exp(a) * exp(b)
        # `a` is scalar so exp(a) poses no problem.
        exp_a = Poly(
            coef = nptf.exp(a.coef), 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim 
        )
        # Since b is nilpotent with highest nonzero power at most n = total degree:
        n = sum([deg - 1 for deg in degs])
        # we can calculate exp(b) as sum of the first n + 1 terms of the power series.

        exp_b = 1 + b
        b_k = b
        for k in range(2, n+1):
            b_k = b_k.__mul__(b, truncation = degs)
            exp_b = exp_b + 1/factorial(k) * b_k
            
        return exp_a * exp_b

        
    def truncated_fun(self, fun_der, degs = None):
        """Return `fun(self)` in the local ring `R[x]/ x^degs` for `fun: R -> R`. 
        
        Generalizes `truncated_exp`.
        
        Args:
            fun_der: callable s.t. `fun_der(k, t)` is a k-th derivative of `fun` at `t \in R`.
                The argument `t` may be a tensor and `fun` should be applied to each coordinate separately.
                The argument `k` is an integer.
            degs: list of ints 
        """
        if degs is None:
            degs = self.degs
        
        g = self
        
        # We divide g into constant `a` and nilpotent part `b` (in the truncated ring)
        a = g.truncate_degs(1)
        b = (g - a).truncate_degs(degs)

        # `b` is nilpotent with highest nonzero power at most n = total degree:
        n = sum([deg - 1 for deg in degs])
        
        # We calculate fun(g) = fun(a + b) as the sum of the first n + 1 terms 
        # of the power series
        # $ \sum_k c_k b^k / k! $
        # where `c_k` is k-th derivative of `fun` at `a`. 

        exp_g = 0
        b_k = 1
        for k in range(n+1):
            c_k = Poly(
                coef = fun_der(k = k, t = a.coef), 
                batch_ndim = self.batch_ndim, 
                var_ndim = self.var_ndim, 
                val_ndim = self.val_ndim 
            )
            
            #print("k = {}; b_k = {}, c_k = {}".format(k, b_k, c_k))
            exp_g = exp_g + c_k * b_k / factorial(k)
            
            b_k = b.__mul__(b_k, truncation = degs)
            
            
        return exp_g
        
    def truncated_inverse(self, degs = None):
        return self.truncated_fun(
            fun_der = lambda k, t: -1 / (-t)**(k+1) * factorial(k),
            degs = degs
        )
        
    
    def truncated_power(self, exponent, degs = None):
        """Return `self` raised to the exponent (possibly real) truncated to `degs`. 
        """
        a = exponent
        return self.truncated_fun(
            fun_der = lambda k, t: binom(a, k) * factorial(k) * t**(a-k),
            degs = degs
        )
    
    
      
    def get_Taylor_grid(self, params, truncs = None):
        assert len(params) == self.var_ndim
        assert issubclass( nptf.np_dtype(self.coef).type, np.floating), (
            "Polynomial should have coef of floating dtype. "
            "At the moment dtype = {}.".format(nptf.np_dtype(self.coef))
        )
        
        if truncs is None:
            truncs = self.degs
        if isinstance(truncs, numbers.Number):
            truncs = [truncs]*len(params)
            
            
        taylors = self.coef
        for i, (par, trunc) in enumerate(zip(params, truncs)):
            taylors = get_1d_Taylor_coef_grid(
                coef = taylors, 
                poly_index = self.batch_ndim + 2 * i, 
                new_index = self.batch_ndim + i, 
                control_times = par, 
                trunc = trunc
            )
            
        
        return TaylorGrid(
            coef=taylors, params = params, 
            batch_ndim=self.batch_ndim, val_ndim=self.val_ndim
        )
    
    def to_PolyMesh(self, params):
        assert len(params) <= self.batch_ndim
        return PolyMesh(
            coef = self.coef, 
            params = params, 
            batch_ndim=self.batch_ndim - len(params), 
            val_ndim=self.val_ndim
        )

    def to_TaylorGrid(self, params):
        assert len(params) <= self.batch_ndim
        return TaylorGrid(
            coef = self.coef, 
            params = params, 
            batch_ndim=self.batch_ndim - len(params), 
            val_ndim=self.val_ndim
        )
        
    def der(self, k = None):
        """Derivative w.r.t. $x_k$."""
        
        if k is None:
            if self.var_ndim == 1:
                k = 0
            else:
                raise Exception("You did not specify by which coordinate you want to differentiate. ")
        
        coef = self.coef
        
        k2 = k + self.batch_ndim 
        deg = self.degs[k]
        ndim = nptf.ndim(self.coef)
        
        selector = [slice(None)] * ndim
        selector[k2] = slice(1, None)
        
        sh = [1] * ndim
        sh[k2] = deg - 1
        
        coef = coef[selector] * np.arange(1, deg).reshape(sh)
        
        return Poly(
            coef=coef,
            batch_ndim = self.batch_ndim,
            var_ndim= self.var_ndim,
            val_ndim=self.val_ndim
        )
        

    @staticmethod
    def constant(c, batch_ndim, var_ndim, val_ndim, ):
        if not ptf.is_tf_object(c):
            c = np.array(c)
        assert nptf.ndim(c) == 0, "Maybe this method does not do what you want." 
        
        c = nptf.reshape(c, [1]* (batch_ndim + var_ndim + val_ndim))
        return Poly(
            coef = c, 
            batch_ndim = batch_ndim, 
            var_ndim = var_ndim, 
            val_ndim = val_ndim 
        )
        
        
#p = Poly( np.array([  2,  30,   0] ))
#print(p)
#print(p.truncated_exp())
#print(
#      p.truncated_fun(lambda k, t: np.exp(t) )
#)
#
#print("truncated_inverse(p) = ", p.truncated_inverse())
#print("p * truncated_inverse(p) = ", p * p.truncated_inverse())
#print("p.der() =", p.der())


def ipow(base, exponent):
    """ For integral non-negative `exponent` calculate `base ** exponent` using only multiplication.
    """
    result = 1
    
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base

        exponent //= 2
        base *= base
        
    return result

        
class PolyMesh(Batched_Object, TF_Runnable):
    
    def __init__(self, coef, params, batch_ndim = 0,  val_ndim = 0):
        
        self.coef = coef
        self.batch_ndim = batch_ndim
        self.params = [np.array(cc, dtype = nptf.np_dtype(coef)) for cc in params ]
        self.val_ndim = val_ndim
        
        assert len(self.coef.shape) == self.batch_ndim + 2 * self.var_ndim + self.val_ndim
        assert list(self.coef.shape[self.batch_ndim: self.batch_ndim + self.var_ndim]) == list(self.bins_shape)
    
        
    def __repr__(self):
        s = "Polymesh( " 
        s += "var_ndim = " + str(self.var_ndim)
        s += ", params.lengts = " + str([len(par) for par in self.params]) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += ", coef.shape = " + str(self.coef.shape) 
        
        s += " )"
            
        
        return s
        
    def __mul__(self, other):
            
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, PolyMesh) else other
        return (f * g).to_PolyMesh(self.params)

    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, PolyMesh) else other
        return f.__add__(g).to_PolyMesh(self.params)
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
    def __truediv__(self, other):
        return 1/other * self
        
    def __pow__(self, exponent):
        n = exponent
        assert isinstance(n, numbers.Integral)
        assert n > 0
        return ipow(self, exponent)
        
        
    
    def der(self, k = None):
        """Derivative w.r.t. $x_k$.$"""
        
        if k is None:
            if self.var_ndim == 1:
                k = 0
            else:
                raise Exception("You did not specify by which coordinate you want to differentiate. ")    
        
        df = self.to_Poly().der(k).to_PolyMesh(self.params)
        
        ## since the parametrization is different in each bin, we must rescale
        bin_index = self.batch_ndim + k
        scale = self.params[k][1:] - self.params[k][:-1]
        
        sh = [1] * nptf.ndim(self.coef)
        sh[bin_index] = -1
        scale = scale.reshape(sh)
        
        df.coef = df.coef / scale
        
        return df
        
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return PolyMesh(
            coef = coef,
            params = self.params,
            batch_ndim = batch_ndim, 
            val_ndim = self.val_ndim            
        )
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return PolyMesh(
                coef = data,
                params = self.params,
                batch_ndim = self.batch_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
    
    @property
    def batch_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[:self.batch_ndim]
    
    @property
    def val_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[self.batch_ndim + 2 *self.var_ndim:]
    
    @property
    def var_ndim(self):
        return len(self.params)
    
    @property
    def degs(self):
        return self.coef.shape[self.batch_ndim + self.var_ndim: self.batch_ndim + 2 *self.var_ndim]
    
    @property
    def grid_shape(self):
        return [len(par) for par in self.params]
    
    @property
    def bins_shape(self):
        return [len(par) - 1 for par in self.params]
    
    
    def __call__(self, x):
        if not is_tf_object(x):
            x = np.array(x)
        
        assert nptf.ndim(x) - 1 == self.batch_ndim, (
            "Batch-ndim of `x` is {} "
            "but the batch-ndim of this MeshGrid is {}.".format(
                nptf.ndim(x) - 1, self.batch_ndim
            )
        )
        assert x.shape[-1] == self.var_ndim, (
            "Variable-dimension of this MeshGrid is {} "
            "but the dimension of `x` you want to plug into is is {}.".format(
                self.var_ndim, x.shape[-1], 
            ) 
        )
        
        ## divide `x` into list of its coordinates
        x_unstack = nptf.unstack(x, axis = -1)
        
        ## force all coordinates of `x` into the range of params
        x_unstack = [
            nptf.maximum(cc[0], nptf.minimum(t, cc[-1])) 
            for cc, t in zip(self.params, x_unstack)
        ]
        
        ## find the bin (polynomial patch) containing x
        bin_indices_unstack = [
             nptf.minimum(get_bin_indices(np.array(cc), t), len(cc)-2) 
             for cc, t in zip(self.params, x_unstack)
            ]
        
        bin_indices = nptf.stack( bin_indices_unstack, axis = -1)
        
        poly = Poly(
            coef = nptf.batched_gather_nd(
                a = self.coef, indices = bin_indices, batch_ndim = self.batch_ndim
            ),
            batch_ndim = self.batch_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim 
        )
        
        ## reparametrize `x` in the relative coordinates of the corresponding bin
        bin_start = nptf.stack(
            [
             nptf.gather(cc, ii)
             for cc, ii in zip(self.params, bin_indices_unstack)
            ],
            axis = -1
        )
        
        bin_end = nptf.stack(
            [
             nptf.gather(cc, ii + 1)
             for cc, ii in zip(self.params, bin_indices_unstack)
            ],
            axis = -1
        ) 
        
        x_rel = (x - bin_start) / (bin_end - bin_start)
    
        return poly(x_rel)
    
    def to_Poly(self):
        return Poly(
            coef = self.coef, 
            batch_ndim=self.batch_ndim + self.var_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim
        )
    
    
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        f = self.to_Poly()
        return f.val_mul(a).to_PolyMesh(self.params)
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `PolyMesh` return `PolyMesh` whose values are sums of the values of the original.
        """
        f = self.to_Poly()
        return f.val_sum(axis).to_PolyMesh(self.params)
    
    
    def bin_indices(self):
        return itt.product(*[range(len(par) -1) for par in self.params])
    
    def bin_start(self, ii):
        return np.array(
            [par[i] for i, par in zip(ii, self.params) ]
        )
    
    def bin_end(self, ii):
        return np.array(
            [par[i + 1] for i, par in zip(ii, self.params) ]
        )
    
    def domain_start(self):
        return [par[0] for par in self.params]
    
    
            
    def plot(self, **kwargs):
        plot_fun(self, 
            start = [par[0] for par in self.params],
            end = [par[-1] for par in self.params],
            **kwargs
        )
        
    def contour_plot(self, show_labels = True, show_grid = True, **kwargs):
        xxx, yyy = [np.linspace(par[0], par[-1], 50) for par in self.params] 
        fff = np.array([[ self([x,y]) for y in yyy ] for x in xxx]).T
        cp = pl.contour( xxx, yyy, fff, **kwargs)
        
        if show_labels:
            pl.clabel(cp)
        
        if show_grid:
            pl.scatter(*zip(*itt.product(*self.params)), marker = "+")
        
        return cp
            
    
    def integrate(self):
        coef = self.coef
        for i in range(self.var_ndim):
            coef = get_1D_integral_of_piecewise_poly(
                coef = coef, 
                bin_axis = self.batch_ndim, 
                polynom_axis = self.batch_ndim + self.var_ndim - i , 
                control_times = self.params[i]
            )
            
        return coef
    
    
class TaylorGrid(Batched_Object, Val_Indexed_Object, TF_Runnable):
    def __init__(self, coef, params, batch_ndim = 0,  val_ndim = 0):
        self.coef = coef
        self.batch_ndim = batch_ndim
        self.params = params
        self.val_ndim = val_ndim
        
        assert len(self.coef.shape) == self.batch_ndim + 2 * self.var_ndim + self.val_ndim
        
        grid_shape_from_coef = list(self.coef.shape[self.batch_ndim: self.batch_ndim + self.var_ndim])
        grid_shape_from_params = list(self.grid_shape)
        assert grid_shape_from_coef == grid_shape_from_params, (
            "The coeficients suggest grid shape " + str(grid_shape_from_coef) +
            " while the one inferred from params is " + str(grid_shape_from_params)
        )
    
    def __repr__(self):
        s = "TaylorGrid( " 
        s += "var_ndim = " + str(self.var_ndim)
        s += ", params.lengts = " + str([len(par) for par in self.params]) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += ", coef.shape = " + str(self.coef.shape) 
        
        s += " )"
            
        return s
        
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return TaylorGrid(
            coef = coef,
            params = self.params,
            batch_ndim = batch_ndim, 
            val_ndim = self.val_ndim            
        )
        
    def _val_getitem(self, selector):
        if not isinstance(selector, tuple):
            selector = (selector,)
            
        coef = self.coef[
            (slice(None), )*(self.batch_ndim + 2 * self.var_ndim)
            + selector
        ]
        
        val_ndim = self.val_ndim - (len(self.coef.shape) - len(coef.shape))
        assert val_ndim >= 0, "val_ndim = " + str(val_ndim)
        return TaylorGrid(
            coef = coef, 
            params = self.params,
            batch_ndim = self.batch_ndim, 
            val_ndim = val_ndim            
        )
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return TaylorGrid(
                coef = data,
                params = self.params,
                batch_ndim = self.batch_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
    
    @property
    def batch_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[:self.batch_ndim]
    
    @property
    def val_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[self.batch_ndim + 2 *self.var_ndim:]
    
    @property
    def var_ndim(self):
        return len(self.params)
    
    @property
    def degs(self):
        return self.coef.shape[self.batch_ndim + self.var_ndim: self.batch_ndim + 2 *self.var_ndim]
    
    @property
    def grid_shape(self):
        return [len(par) for par in self.params]
    
    @property
    def bins_shape(self):
        return [len(par) - 1 for par in self.params]
    
    def __mul__(self, other, truncation = None):
        if truncation is None:
            truncation = np.minimum(self.degs, getattr(other, "degs", 0))
            
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, TaylorGrid) else other
        return f.__mul__(g, truncation=truncation).to_TaylorGrid(self.params)

    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, TaylorGrid) else other
        return f.__add__(g).to_TaylorGrid(self.params)
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
        
    def __truediv__(self, other):
        return 1/other * self
        
    def __rtruediv__(self, other):
        return self.truncated_inverse() * other
        
    def __pow__(self, other):
        return self.truncated_power(other)
        
    def truncate_degs(self, limit_degs):
        f = self.to_Poly()
        return f.truncate_degs(limit_degs).to_TaylorGrid(self.params)
    
    def pad_degs(self, minimal_degs):
        f = self.to_Poly()
        return f.pad_degs(minimal_degs).to_TaylorGrid(self.params)
        
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        f = self.to_Poly()
        return f.val_mul(a).to_TaylorGrid(self.params)
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `TaylorGrid` return `TaylorGrid` whose values are sums of the values of the original.
        """
        f = self.to_Poly()
        return f.val_sum(axis).to_TaylorGrid(self.params)
        
        
    
    def to_Poly(self):
        return Poly(
            coef = self.coef, 
            batch_ndim=self.batch_ndim + self.var_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim
        )
    
    def get_poly_at_node(self, ii):
        ii = tuple([None] * self.batch_ndim + list(ii))
        return Poly(coef = self.coef[ii], batch_ndim = self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim )
    
    def get_node(self, ii):
        return np.array([par[i] for i, par in zip(ii, self.params) ])
    
    def grid_indices(self):
        return itt.product(*[range(len(par)) for par in self.params])
    
    
    def plot(self, **kwargs):
        for ii in self.grid_indices():
            this_node = self.get_node(ii)
            previous_node = self.get_node([max(0, i-1) for i in ii])
            next_node = self.get_node([min(i+1, len(par) - 1) for i,par in zip(ii, self.params)])
            
            poly = self.get_poly_at_node(ii)
            poly = poly.shift(this_node)
            poly.plot(
                start = 2/3 * this_node + 1/3 * previous_node,
                end = 2/3 * this_node + 1/3 * next_node,
                **kwargs
            )
            
    def get_spline(self):
        coef = self.coef
        
        for i, par in enumerate(self.params):
            coef = get_spline_from_taylors_1D(
                taylor_grid_coeffs = coef, 
                bin_axis = self.batch_ndim + i, 
                polynom_axis = self.batch_ndim + self.var_ndim + i, 
                control_times = par
            )
            
        return PolyMesh(
            coef = coef, params = self.params, batch_ndim=self.batch_ndim, val_ndim=self.val_ndim
        )
        
    def integrate_spline(self):
        coef = self.coef
        for i in range(self.var_ndim):
            coef = get_integral_of_spline_from_taylors_1D(
                taylor_grid_coeffs= coef, 
                bin_axis = self.batch_ndim, 
                polynom_axis = self.batch_ndim + self.var_ndim - i , 
                control_times = self.params[i]
            )
            
        return coef

    
    def truncated_exp(self, degs = None):
        f = self.to_Poly()
        return f.truncated_exp(degs).to_TaylorGrid(self.params)
    
    def truncated_inverse(self, degs = None):
        f = self.to_Poly()
        return f.truncated_inverse(degs).to_TaylorGrid(self.params)
    
    def truncated_power(self, exponent, degs = None):
        """Return `self` raised to the exponent (possibly real) truncated to `degs`. 
        """
        f = self.to_Poly()
        return f.truncated_power(
                    exponent = exponent, degs = degs
            ).to_TaylorGrid(self.params)
        
    
    
    @staticmethod
    def from_Catmull_Rom(coef, params, batch_ndim = 0, val_ndim = None):
        if val_ndim is None:
            val_ndim = len(coef.shape) - batch_ndim - len(params)
            
        for i, par in enumerate(params):
            coef = get_Catmul_Rom_Taylors_1D(
                coef = coef,
                control_index = batch_ndim + i,
                control_times = par,
                added_index = batch_ndim + len(params) + i,
            )
            
        return TaylorGrid(coef, params, batch_ndim= batch_ndim, val_ndim=val_ndim)
    
    @staticmethod
    def from_Gauss_pdf(params, mu, K, batch_ndim = None, var_ndim = None):
        if batch_ndim is None:
            batch_ndim = nptf.ndim(mu) - 1

        if var_ndim is None:
            var_ndim = int(mu.shape[-1])

        assert (
            (batch_ndim == nptf.ndim(mu) - 1 == nptf.ndim(K) - 2) and
            (var_ndim == mu.shape[-1] == K.shape[-1] == K.shape[-2] )
        ),("Problem with dimensions: mu.shape =" + str(mu.shape) + 
           ", K.shape = " + str(K.shape) + ", batch_ndim = " + str(batch_ndim) +
           ", var_ndim = " + str(var_ndim)
        )

        exponent = -1/2 * (
            Poly.from_tensor(K, batch_ndim=batch_ndim) + 
            (-2) * Poly.from_tensor(
                (K @ mu[..., None])[..., 0], 
                batch_ndim=batch_ndim,
                var_ndim = var_ndim
            )  +
            Poly.from_tensor(
                (mu[..., None,:] @ K @ mu[..., None])[...,0, 0], 
                batch_ndim=batch_ndim,
                var_ndim = var_ndim
            )     
        )

        tg = exponent.get_Taylor_grid(params = params, truncs = 2)
        tg = tg.truncated_exp()

        const = (2 * np.pi) ** (-tg.var_ndim / 2) * nptf.det(K)**(1/2)
        tg *= const

        return tg
    
    
      