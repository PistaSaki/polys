import numpy as np
import ipywidgets as ipw

from pitf import tf_runnables as rb

def get_common_broadcasted_shape(batch_shapes):
    ndim = len(batch_shapes[0])
    assert all([ndim == len(shape) for shape in batch_shapes]), "All batches should have the same noof dimensions/indices."
    batch_shape = np.array(batch_shapes).max(axis = 0)
    assert all([shape[i] in {1, batch_shape[i]} for shape in batch_shapes for i in range(len(batch_shape))]), (
        "Can not broadcast batch shapes " + str(batch_shapes) 
    )
    
    return list(batch_shape)

##Test:
#get_common_broadcasted_shape([[2, 1], [1, 4]])
    
def unbroadcast_index(ii, shape):
    return tuple([i if dim>1 else 0 for i, dim in zip(ii, shape)])

##Test:
#unbroadcast_index([10, 2], [2, 1])

def interact_along_batch(batch_shape, display_fun = None,):
    if display_fun is None:
        def display_fun(iii):
            print("batch:", iii)
    
    batch_index_sliders = {
        "i" + str(i): ipw.IntSlider(value = 0, min = 0, max = dim-1)
        for i, dim in enumerate(batch_shape)
    }
    
    def fun(**indices_in_dic):
        indices = [indices_in_dic[k] for k in sorted(indices_in_dic.keys())]
        display_fun(tuple(indices))
        
    ipw.interact(
        fun,
        **batch_index_sliders
    )
 
##TEST:
#interact_along_batch([2, 3, 4])    

def display_tensors_along_batch(tensors, batch_ndim = 1, 
                         display_fun = None,
                         sess = None, feed_dict = None,):
    
    """
    `tensors` is a list of tensors (np.arrays or tensorflow tensors).
    `display_fun` takes two arguments: a list `batch_tensors` and a tuple of ints `batch_index`.
    """
    
    tensors = rb.run_in_session(tensors, sess = sess, feed_dict=feed_dict)
    
    if display_fun is None:
        def display_fun(batch_tensors, batch_index):
            print("batch:", batch_index)
            for t in batch_tensors:
                print(t)          
                
    if batch_ndim == 0:
        display_fun(Ellipsis)
        return
                
    interact_along_batch(
        batch_shape = get_common_broadcasted_shape([t.shape[:batch_ndim] for t in tensors]),
        display_fun = lambda iii: display_fun(
            batch_tensors = [t[unbroadcast_index(iii, t.shape)] for t in tensors],
            batch_index = iii
        )
    )
        
    
## Example
#a = np.arange(24).reshape([3, 2, 4])
#b = np.arange(3)[:, None, None]
#display_tensors_along_batch(
#    tensors =[
#        a, b, a * b
#    ],
#    batch_ndim = 2
#)
    

class Batch_Indexer:
    def __init__(self, obj):
        self.obj = obj
        
    def __getitem__(self, selector):
        return self.obj._get_batch(selector)

class Batched_Object:
    @property
    def batch(self):
        return Batch_Indexer(obj = self)
    
    def print_over_batch(self):
        interact_along_batch(
            batch_shape = self.batch_shape, 
            display_fun = lambda ii: print(self.batch[ii])
        )
        
    def plot_over_batch(self):
        interact_along_batch(
            batch_shape = self.batch_shape, 
            display_fun = lambda ii: self.batch[ii].plot()
        )

def print_batched_objects(objs, names = None):
    if names is None:
        names = [ "obj_" + str(i) for i, o in enumerate(objs) ]
        
    def fun(ii):
        print("batch_index =", ii)
        for name, obj in zip(names, objs):
            print( name, "=", obj.batch[ii])
            
    interact_along_batch(
            batch_shape = get_common_broadcasted_shape([obj.batch_shape for obj in objs]), 
            display_fun = fun
        )
    

