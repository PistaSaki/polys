import numpy as np

import keras
from keras.engine.topology import Layer
import keras.backend as K

from polys import TaylorGrid


class CatmullRomLayer(Layer):
    def __init__(
        self, controls, val_shape = None, mean = None , **kwargs
    ):
        """
        Args:
            controls: list of 1D numpy arrays
        """
        assert K.backend() == 'tensorflow'
        
        assert isinstance(controls, list)
        self.controls = [np.array(c, dtype = K.floatx()) for c in controls]
        
        if mean is not None:
            assert mean.shape == tuple([len(c) for c in self.controls])
            if val_shape is not None:
                assert tuple(val_shape) == mean.shape[len(self.controls):]
        
        if mean is None:
            assert val_shape is not None
            mean = np.zeros([len(c) for c in self.controls] + list(val_shape))
            
        self.mean = np.array(mean, dtype = K.floatx())
        
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(self.controls) == input_shape[1]
        
        self.control_values = self.add_weight(
            name = "control_values",
            shape=self.mean.shape,
            initializer= keras.initializers.RandomNormal(self.mean),
            trainable=True
        )
        
        self.tg = TaylorGrid.from_Catmull_Rom(
            coef = self.control_values, 
            params = self.controls,
            batch_ndim = 0,
            val_ndim = self.mean.ndim - len(self.controls)
        )
        
        self.spline = self.tg.get_spline()
        
        super().build(input_shape)  # Be sure to call this at the end
    
    def call(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == len(self.controls)
        return self.spline.batch[None](x)
