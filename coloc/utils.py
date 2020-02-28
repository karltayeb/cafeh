from functools import lru_cache, wraps
import numpy as np
from scipy.special import gamma

def gamma_logpdf(x, alpha, beta):
    alpha * np.log(beta) - gamma(alpha) + (alpha - 1) * np.log(x) - beta * x

def np_cache(*args, **kwargs):
    ''' LRU cache implementation for methods whose FIRST parameter is a numpy array
        modified from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            hashed = array_to_hashable(*args)
            return cached_wrapper(hashed, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashed, **kwargs):
            return function(*hashable_to_array(hashed), **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator

def np_cache_class(*args, **kwargs):
    ''' LRU cache implementation for methods whose FIRST parameter is a numpy array
        modified from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            hashed = array_to_hashable(*args)
            return cached_wrapper(self, hashed, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(self, hashed, **kwargs):
            return function(self, *hashable_to_array(hashed), **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator

def pack(*args):
    """
    take a list of numpy arrays, pack them into a single array
    return single flat numpy array
    return list of shape tuples for unpacking
    """
    
    flat = np.concatenate([x.flatten() for x in args])
    size = [0]
    size.extend([x.size for x in args])
    size = tuple(np.cumsum(size))
    shapes = tuple([x.shape for x in args])
    return flat, size, shapes

def unpack(flat, size, shape):
    return (flat[size[i]:size[i+1]].reshape(shape[i]) for i in range(len(shape)))

def array_to_hashable(*args):
    """
    take list of arrays
    return ((bytes, shape) for each array)
    """
    return tuple((a.tobytes(), a.shape) for a in args)

def hashable_to_array(hashable_tuple):
    return [np.frombuffer(a[0]).reshape(a[1]) for a in hashable_tuple]