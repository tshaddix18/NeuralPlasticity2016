import numpy as np
import sympy

from brian2.core.functions import DEFAULT_FUNCTIONS, Function

if 'sign' not in DEFAULT_FUNCTIONS:
    # Only a temporary workaround until this code is merged into the main
    # Brian repository
    DEFAULT_FUNCTIONS['sign'] = Function(pyfunc=np.sign, sympy_func=sympy.sign,
                                         arg_units=[None], return_unit=1)
    # Implementation for C++
    sign_code = '''
    template <typename T> int sign_(T val) {
         return (T(0) < val) - (val < T(0));
    }
    '''
    DEFAULT_FUNCTIONS['sign'].implementations.add_implementation('cpp',
                                                                 code=sign_code,
                                                                 name='sign_')

    # Implementation for Cython
    sign_code = '''
    ctypedef fused _to_sign:
        char
        short
        int
        float
        double

    cdef int _int(_to_sign x):
        return (0 < x) - (x < 0)
    '''
    DEFAULT_FUNCTIONS['sign'].implementations.add_implementation('cython',
                                                                 code=sign_code,
                                                                 name='_sign')

if 'pknorm' not in DEFAULT_FUNCTIONS:
    # Only a temporary workaround until this code is merged into the main
    # Brian repository
    pknorm = lambda peak, taur, taud : peak*(1./taud - 1./taur)/( (taur/taud)**(taud/(taud-taur))-(taur/taud)**(taur/(taud-taur)) )
    DEFAULT_FUNCTIONS['pknorm'] = Function(pyfunc=pknorm,
                                           arg_units=[1, second, second], return_unit=Hz)
    # # Implementation for C++
    # sign_code = '''
    # template <typename T> int sign_(T val) {
    #      return (T(0) < val) - (val < T(0));
    # }
    # '''
    # DEFAULT_FUNCTIONS['sign'].implementations.add_implementation('cpp',
    #                                                              code=sign_code,
    #                                                              name='sign_')
    #
    # # Implementation for Cython
    # sign_code = '''
    # ctypedef fused _to_sign:
    #     char
    #     short
    #     int
    #     float
    #     double
    #
    # cdef int _int(_to_sign x):
    #     return (0 < x) - (x < 0)
    # '''
    # DEFAULT_FUNCTIONS['sign'].implementations.add_implementation('cython',
    #                                                              code=sign_code,
    #                                                              name='_sign')
    #
