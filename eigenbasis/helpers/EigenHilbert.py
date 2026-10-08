# import eigenbasis
# from eigenbasis import cluster
 
from typing import Optional
from fractions import Fraction
from pathlib import Path
import os
from datetime import datetime, timedelta
import warnings
import jax
import numpy as np
import netket as nk
from netket import config as nkconfig
from netket.utils import StaticRange, mpi, struct
from netket.errors import UndeclaredSpinOderingWarning
from netket.hilbert.homogeneous import HomogeneousHilbert
#from netket.hilbert.constraint import DiscreteHilbertConstraint, SumConstraint
from netket.utils.types import Scalar, Array
from netket.hilbert.constraint.base import DiscreteHilbertConstraint
import jax.numpy as jnp


 
class EigenConstraint(DiscreteHilbertConstraint):
    """
    Constraint of an Hilbert space enforcing a total sum of all the values in the degrees of freedom.
    Constructed by specifying the total sum. For Fock-like spaces this is the total population,
    while for Spin-like spaces this is the magnetisation.
    """
    sum_value: Scalar = struct.field(pytree_node=False)
    reference_values: Array = struct.field(pytree_node=False)
    def __init__(self, sum_value: Scalar, reference_values:np.ndarray):
        if sum_value is None:
            raise TypeError("sum_value must be a number.")
 
        self.reference_values = jax.numpy.array(reference_values)
        self.sum_value = sum_value
        # print('self.sum_value = ', self.sum_value)
        # print(self.reference_values)
 
#    @jax.jit

    def __call__(self, x: Array) -> Array:
        x = x.astype(int)
        sz = jax.numpy.sum(self.reference_values.at[x].get(),axis=-1)
        # print('sz (calc) = ', sz)
        # print(sz.shape)
        # print(sz.dtype)
        # return sz == self.sum_value
        return abs(sz - self.sum_value) < 1e-9

#        return x.sum(axis=-1) == self.sum_value
 
    def __hash__(self):
        return hash(("EigenConstraint", self.sum_value))
 
    def __eq__(self, other):
        if isinstance(other, EigenConstraint):
            return self.sum_value == other.sum_value
        return False
 
    def __repr__(self):
        return f"EigenConstraint({self.sum_value})"
 
        

class EigenHilbert(HomogeneousHilbert):

    def __init__(
        self,
        N: int = 1,
        Nlocal: int | None = None,
        total_sz: float | None = None,
        Sz_operator: np.ndarray | None = None, 
        constraint: DiscreteHilbertConstraint | None = None,
    ):

    #build =arrays with the eigenstates quantum numbers and theori energies.

#        basis, ops = eigenbasis.cluster(Nlocal)

        # self.basis, self.ops = cluster(Nlocal)
        nk.config.netket_random_state_fallback_warning = False
        self.Sz = Sz_operator
        local_size = Sz_operator.shape[0]
        self.Nlocal = Nlocal
        local_states = StaticRange(0,1,local_size,dtype=np.int8)
        # print(Sz_operator)

        if total_sz is not None:
            if constraint is not None:
                raise ValueError(
                    "Cannot specify at the same time a total magnetization "
                    "constraint and a `custom_constraint."
                )

            constraint = EigenConstraint(total_sz,self.Sz.diagonal())

        self._total_sz = total_sz

        super().__init__(local_states, N, constraint=constraint)

    def __pow__(self, n):

        return NotImplemented

    def _mul_sametype_(self, other):

        return NotImplemented

    def ptrace(self, sites: int | list) -> Optional["Eigen"]:

        return None

    def __repr__(self):

        if self._total_sz is not None:

            constraint = f", total_sz={self._total_sz}"

        elif self.constrained:

            constraint = f", {self._constraint}"

        else:

            constraint = ""

        return f"Eigen(N={self.size}, Nlocal={self.Nlocal}, total_sz={self._total_sz}, ordering={constraint})"

    @property

    def _attrs(self):

        return (self.size, self.constraint)
    

 