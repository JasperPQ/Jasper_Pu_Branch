import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        """Return the node value for the piecewise-constant left-continuous interpolant.

        Convention:
        - flat extrapolation outside the axis range
        - for x exactly on a node, return that node's value
        - otherwise on the interval (axis[i-1], axis[i]], use values[i]
        """
        x = float(x)

        if x <= self.axis1_[0]:
            return float(self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.values_[-1])

        exact_match = np.isclose(self.axis1_, x)
        if np.any(exact_match):
            idx = int(np.where(exact_match)[0][0])
            return float(self.values_[idx])

        idx = int(np.searchsorted(self.axis1_, x, side='right'))
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        """Integrate the piecewise-constant interpolant over [start_x, end_x]."""
        if end_x < start_x:
            return -self.integrate(end_x, start_x)

        total = 0.0

        # Left flat wing.
        if start_x <= self.axis1_[0]:
            left_overlap = min(end_x, self.axis1_[0]) - start_x
            if left_overlap > 0:
                total += float(self.values_[0]) * left_overlap
            start_x = max(start_x, self.axis1_[0])

        # Right flat wing.
        if end_x >= self.axis1_[-1]:
            right_overlap = end_x - max(start_x, self.axis1_[-1])
            if right_overlap > 0:
                total += float(self.values_[-1]) * right_overlap
            end_x = min(end_x, self.axis1_[-1])

        # Interior buckets: value is constant on (axis1[i-1], axis1[i]] with the level at values[i].
        for i in range(1, len(self.axis1_)):
            left = self.axis1_[i - 1]
            right = self.axis1_[i]
            overlap_left = max(start_x, left)
            overlap_right = min(end_x, right)
            if overlap_right > overlap_left:
                total += float(self.values_[i]) * (overlap_right - overlap_left)

        return float(total)

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        """Gradient of the interpolated value with respect to the ordinate values."""
        grad = np.zeros(len(self.values_), dtype=float)

        if x <= self.axis1_[0]:
            grad[0] = 1.0
            return grad
        if x >= self.axis1_[-1]:
            grad[-1] = 1.0
            return grad

        exact_match = np.isclose(self.axis1_, x)
        if np.any(exact_match):
            idx = int(np.where(exact_match)[0][0])
            grad[idx] = 1.0
            return grad

        idx = int(np.searchsorted(self.axis1_, x, side='right'))
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        """Gradient of the integral over [start_x, end_x] with respect to the ordinate values."""
        if end_x < start_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(end_x, start_x)

        grad = np.zeros(len(self.values_), dtype=float)

        if start_x <= self.axis1_[0]:
            overlap = min(end_x, self.axis1_[0]) - start_x
            if overlap > 0:
                grad[0] += overlap
            start_x = max(start_x, self.axis1_[0])

        if end_x >= self.axis1_[-1]:
            overlap = end_x - max(start_x, self.axis1_[-1])
            if overlap > 0:
                grad[-1] += overlap
            end_x = min(end_x, self.axis1_[-1])

        for i in range(1, len(self.axis1_)):
            left = self.axis1_[i - 1]
            right = self.axis1_[i]
            overlap_left = max(start_x, left)
            overlap_right = min(end_x, right)
            if overlap_right > overlap_left:
                grad[i] += overlap_right - overlap_left

        return grad


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')