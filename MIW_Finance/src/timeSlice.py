
# built-in libraries
import matplotlib.pyplot as plt
import numpy as np


class TimeSlice:

    def __init__(self, tau: float, positions_list: list, values_list: list, default_value = 0.):
        if len(positions_list) != len(values_list) :
            f"values_list does not follow the dimensions of the positions list, " \
                f"using default value {default_value} instead"
            values_list = [default_value for _ in positions_list]
        self._tau: float = tau          # T-t
        self._data = list( zip(positions_list,values_list) )
        self._length = len(self._data)

    def __getitem__(self, i):
        """
        Square-brackets return component
        """
        return self._data[i]
    def __setitem__(self, i, tuple2):
        """
        Square-brackets set component
        """
        self._data[i] = tuple2
    def __repr__(self):
        output = f"At time tau={self._tau}:\n"
        for datum in self._data:
            output += f"{datum[0]:16.2f} : {datum[1]:6.2f} \n"
        return output

    @property
    def tau(self):
        return self._tau
    @property
    def data(self):
        return self._data

    @property
    def positions(self):
        return np.array([datum[0] for datum in self._data])
    @positions.setter
    def positions(self, other):
        if len(other) != len(self._data) :
            raise ValueError
        for idx, pos in enumerate(other):
            self._data[idx] = (pos, self.value_n(idx))

    @property
    def values(self):
        return np.array([datum[1] for datum in self._data])
    @values.setter
    def values(self, other):
        if len(other) != len(self._data) :
            raise ValueError
        for idx, val in enumerate(other):
            self._data[idx] = (self.position_n(idx), val)

    def position_n(self,n):
        return (self._data[n])[0]
    def value_n(self,n):
        return (self._data[n])[1]

    def slope_at_point_n(self, n):
        if n == 0 :
            return ( self.value_n(n+1) - self.value_n(n) ) / ( self.position_n(n+1) - self.position_n(n) )
        elif n == self._length-1 :
            return ( self.value_n(n) - self.value_n(n-1) ) / ( self.position_n(n) - self.position_n(n-1) )
        # else
        return ( self.value_n(n+1) - self.value_n(n-1) ) / ( self.position_n(n+1) - self.position_n(n-1) )

    def acceleration_at_point_n(self, n):
        if n == 0 :
            delta1 = self.position_n(n+1) - self.position_n(n)
            delta2 = self.position_n(n+2) - self.position_n(n)
            norm = delta1*delta2*(delta2-delta1)/2
            return ( delta1*self.value_n(n+2) - delta2*self.value_n(n+1) + (delta2-delta1)*self.value_n(n) )/norm
        elif n == self._length-1 :
            delta1 = self.position_n(n) - self.position_n(n-1)
            delta2 = self.position_n(n) - self.position_n(n-2)
            norm = (delta2-delta1)/2
            return ( self.value_n(n-2)/delta2 - self.value_n(n-1)/delta1
                     + (delta2-delta1)*self.value_n(n)/(delta1*delta2) )/norm
        # else
        delta1 = self.position_n(n+1) - self.position_n(n)
        delta2 = self.position_n(n) - self.position_n(n-1)
        norm = delta1*delta2*(delta2+delta1)/2
        return ( delta2*self.value_n(n+1) + delta1*self.value_n(n-1) - (delta1+delta2)*self.value_n(n) ) / norm

    def get_value_at_position(self, S:float) :
        if S > self.position_n(-1) or S < self.position_n(0):
            print(f"Cannot interpolate outside of region {self.position_n(0)}--{self.position_n(-1)}")
        # find position_n which comes before S
        idx, pos = 0, 0.
        while True :
            testidx = idx+1
            testpos = self.position_n(testidx)
            if testpos > S :
                break
            idx = testidx
            pos = testpos

        return self.value_n(idx) + (S-pos)*self.slope_at_point_n(idx) + (S-pos)**2*self.acceleration_at_point_n(idx)/2


    def print_velocities(self):
        for idx in range(self._length):
            print(f"dV/dS at pos {self.position_n(idx):5.1f}:  {self.slope_at_point_n(idx):5.1f}")

    def print_accelerations(self):
        for idx in range(self._length):
            print(f"d^2V/dS^2 at pos {self.position_n(idx):5.1f}:  {self.acceleration_at_point_n(idx):5.1f}")

    def plot_slice(self):
        plt.plot(self.positions,self.values)
        plt.show()

    def validate_monotonic_increasing(self):
        for idx, val in enumerate( self.values[1:]) :
            n = idx+1
            if self.value_n(n) - self.value_n(n-1) < 0 :
                print(f"{n=} {self.position_n(n)=} {self.value_n(n-1)=} {self.value_n(n)=}")
                # self.plot_slice()
                return False
        return True







