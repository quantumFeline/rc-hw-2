import numpy as np

class PID:
    def __init__(
            self, gain_prop: float, gain_int: float, gain_der: float, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        """
        :param gain_prop: initial proportional gain
        :param gain_int: initial integral gain
        :param gain_der: initial derivative gain
        :param sensor_period: length of a single timestep between the sensor measurements
        :param output_limits: a tuple of (min, max) output values
        """
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.output_limits = output_limits
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        """
        Calculate the output signal to the motors based on the sensor readings and the previous information.
        If the output signal is outside the allowed range, it must be clamped between output_limits values.
        :param commanded_variable:
        :param sensor_readings:
        :return: the output signal
        """
        output = np.clip(commanded_variable, self.output_limits[0], self.output_limits[1])
        return output
    # END OF TODO
