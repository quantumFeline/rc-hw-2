import numpy as np

class PID:
    def __init__(
            self, gain_prop: float, gain_int: float, gain_der: float, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        """
        :param gain_prop: coefficient of the proportional gain
        :param gain_int: coefficient of the integral gain
        :param gain_der: coefficient of the derivative gain
        :param sensor_period: length of a single timestep between the sensor measurements
        :param output_limits: a tuple of (min, max) output values
        """
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.current_sum = 0 # For integral component.
        self.current_difference = 0 # For derivative component.
        self.output_limits = output_limits
        self.fadeout = 0.97
        # END OF TODO

    def update_components(self, error: list[float]):
        """
        Update the PID components.
        :param error: errors of the PID component
        """
        if len(error) == 2:
            current_error = error[0]
            previous_error = error[1]
            self.current_sum = self.current_sum * self.fadeout + current_error * self.sensor_period
            self.current_difference = (current_error - previous_error) / self.sensor_period
        else:
            raise NotImplemented("Non-linear derivative calculation is not yet implemented")

    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        """
        Calculate the output signal to the motors based on the sensor readings and the previous measurements.
        If the output signal is outside the allowed range, it must be clamped between output_limits values.
        :param commanded_variable: desired value of the commanded variable.

        :param sensor_readings: list of the current and previous values, in reverse historical order.
        :return: the output signal for the commanded variable.
        """
        current_error = commanded_variable - sensor_readings[0]
        previous_error = commanded_variable - sensor_readings[1]
        self.update_components([current_error, previous_error])

        # Scale
        proportional = self.gain_prop * current_error
        derivative = self.gain_der * self.current_difference
        integral = self.gain_int * self.current_sum

        output = proportional + derivative + integral
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        return output
    # END OF TODO
