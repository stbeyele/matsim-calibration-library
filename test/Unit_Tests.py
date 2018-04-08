from utilities.error_utilities import Error_Calculator


def run_error_test():
    error_calculation = Error_Calculator([1, 2, 3], 'Freude')
    test = error_calculation.calc_error([3, 2, 4])
