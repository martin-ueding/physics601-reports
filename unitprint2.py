#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2012-2013 Martin Ueding <dev@martin-ueding.de>
# Licensed under The MIT License

'''
Pretty prints numerical value and error pairs.

This is especially handy for lab reports or similar written work where a lot of
measurements are presented. The siunitx() function will generate output
suitable for the LaTeX package siunitx and its ``\\SI`` and ``\\num`` commands.

It will check the following:

- Value and error have the same number of significant digits.
'''

import math

__docformat__ = "restructuredtext en"

class Quantity(object):
    def __init__(self, value, error=None, digits=3, error_digits=1, allowed_hang=3):
        '''
        :param allowed_hang: If the value exponent is below or equal to this
        number, it will printed like that. So a 100 is preserved if this is set
        to 3. A 1000 will be converted to ``1.00e3``.
        '''
        if value == 0:
            value_log = 0
        else:
            value_log = int(math.floor(math.log(abs(value), 10)))

        if error is None or error == 0:
            if abs(value_log) > allowed_hang:
                self.value_mantissa = ("{:."+str(digits-1)+"f}").format(value * 10**(- value_log))
                self.error_mantissa = None
                self.exponent = value_log
            else:
                self.value_mantissa = ("{:."+str(max(digits-1 - value_log, 0))+"f}").format(value)
                self.error_mantissa = None
                self.exponent = 0
        else:
            error_log = int(math.floor(math.log(abs(error), 10)))

            difference = value_log - error_log

            value_dis = value * 10**(- value_log)
            error_dis = error * 10**(-difference - error_log)
            exp = value_log

            if abs(value_log) > allowed_hang:
                here_digits = error_digits - 1 + max(difference, 0)

                self.value_mantissa = ("{:."+str(here_digits)+"f}").format(value_dis)
                self.error_mantissa = ("{:."+str(here_digits)+"f}").format(error_dis)
                self.exponent = exp
            else:
                here_digits = max(error_digits - 1 -error_log, 0)

                self.value_mantissa = ("{:."+str(here_digits)+"f}").format(value)
                self.error_mantissa = ("{:."+str(here_digits)+"f}").format(error)
                self.exponent = 0

    def to_siunitx(self):
        if self.error_mantissa is None:
            if self.exponent == 0:
                return "{}".format(self.value_mantissa)
            else:
                return "{}e{}".format(self.value_mantissa, self.exponent)
        else:
            if self.exponent == 0:
                return "{} +- {}".format(self.value_mantissa, self.error_mantissa)
            else:
                return "{} +- {} e{}".format(self.value_mantissa, self.error_mantissa, self.exponent)

def siunitx(value, error=None, **kwargs):
    '''
    Convenience function for generating output for the LaTeX siunitx package.

    The given parameters will be used to generate a Quantity object. If
    ``value`` is an iterable object, a Quantity object will be generated for
    each item. That way, it is possible to use this function on two numpy.array
    instances.

    :type value: int or list
    :type error: int or list
    '''
    if hasattr(value, "__iter__"):
        if error is None:
            return [Quantity(v, None, **kwargs).to_siunitx() for v in value]
        else:
            return [Quantity(v, e, **kwargs).to_siunitx() for v, e in zip(value, error)]
    else:
        q = Quantity(value, error, **kwargs)
        return q.to_siunitx()

def format(value, error=None, unit=None, lit=None, latex=False):
    """
    Formats the given value and error in a human readable form. If an error is
    supplied, it will calculate the relative error. If a literature value is
    given, the deviation from the canonical value is calculated and the error
    is given as a ratio and in the number of standard deviations.

    :param value: Value itself
    :type value: float
    :param error: Error of the value
    :type error: None or float
    :param unit: Physical unit
    :type unit: None or str
    :param lit: Canonical value
    :type lit: None or float
    :return: Formatted output
    :rtype: str
    """

    parts = []

    if unit is not None:
        parts.append(unit)

    if error is not None:
        parts.append("({:.0%})".format(error/value))

    if lit is not None:
        lit_parts = []
        lit_parts.append("{:+.0%}".format((value-lit)/lit))
        if error is not None:
            lit_parts.append("{:+.1f}σ".format((value-lit)/error))
        parts.append("[" + ", ".join(lit_parts) + "]")

    return ' '.join(parts)

# vim: spell
