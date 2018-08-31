"""
Pre-defined classes without crucial functions.
"""


class SeriesNotFoundError(Exception):
    """
    The Error Risen to indicate that time series asked cannot be load.
    """
    pass


class PanelNotFoundError(Exception):
	pass


class PanelFailure(Exception):
	pass