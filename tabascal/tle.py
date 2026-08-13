"""Compatibility imports for the renamed :mod:`tabascal.orbit` module.

New code should import orbit-record orchestration from :mod:`tabascal.orbit`.
This module remains so existing callers of the historical TLE-only API do not
break while TABASCAL supports both TLE and OMM records.
"""

from tabascal.orbit import *  # noqa: F401,F403
from tabascal import orbit as _orbit


def __getattr__(name):
    """Forward historical private imports during the compatibility period."""
    return getattr(_orbit, name)
