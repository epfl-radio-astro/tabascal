"""How much of the GPU a tabascal command claims up front.

JAX preallocates 75 % of the device on the first operation. That is the right
default for a job that owns the card and the wrong one for everything else: a
run cannot share a GPU with anything, and a command that only copies a zarr
into a Measurement Set takes three quarters of the device to do it. Every
tabascal entry point therefore asks for memory on demand instead.

``setdefault``, never assignment. Preallocation minimises fragmentation, so a
long run on a card it owns is exactly the case for turning it back on, and
``XLA_PYTHON_CLIENT_PREALLOCATE=true`` is how JAX says to do that. Assigning
over it -- which ``_run_tabascal_impl`` used to do at import, a few lines after
the CLI had already deferred to the user -- takes that choice away silently.

Imports nothing but ``os``: it is called before the parser on entry points that
must not pay for the JAX import to answer ``-h``.
"""

import os

#: JAX's own switch. Unset means we supply on-demand; set means the user has
#: chosen and we leave it alone.
PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"


def default_memory_on_demand() -> None:
    """Ask JAX for memory on demand unless the user has said otherwise.

    Call it at the top of an entry point's ``main()``, before the first import
    that could bring the device backend up -- the variable is read when the
    backend initialises, and setting it afterwards does nothing at all.
    """

    os.environ.setdefault(PREALLOCATE_ENV, "false")
