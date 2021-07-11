# HIF: The Hubris/Humility Interchange Format

When debugging Hubris, we have found it handy to get a task to execute as
a proxy for an external task to (e.g.) perform I2C transactions.  This was
born as an ad hoc facility, effected done by the debugger stopping the
target and writing parameters to well-known memory locations, but as our
need for proxy behavior became more sophisticated, it became clear that we
needed something more general purpose.  HIF represents a simple,
stack-based machine that can be executed in a memory-constrained no-std
environment.  While this was designed to accommodate Hubris and Humility,
there is nothing specific to either, and may well have other uses.

