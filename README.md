# hif

## HIF: The Hubris/Humility Interchange Format

When debugging Hubris, we have found it handy to get a task to execute as
a proxy for an external task.  This was born as an ad hoc facility to
induce I2C transactions, effected by the debugger stopping the target and
writing parameters to well-known memory locations; as our need for proxy
behavior became more sophisticated, it became clear that we needed
something more general purpose.  HIF represents a simple, stack-based
machine that can be executed in a memory-constrained no-std environment.
While this was designed to accommodate Hubris and Humility, there is
nothing specific to either, and may well have other uses.

### Machine model

The HIF machine model consists of program text consisting of an array of
`Op` structures denoting operations; a stack of 32-bit values for
arguments and variables; a finite number of branch targets corresponding
to declared labels in program text; and an append-only return stack that
consists of an array of `FunctionResult` enums.  Of note, this model has
no registers (though such extensions are clearly possible).

### Paramaterization and errors

As much as possible, HIF tries to be *mechanism* rather than *policy*,
leaving it up to the entity that interprets HIF to parametarize the
machine model (stack size, return area size) and enforce any run-time
limits on executed operations.  If errors are encountered in execution
(due to either illegal operations or running against limits), an error is
explicitly returned.

### Serialization/Deserialization

It is an essential constraint of HIF that it can execute in a `no_std`
environment; for deserialization (of program text) and serialization (of
the return stack), HIF uses [postcard](https://crates.io/crates/postcard).

### Functions

A function is denoted by a `TargetFunction` structure that contains a
`u8` index into the `functions` array passed into `execute`.  HIF makes no
assumption about what these functions are or how they are discovered; if
the program text attempts to call an invalid function, an error returned.
Functions themselves are implemented by whomever is calling `excute`; they
take the stack, a slice that is a read-only memory, and the (mutable)
return stack as arguments.  Functions are expected to return a failure if
the stack contains an incorrect number of arguments, or if the function
attempts to access memory not contained in the read-only slice, or if the
return stack is too small to contain the return value(s).  Functions do
not consume arguments from the stack; callers must explicitly drop
parameters from the stack if they are no longer needed.

### Labels

The `Op::Label` operation denotes a target for a branch, allowing
branches to be symbolic rather than in terms of text offsets.  Labels are
created on the stack; the number of labels is a const generic paramater to
`execute`.

### Stack manipulation operations

HIF offers basic stack manipulation operations like `Op::Push`,
`Op::Dup`, `Op::Swap` and `Op::Drop`.  The call stack is one of
`Option<u32>`; `Op::PushNone` pushes `None`, while all other push
operations ultimately push `Some<u32>`.

### Arithmetic operations

HIF offers a paucity of arithmetic operations.  These operations consume
the top two elements of the stack, and push the result.


License: MPL-2.0
