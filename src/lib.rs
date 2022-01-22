//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#![no_std]
//
// Note that the below rustdoc is used by `cargo readme` to generate the
// README, but due to https://github.com/livioribeiro/cargo-readme/issues/70,
// it must be cleaned up manually, e.g.:
//
// ```
// cargo readme | sed 's/\[\(`[^`]*`\)]/\1/g' > README.md
// ```
//

//! # HIF: The Hubris/Humility Interchange Format
//!
//! When debugging Hubris, we have found it handy to get a task to execute as
//! a proxy for an external task.  This was born as an ad hoc facility to
//! induce I2C transactions, effected by the debugger stopping the target and
//! writing parameters to well-known memory locations; as our need for proxy
//! behavior became more sophisticated, it became clear that we needed
//! something more general purpose.  HIF represents a simple, stack-based
//! machine that can be executed in a memory-constrained no-std environment.
//! While this was designed to accommodate Hubris and Humility, there is
//! nothing specific to either, and may well have other uses.
//!
//! ## Machine model
//!
//! The HIF machine model consists of program text consisting of an array of
//! [`Op`] structures denoting operations; a stack of 32-bit values for
//! arguments and variables; a finite number of branch targets corresponding
//! to declared labels in program text; and an append-only return stack that
//! consists of an array of [`FunctionResult`] enums.  Of note, this model has
//! no registers (though such extensions are clearly possible).
//!
//! ## Paramaterization and errors
//!
//! As much as possible, HIF tries to be *mechanism* rather than *policy*,
//! leaving it up to the entity that interprets HIF to parametarize the
//! machine model (stack size, return area size) and enforce any run-time
//! limits on executed operations.  If errors are encountered in execution
//! (due to either illegal operations or running against limits), an error is
//! explicitly returned.
//!
//! ## Serialization/Deserialization
//!
//! It is an essential constraint of HIF that it can execute in a `no_std`
//! environment; for deserialization (of program text) and serialization (of
//! the return stack), HIF uses [postcard](https://crates.io/crates/postcard).
//!
//! ## Functions
//!
//! A function is denoted by a [`TargetFunction`] structure that contains a
//! `u8` index into the `functions` array passed into `execute`.  HIF makes no
//! assumption about what these functions are or how they are discovered; if
//! the program text attempts to call an invalid function, an error returned.
//! Functions themselves are implemented by whomever is calling `execute`; they
//! take the stack, a slice that is a read-only memory, and the (mutable)
//! return stack as arguments.  Functions are expected to return a failure if
//! the stack contains an incorrect number of arguments, or if the function
//! attempts to access memory not contained in the read-only slice, or if the
//! return stack is too small to contain the return value(s).  Functions do
//! not consume arguments from the stack; callers must explicitly drop
//! parameters from the stack if they are no longer needed.
//!
//! ## Labels
//!
//! The [`Op::Label`] operation denotes a target for a branch, allowing
//! branches to be symbolic rather than in terms of text offsets.  Labels are
//! created on the stack; the number of labels is a const generic paramater to
//! `execute`.
//!
//! ## Stack manipulation operations
//!
//! HIF offers basic stack manipulation operations like [`Op::Push`],
//! [`Op::Dup`], [`Op::Swap`] and [`Op::Drop`].  The call stack is one of
//! `Option<u32>`; [`Op::PushNone`] pushes `None`, while all other push
//! operations ultimately push `Some<u32>`.
//!
//! ## Arithmetic operations
//!
//! HIF offers a paucity of arithmetic operations.  These operations consume
//! the top two elements of the stack, and push the result.
//!
use pkg_version::*;
use postcard::{take_from_bytes, to_slice};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct TargetFunction(pub u8);

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct Target(pub u8);

///
/// Enum that defines a HIF operation
///
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub enum Op {
    /// Define a label as `Target`
    Label(Target),

    /// Call the specified function
    Call(TargetFunction),

    /// Drop the top element on the stack
    Drop,

    /// Drop the top N elements on the stack
    DropN(u8),

    /// Push `Some<u32>` onto the stack that contains the specified `u8`
    Push(u8),

    /// Push `Some<u32>` onto the stack that contains the specified `u16`
    Push16(u16),

    /// Push `Some<u32>` onto the stack that contains the specified value
    Push32(u32),

    /// Push `None` onto the stack
    PushNone,

    /// Swap the top two elements on the stack
    Swap,

    /// Duplicate the top element of the stack, pushing it
    Dup,

    /// Add the top two elements of the stack, replacing them with the result
    Add,

    /// Bitwise AND top two elements of the stack, replacing them with result
    And,

    /// Bitwise OR top two elements of the stack, replacing them with result
    Or,

    /// Bitwise XOR top two elements of the stack, replacing them with result
    Xor,

    /// Compare the top two elements of the stack, branching if the topmost
    /// is less than the second topmost
    BranchLessThan(Target),

    /// Compare the top two elements of the stack, branching if the topmost
    /// is less than or equal to the second topmost
    BranchLessThanOrEqualTo(Target),

    /// Compare the top two elements of the stack, branching if the topmost
    /// is greater than the second topmost
    BranchGreaterThan(Target),

    /// Compare the top two elements of the stack, branching if the topmost
    /// is greater than or equal to the second topmost
    BranchGreaterThanOrEqualTo(Target),

    /// Compare the top two elements of the stack, branching if they are equal
    BranchEqualTo(Target),

    /// Always branch
    BranchAlways(Target),

    /// Denote the end of execution. All execution must end with `Done`
    Done,
}

///
/// Enum to denote an *illegal operation*, a failure that constitutes
/// illegal program text that can't ever operate correctly
///
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub enum IllegalOp {
    /// Specified target does not exist
    NoTarget,

    /// Attempt to specify a target that exceeds the bounds
    BadTarget,

    /// Attempt to call a function that exceeds the bounds
    BadFunction,
}

///
/// Enum to denote a *fault*, a run-time failure that isn't an illegal
/// operation
///
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub enum Fault {
    /// Attempt to push to a full stack
    StackOverflow,

    /// Attempt to reference beneath the bottom of the stack
    StackUnderflow,

    /// Attempt to operate on a stack value that was `None`
    OperationOnNone,

    /// Attempt to drop on an empty stack
    DropUnderflow,

    /// Attempt to dup on an empty stack
    DupUnderflow,

    /// Attempt to call a function with an insufficient number of
    /// parameters pushed onto the stack
    MissingParameters,

    /// Attempt to call a function with a bad parameter, the index of
    /// which is denoted by the datum
    BadParameter(u8),

    /// Attempt to call a function with an empty parameter, the index of
    /// which is denoted by the datum
    EmptyParameter(u8),

    /// Attempt to overflow return stack from within a function
    ReturnValueOverflow,

    /// Attempt to overflow return stack with a function's return value
    ReturnStackOverflow,

    /// Attempt to overflow return stack with [`Op::Done`]
    DoneStackOverflow,

    /// Attempt to access beyond the bounds of the memory
    AccessOutOfBounds,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub enum Failure {
    IllegalOp(IllegalOp),
    Fault(Fault),
    Invalid,
    FunctionError(u32),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FunctionResult<'a> {
    Success(&'a [u8]),
    Failure(u32),
    Done,
}

pub type Function =
    fn(&[Option<u32>], &[u8], &mut [u8]) -> Result<usize, Failure>;

pub const HIF_VERSION_MAJOR: u32 = pkg_version_major!();
pub const HIF_VERSION_MINOR: u32 = pkg_version_minor!();
pub const HIF_VERSION_PATCH: u32 = pkg_version_patch!();

///
/// The [`execute`] function actually executes the HIF machine.  It takes
/// program text (a slice of `u8` that will be deserialized into an array of
/// `Op` enums), functions (a slice of `Function`), a slice of read-only data
/// representing memory, a stack (which should be initialized to an array of
/// `None`), a return stack, a scratch area, and a closure to be called on
/// every operation.  (Note that the number of labels is parameterized as a
/// const generic.)
///
//
// Note that the `where` clause is used here on `check` instead of the more
// conventional `impl FnMut` position because of a Rust limitation; see
// [#85475](https://github.com/rust-lang/rust/issues/85475) for more details.
//
pub fn execute<'a, F, const NLABELS: usize>(
    text: &[u8],
    functions: &[Function],
    data: &[u8],
    stack: &mut [Option<u32>],
    rstack: &mut [u8],
    scratch: &mut [u8],
    mut check: F,
) -> Result<(), Failure>
where
    F: FnMut(usize, &Op) -> Result<(), Failure>,
{
    let mut pc = text;
    let mut sp = 0;
    let mut rp = 0;

    let mut labels = [None; NLABELS];

    fn labelndx(labels: &[Option<&[u8]>], val: u8) -> Result<usize, Failure> {
        let ndx = val as usize;

        if ndx >= labels.len() {
            Err(Failure::IllegalOp(IllegalOp::BadTarget))
        } else {
            Ok(ndx)
        }
    }

    fn target<'a>(
        labels: &[Option<&'a [u8]>],
        val: u8,
    ) -> Result<&'a [u8], Failure> {
        match labels[labelndx(&labels, val)?] {
            Some(target) => Ok(target),
            None => Err(Failure::IllegalOp(IllegalOp::NoTarget)),
        }
    }

    fn function(functions: &[Function], val: u8) -> Result<Function, Failure> {
        let ndx = val as usize;

        if ndx >= functions.len() {
            Err(Failure::IllegalOp(IllegalOp::BadFunction))
        } else {
            Ok(functions[ndx])
        }
    }

    fn operands(
        stack: &[Option<u32>],
        sp: usize,
    ) -> Result<(u32, u32), Failure> {
        if sp < 2 {
            return Err(Failure::Fault(Fault::StackUnderflow));
        }

        match (stack[sp - 1], stack[sp - 2]) {
            (Some(lhs), Some(rhs)) => Ok((lhs, rhs)),
            _ => Err(Failure::Fault(Fault::OperationOnNone)),
        }
    }

    fn drop(stack: &mut [Option<u32>], sp: usize) -> Result<usize, Failure> {
        if sp == 0 {
            return Err(Failure::Fault(Fault::DropUnderflow));
        }
        stack[sp - 1] = None;
        Ok(sp - 1)
    }

    fn push(
        stack: &mut [Option<u32>],
        sp: usize,
        val: Option<u32>,
    ) -> Result<usize, Failure> {
        if sp == stack.len() {
            return Err(Failure::Fault(Fault::StackOverflow));
        }

        stack[sp] = val;
        Ok(sp + 1)
    }

    fn dup(stack: &mut [Option<u32>], sp: usize) -> Result<usize, Failure> {
        if sp == 0 {
            return Err(Failure::Fault(Fault::DupUnderflow));
        }

        push(stack, sp, stack[sp - 1])
    }

    loop {
        match take_from_bytes::<Op>(pc) {
            Ok((op, npc)) => {
                check(pc.as_ptr() as usize - text.as_ptr() as usize, &op)?;

                pc = npc;

                match op {
                    Op::Label(Target(val)) => {
                        labels[labelndx(&labels, val)?] = Some(npc);
                    }

                    Op::Push(val) => {
                        sp = push(stack, sp, Some(val as u32))?;
                    }

                    Op::Push16(val) => {
                        sp = push(stack, sp, Some(val as u32))?;
                    }

                    Op::Push32(val) => {
                        sp = push(stack, sp, Some(val))?;
                    }

                    Op::PushNone => {
                        sp = push(stack, sp, None)?;
                    }

                    Op::Dup => {
                        sp = dup(stack, sp)?;
                    }

                    Op::Add => {
                        let (lhs, rhs) = operands(&stack, sp)?;
                        sp = drop(stack, sp)?;
                        stack[sp - 1] = Some(lhs + rhs);
                    }

                    Op::And => {
                        let (lhs, rhs) = operands(&stack, sp)?;
                        sp = drop(stack, sp)?;
                        stack[sp - 1] = Some(lhs & rhs);
                    }

                    Op::Or => {
                        let (lhs, rhs) = operands(&stack, sp)?;
                        sp = drop(stack, sp)?;
                        stack[sp - 1] = Some(lhs | rhs);
                    }

                    Op::Xor => {
                        let (lhs, rhs) = operands(&stack, sp)?;
                        sp = drop(stack, sp)?;
                        stack[sp - 1] = Some(lhs ^ rhs);
                    }

                    Op::Drop => {
                        sp = drop(stack, sp)?;
                    }

                    Op::DropN(val) => {
                        for _i in 0..val {
                            sp = drop(stack, sp)?;
                        }
                    }

                    Op::Swap => {
                        if sp < 2 {
                            return Err(Failure::Fault(Fault::StackUnderflow));
                        }

                        let tmp = stack[sp - 1];
                        stack[sp - 1] = stack[sp - 2];
                        stack[sp - 2] = tmp;
                    }

                    Op::BranchLessThan(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs < rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchLessThanOrEqualTo(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs <= rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchGreaterThan(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs > rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchGreaterThanOrEqualTo(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs >= rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchEqualTo(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs == rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchAlways(Target(val)) => {
                        pc = target(&labels, val)?;
                    }

                    Op::Done => {
                        let done = FunctionResult::Done;

                        if let Err(_) = to_slice(&done, &mut rstack[rp..]) {
                            return Err(Failure::Fault(
                                Fault::DoneStackOverflow,
                            ));
                        }

                        return Ok(());
                    }

                    Op::Call(TargetFunction(val)) => {
                        let target = function(functions, val)?;
                        let rval = target(&stack[0..sp], data, scratch);

                        let rval = match rval {
                            Ok(r) => FunctionResult::Success(&scratch[0..r]),
                            Err(failure) => {
                                if let Failure::FunctionError(code) = failure {
                                    FunctionResult::Failure(code)
                                } else {
                                    return Err(failure);
                                }
                            }
                        };

                        let ser = to_slice(&rval, &mut rstack[rp..]);

                        match ser {
                            Ok(ref ser) => {
                                rp += ser.len();
                            }
                            Err(_) => {
                                return Err(Failure::Fault(
                                    Fault::ReturnStackOverflow,
                                ));
                            }
                        }
                    }
                }
            }

            Err(_) => {
                return Err(Failure::Invalid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::*;

    //
    // A test function that returns twice its parameter if the parameter
    // is even, otherwise it returns a failure with the parameter as the
    // error code.
    //
    fn loopy(
        stack: &[Option<u32>],
        _data: &[u8],
        rval: &mut [u8],
    ) -> Result<usize, Failure> {
        if stack.len() == 0 {
            Err(Failure::Fault(Fault::MissingParameters))
        } else if rval.len() < 1 {
            Err(Failure::Fault(Fault::ReturnValueOverflow))
        } else {
            match stack[stack.len() - 1] {
                Some(val) => {
                    if val % 2 == 0 {
                        rval[0] = (val * 2) as u8;
                        Ok(1)
                    } else {
                        Err(Failure::FunctionError(val))
                    }
                }
                None => Err(Failure::Fault(Fault::BadParameter(0))),
            }
        }
    }

    fn okno(
        _stack: &[Option<u32>],
        _data: &[u8],
        _rval: &mut [u8],
    ) -> Result<usize, Failure> {
        Ok(0)
    }

    fn loadat(
        stack: &[Option<u32>],
        data: &[u8],
        rval: &mut [u8],
    ) -> Result<usize, Failure> {
        if stack.len() == 0 {
            Err(Failure::Fault(Fault::MissingParameters))
        } else if rval.len() < 1 {
            Err(Failure::Fault(Fault::ReturnValueOverflow))
        } else {
            match stack[0] {
                Some(val) => {
                    let ndx = val as usize;

                    if ndx < data.len() {
                        rval[0] = data[ndx];
                        Ok(1)
                    } else {
                        Err(Failure::Fault(Fault::AccessOutOfBounds))
                    }
                }
                None => Err(Failure::Fault(Fault::BadParameter(0))),
            }
        }
    }

    fn run(
        ops: &[Op],
        assert_stack: Option<&[Option<u32>]>,
    ) -> Result<Vec<Result<Vec<u8>, u32>>, Failure> {
        const NLABELS: usize = 8;
        let mut stack = [None; 8];
        let mut rstack = [0u8; 256];
        let mut scratch = [0u8; 256];
        let mut text: Vec<u8> = vec![];
        text.resize_with(2048, Default::default);

        assert!(
            assert_stack.is_none() || assert_stack.unwrap().len() < stack.len()
        );

        let functions: &[Function] = &[loopy, okno, loadat];

        let buf = &mut text.as_mut_slice();
        let mut current = 0;

        for op in ops {
            let serialized = to_slice(op, &mut buf[current..]).unwrap();
            current += serialized.len();
        }

        let mut ninstr = 0;
        let data = [0x1du8; 1];

        execute::<_, NLABELS>(
            &buf[0..],
            functions,
            &data[0..1],
            &mut stack,
            &mut rstack,
            &mut scratch,
            |offset, op| {
                std::println!("offset 0x{:04x}: {:?}", offset, op);
                ninstr += 1;
                Ok(())
            },
        )?;

        std::println!("{} total instructions", ninstr);

        if let Some(assert_stack) = assert_stack {
            for i in 0..assert_stack.len() {
                if assert_stack[i] != stack[i] {
                    panic!(
                        "stack mismatch; expected: {:#x?} found: {:#x?}",
                        assert_stack,
                        &stack[0..assert_stack.len()]
                    );
                }
            }
        }

        let mut rvec = vec![];
        let mut result = &rstack[0..];

        loop {
            match take_from_bytes::<FunctionResult>(result) {
                Ok((rval, next)) => {
                    match rval {
                        FunctionResult::Done => {
                            return Ok(rvec);
                        }

                        FunctionResult::Success(ref payload) => {
                            rvec.push(Ok(payload.to_vec()))
                        }

                        FunctionResult::Failure(code) => rvec.push(Err(code)),
                    }

                    result = next;
                }

                Err(err) => {
                    std::println!("failed to parse: {:?}", err);
                }
            }
        }
    }

    fn fault(op: &[Op], expected: Fault) {
        let rval = run(op, None);

        if let Err(Failure::Fault(fault)) = rval {
            assert_eq!(fault, expected);
        } else {
            panic!("unexpected rval: {:?}", rval);
        }
    }

    fn illop(op: &[Op], expected: IllegalOp) {
        let rval = run(op, None);

        if let Err(Failure::IllegalOp(illop)) = rval {
            assert_eq!(illop, expected);
        } else {
            panic!("unexpected rval: {:?}", rval);
        }
    }

    #[test]
    fn underflow() {
        fault(&[Op::Push(0), Op::Add], Fault::StackUnderflow);
    }

    #[test]
    fn dup_underflow() {
        fault(&[Op::Dup], Fault::DupUnderflow);
    }

    #[test]
    fn drop_underflow() {
        fault(&[Op::Drop], Fault::DropUnderflow);
    }

    #[test]
    fn swap_underflow() {
        fault(&[Op::Push(0), Op::Swap], Fault::StackUnderflow);
    }

    #[test]
    fn dup() {
        let op = [Op::Push(0x02), Op::Dup, Op::Add, Op::Done];

        run(&op, Some(&[Some(0x04)])).unwrap();
    }

    #[test]
    fn dropn() {
        let op = [
            Op::Push(0x09),
            Op::Push(0x19),
            Op::Push(0x1d),
            Op::Push(0xe),
            Op::DropN(2),
            Op::Done,
        ];

        run(&op, Some(&[Some(0x09), Some(0x19)])).unwrap();
    }

    #[test]
    fn dropn_underflow() {
        let op = [
            Op::Push(0x09),
            Op::Push(0x19),
            Op::Push(0x1d),
            Op::Push(0xe),
            Op::DropN(5),
        ];

        fault(&op, Fault::DropUnderflow);
    }

    #[test]
    fn and() {
        let op = [
            Op::Push32(0x55aa55aa),
            Op::Push32(0x11111111),
            Op::And,
            Op::Done,
        ];

        run(&op, Some(&[Some(0x11001100)])).unwrap();
    }

    #[test]
    fn or() {
        let op = [Op::Push32(0x55aa55aa), Op::Push16(0xaa55), Op::Or, Op::Done];

        run(&op, Some(&[Some(0x55aaffff)])).unwrap();
    }

    #[test]
    fn xor() {
        let op = [
            Op::Push16(0xaaaa),
            Op::Push32(0x55aa55aa),
            Op::Xor,
            Op::Done,
        ];

        run(&op, Some(&[Some(0x55aaff00)])).unwrap();
    }

    #[test]
    fn overflow() {
        let op = [
            Op::Label(Target(0)),
            Op::Push(0x1d),
            Op::BranchAlways(Target(0)),
        ];

        fault(&op, Fault::StackOverflow);
    }

    #[test]
    fn none() {
        let op = [Op::PushNone, Op::Push(1), Op::Add];

        fault(&op, Fault::OperationOnNone);
    }

    #[test]
    fn bad_parameter() {
        let op = [Op::PushNone, Op::Call(TargetFunction(0))];

        fault(&op, Fault::BadParameter(0));
    }

    #[test]
    fn missing_parameter() {
        let op = [Op::Call(TargetFunction(0))];

        fault(&op, Fault::MissingParameters);
    }

    #[test]
    fn return_stack_overflow() {
        let op = [
            Op::Push(1),
            Op::Label(Target(0)),
            Op::Call(TargetFunction(0)),
            Op::BranchAlways(Target(0)),
        ];

        fault(&op, Fault::ReturnStackOverflow);
    }

    #[test]
    fn bad_target() {
        let op = [Op::Label(Target(0)), Op::BranchAlways(Target(255))];

        illop(&op, IllegalOp::BadTarget);
    }

    #[test]
    fn bad_label() {
        let mut ops = vec![];

        for i in 0..4 {
            ops.push(Op::Label(Target(i)));
        }

        ops.push(Op::Done);

        //
        // We have at least 4 labels, so this should be fine...
        //
        let rval = run(&ops, None);
        assert_eq!(rval, Ok(vec![]));

        ops.pop();

        //
        // Assuming that we don't have 100 labels.
        //
        for i in 4..100 {
            ops.push(Op::Label(Target(i)));
        }

        ops.push(Op::Done);
        illop(&ops, IllegalOp::BadTarget);
    }

    #[test]
    fn no_label() {
        let op = [Op::BranchAlways(Target(0))];

        illop(&op, IllegalOp::NoTarget);
    }

    #[test]
    fn bad_function() {
        let op = [Op::Call(TargetFunction(20))];

        illop(&op, IllegalOp::BadFunction);
    }

    #[test]
    fn call_loopy() {
        let iter = 10;

        let op = [
            Op::Push(iter),
            Op::Push(0),
            Op::Label(Target(0)),
            Op::Call(TargetFunction(0)),
            Op::Push(1),
            Op::Add,
            Op::BranchLessThan(Target(0)),
            Op::Done,
        ];

        let rval = run(&op, None);
        assert!(rval.is_ok());

        let rval = rval.unwrap();
        assert!(rval.len() == iter as usize);

        for i in 0..rval.len() {
            if i % 2 == 0 {
                match &rval[i] {
                    Ok(val) => {
                        assert!(val.len() == 1);
                        assert!(val[0] == (i * 2) as u8);
                    }

                    Err(_) => {
                        panic!("unexpected failure");
                    }
                }
            } else {
                match &rval[i] {
                    Err(err) => {
                        assert!(*err == i as u32);
                    }

                    Ok(_) => {
                        panic!("unexpected success");
                    }
                }
            }
        }
    }

    #[test]
    fn call_okno() {
        let op = [Op::Call(TargetFunction(1)), Op::Done];

        assert_eq!(run(&op, None), Ok(vec![Ok(vec![])]));
    }

    #[test]
    fn call_loadat_good() {
        let op = [Op::Push(0), Op::Call(TargetFunction(2)), Op::Done];
        assert_eq!(run(&op, None), Ok(vec![Ok(vec![0x1d])]));
    }

    #[test]
    fn call_loadat_bad() {
        let op = [Op::Push(1), Op::Call(TargetFunction(2)), Op::Done];
        fault(&op, Fault::AccessOutOfBounds);
    }

    #[test]
    fn function_obiwan() {
        let op = [Op::Call(TargetFunction(3)), Op::Done];

        illop(&op, IllegalOp::BadFunction);
    }
}
