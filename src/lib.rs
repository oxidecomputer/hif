#![no_std]

///
/// HIF: The Hubris/Humility Interchange Format
///
/// When debugging Hubris, we have found it handy to get a task to execute as
/// a proxy for an external task to (e.g.) perform I2C transactions.  This was
/// born as an ad hoc facility, effected done by the debugger stopping the
/// target and writing parameters to well-known memory locations, but as our
/// need for proxy behavior became more sophisticated, it became clear that we
/// needed something more general purpose.  HIF represents a simple,
/// stack-based machine that can be executed in a memory-constrained no-std
/// environment.  While this was designed to accommodate Hubris and Humility,
/// there is nothing specific to either, and may well have other uses.
///
use postcard::{take_from_bytes, to_slice};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct TargetFunction(u8);

#[derive(Serialize, Deserialize, Debug)]
pub struct Target(u8);

#[derive(Serialize, Deserialize, Debug)]
pub enum Op {
    Label(Target),
    Call(TargetFunction),
    Drop,
    Push(u8),
    Push16(u16),
    Push32(u32),
    PushNone,
    Dup,
    Add,
    BranchLessThan(Target),
    BranchGreaterThan(Target),
    BranchEqualTo(Target),
    BranchAlways(Target),
    Done,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum IllegalOp {
    NoTarget,
    BadTarget,
    BadFunction,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum Fault {
    StackOverflow,
    StackUnderflow,
    OperationOnNone,
    DropUnderflow,
    DupUnderflow,
    MissingParameter(u8),
    BadParameter(u8),
    ReturnValueOverflow,
    ReturnStackOverflow,
    DoneStackOverflow,
}

#[derive(Serialize, Deserialize, Debug)]
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

pub type Function = fn(&[Option<u32>], &mut [u8]) -> Result<usize, Failure>;

pub fn execute<'a>(
    text: &[u8],
    functions: &[Function],
    stack: &mut [Option<u32>],
    rstack: &mut [u8],
    scratch: &mut [u8],
) -> Result<(), Failure> {
    let mut pc = text;
    let mut sp = 0;
    let mut rp = 0;

    // Three labels ought to be enough for anyone!
    let mut labels = [None, None, None];

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

        if ndx > functions.len() {
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

        push(stack, sp, stack[sp])
    }

    loop {
        match take_from_bytes::<Op>(pc) {
            Ok((op, npc)) => {
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

                    Op::Drop => {
                        sp = drop(stack, sp)?;
                    }

                    Op::BranchLessThan(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs < rhs {
                            pc = target(&labels, val)?;
                        }
                    }

                    Op::BranchGreaterThan(Target(val)) => {
                        let (lhs, rhs) = operands(&stack, sp)?;

                        if lhs > rhs {
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
                        let rval = target(&stack[0..sp], scratch);

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
    fn func(stack: &[Option<u32>], rval: &mut [u8]) -> Result<usize, Failure> {
        if stack.len() == 0 {
            Err(Failure::Fault(Fault::MissingParameter(0)))
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

    fn run(op: &[Op]) -> Result<Vec<Result<Vec<u8>, u32>>, Failure> {
        let mut stack = [None; 8];
        let mut rstack = [0u8; 256];
        let mut scratch = [0u8; 256];
        let mut text = [0u8; 256];
        let functions: &[Function] = &[func];

        let serialized = to_slice(op, &mut text).unwrap();

        execute(
            &serialized[1..],
            functions,
            &mut stack,
            &mut rstack,
            &mut scratch,
        )?;

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
        let rval = run(op);

        if let Err(Failure::Fault(fault)) = rval {
            assert_eq!(fault, expected);
        } else {
            panic!("unexpected rval: {:?}", rval);
        }
    }

    fn illop(op: &[Op], expected: IllegalOp) {
        let rval = run(op);

        if let Err(Failure::IllegalOp(illop)) = rval {
            assert_eq!(illop, expected);
        } else {
            panic!("unexpected rval: {:?}", rval);
        }
    }

    #[test]
    fn underflow() {
        fault(&[Op::Add], Fault::StackUnderflow);
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

        fault(&op, Fault::MissingParameter(0));
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
    fn loopy() {
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

        let rval = run(&op);
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
}
