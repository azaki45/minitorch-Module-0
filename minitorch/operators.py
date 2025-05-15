"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        Product of x and y

    """
    return x * y


def id(x: Any) -> Any:
    """Returns the input as it is.

    Args:
    ----
        x: any input

    Returns:
    -------
        The unchanged input

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two input values.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        Sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Negates the input value.

    Args:
    ----
        x: a floating point value

    Returns:
    -------
        negation of x

    """
    return x * (-1.0)


def lt(x: float, y: float) -> float:
    """Checks if one number is less than the other.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        true if x<y or vice-versa

    """
    return 1.0 if x<y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        True if x==y and False otherwise

    """
    return 1.0 if x==y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        The greater value between x and y

    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        x: a floating point value
        y: a floating point value

    Returns:
    -------
        true if x and y are close in value
        false otherwise

    """
    if abs(x - y) <= 1e-2:
        return True
    else:
        return False


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x: a floating point value

    Returns:
    -------
        The sigmoid output of x

    """
    return 1.0 / (1.0 + math.exp(-1.0 * x))


def relu(x: float) -> float:
    """Calculates the ReLU activation of the input.

    Args:
    ----
        x: a floating point value

    Returns:
    -------
        ReLU activation value of x

    """
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural log.

    Args:
    ----
        x: a floating point value

    Returns:
    -------
        Natural log of x

    """
    assert x > 0, "Invalid negative input to the log function"
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential.

    Args:
    ----
        x: a floating point value

    Returns;
        The exponential value of x

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: a floating point opeand of the log function
        y: constant operand being multiplied to the log function

    Returns:
    -------
        Returns y/x according to the function description

    """
    return y / x


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x: a floating point value

    Returns:
    -------
        1/x as the reciprocal of x

    """
    assert x != 0, "Division by zero error"
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x: a floating point value argument of reciprocal
        y: constant multiplier of the reciprocal function

    Returns:
    -------
        -(y/x**2) as the derivative of (y/x)

    """
    assert x != 0, "Division by zero error"
    return (-1.0 * y) / x**2


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: input of the ReLU function
        y: constant multiplier of the ReLU function

    Returns:
    -------
        d/dx(y*ReLU(x))

    """
    if x < 0:
        return 0.0
    else:
        return y

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], Any], ls: Iterable[float]) -> Iterable[Any]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
        fn: function to be applied
        ls: list of values for the function to be applied

    Returns:
        modified list after the function application

    """
    out = []
    for mem in ls:
        out.append(fn(mem))
    return out

def zipWith(fn: Callable[[float, float], Any], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[Any]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
        fn: function to be applied
        ls1: the first iterable
        ls2: the second iterable

    Returns:
        modified list as [fn(x,y)], where x:ls1 and y:ls2

    """
    out = []
    assert len(ls1) == len(ls2), f'Given interables have different lengths {len(ls1)}, and {len(ls2)}'
    for mem1,mem2 in zip(ls1,ls2):
        out.append(fn(mem1, mem2))
    return out

def reduce(fn: Callable[[float, float], Any], ls: Iterable[float]) -> Any:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
        fn: function to be applied
        ls: iterable list

    Returns:
        consolidated value after fn application on each member of ls

    """
    out=0.0
    if len(ls) >= 1:
        out = ls[0]
    for i in range(1, len(ls)):
        out = fn(out, ls[i])
    return out



def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map.

    Args:
        ls: iterable to be negated

    Returns:
        negated iterable

    """
    out = map(neg, ls)
    return out

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
        ls1: the first Iterable
        ls2: the second Iterable

    Returns:
        list with sum of corresponding elements of ls1 and ls2

    """
    out = zipWith(add, ls1, ls2)
    return out

def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
        ls: iterable whose members are to be addded

    Returns:
        sum of the entire list

    """
    return reduce(add, ls)

def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
        ls: iterable whose members are to be multiplied

    Returns:
        product of the entire list

    """
    return reduce(mul, ls)