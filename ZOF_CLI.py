#!/usr/bin/env python3
"""
ZOF_CLI.py
Command-line interface for Zero-Of-Function solver (six methods).
Usage examples:
    python ZOF_CLI.py --method bisection --f "x**3-2*x-5" --a 1 --b 3 --tol 1e-6 --maxiter 50
    python ZOF_CLI.py --method newton --f "x**3-2*x-5" --x0 2 --tol 1e-8
"""

import argparse
import math
import sys
from typing import Callable, Dict, List, Tuple

# -----------------------
# Helper: safe evaluator
# -----------------------
def make_func(expr: str) -> Callable[[float], float]:
    """
    Build a function f(x) from a user expression string.
    Allowed names: math module functions and 'x'.
    """
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    # Add 'x' as placeholder; compile to code object
    code = compile(expr, "<string>", "eval")
    def f(x):
        local = {"x": x}
        return eval(code, {"__builtins__": {} , **allowed_names}, local)
    return f

# -----------------------
# Root-finding methods
# Each returns (root, history_list, converged)
# history_list: list of dicts with iteration info
# -----------------------
def bisection(f: Callable[[float], float], a: float, b: float, tol: float, maxiter: int):
    fa, fb = f(a), f(b)
    history = []
    if fa == 0:
        return a, [{"iter":0,"a":a,"b":b,"c":a,"f(c)":fa,"error":0}], True
    if fb == 0:
        return b, [{"iter":0,"a":a,"b":b,"c":b,"f(c)":fb,"error":0}], True
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection.")
    c = a
    for itr in range(1, maxiter+1):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a) / 2.0
        history.append({"iter": itr, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if abs(fc) == 0 or error < tol:
            return c, history, True
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, history, False

def regula_falsi(f: Callable[[float], float], a: float, b: float, tol: float, maxiter: int):
    fa, fb = f(a), f(b)
    history = []
    if fa == 0:
        return a, [{"iter":0,"a":a,"b":b,"c":a,"f(c)":fa,"error":0}], True
    if fb == 0:
        return b, [{"iter":0,"a":a,"b":b,"c":b,"f(c)":fb,"error":0}], True
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    c = a
    prev_c = None
    for itr in range(1, maxiter+1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(c - (prev_c if prev_c is not None else c))
        history.append({"iter": itr, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})
        if abs(fc) == 0 or error < tol:
            return c, history, True
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        prev_c = c
    return c, history, False

def secant(f: Callable[[float], float], x0: float, x1: float, tol: float, maxiter: int):
    history = []
    f0, f1 = f(x0), f(x1)
    for itr in range(1, maxiter+1):
        if f1 - f0 == 0:
            raise ZeroDivisionError("Zero denominator in secant formula.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        error = abs(x2 - x1)
        history.append({"iter": itr, "x0": x0, "x1": x1, "x2": x2, "f(x2)": f2, "error": error})
        if abs(f2) == 0 or error < tol:
            return x2, history, True
        x0, f0 = x1, f1
        x1, f1 = x2, f2
    return x2, history, False

def newton_raphson(f: Callable[[float], float], x0: float, tol: float, maxiter: int, df=None):
    history = []
    def numeric_df(x, h=1e-6):
        return (f(x+h) - f(x-h)) / (2*h)
    x = x0
    for itr in range(1, maxiter+1):
        derivative = df(x) if df is not None else numeric_df(x)
        fx = f(x)
        if derivative == 0:
            raise ZeroDivisionError("Derivative is zero; Newton-Raphson fails.")
        x_new = x - fx / derivative
        error = abs(x_new - x)
        history.append({"iter": itr, "x": x, "f(x)": fx, "df(x)": derivative, "x_new": x_new, "error": error})
        if abs(fx) == 0 or error < tol:
            return x_new, history, True
        x = x_new
    return x, history, False

def fixed_point(g: Callable[[float], float], x0: float, tol: float, maxiter: int):
    history = []
    x = x0
    for itr in range(1, maxiter+1):
        x_new = g(x)
        error = abs(x_new - x)
        history.append({"iter": itr, "x": x, "g(x)": x_new, "error": error})
        if error < tol:
            return x_new, history, True
        x = x_new
    return x, history, False

def modified_secant(f: Callable[[float], float], x0: float, delta: float, tol: float, maxiter: int):
    history = []
    x = x0
    for itr in range(1, maxiter+1):
        f_x = f(x)
        denom = f(x + delta * x) - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in modified secant.")
        x_new = x - (delta * x * f_x) / denom
        error = abs(x_new - x)
        history.append({"iter": itr, "x": x, "f(x)": f_x, "x_new": x_new, "error": error})
        if abs(f_x) == 0 or error < tol:
            return x_new, history, True
        x = x_new
    return x, history, False

# -----------------------
# CLI wiring
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ZOF CLI - root finding")
    parser.add_argument("--method", required=True, choices=["bisection","regulafalsi","secant","newton","fixedpoint","modifiedsecant"], help="Method")
    parser.add_argument("--f", required=True, help="Function in 'x' (python expression). Example: 'x**3-2*x-5'")
    parser.add_argument("--df", help="Derivative (for newton) optional. Example: '3*x**2-2'")
    parser.add_argument("--g", help="g(x) for fixed-point iteration")
    parser.add_argument("--a", type=float, help="Left endpoint (bisection/regulafalsi)")
    parser.add_argument("--b", type=float, help="Right endpoint (bisection/regulafalsi)")
    parser.add_argument("--x0", type=float, help="Initial guess x0")
    parser.add_argument("--x1", type=float, help="Initial guess x1 (secant)")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance")
    parser.add_argument("--maxiter", type=int, default=50, help="Maximum iterations")
    parser.add_argument("--delta", type=float, default=1e-3, help="Delta for modified secant")
    return parser.parse_args()

def pretty_print(history):
    for row in history:
        print(row)

def main():
    args = parse_args()
    f = make_func(args.f)
    df = make_func(args.df) if args.df else None
    g = make_func(args.g) if args.g else None

    try:
        if args.method == "bisection":
            if args.a is None or args.b is None:
                raise ValueError("Bisection needs --a and --b")
            root, history, conv = bisection(f, args.a, args.b, args.tol, args.maxiter)
        elif args.method == "regulafalsi":
            if args.a is None or args.b is None:
                raise ValueError("Regula Falsi needs --a and --b")
            root, history, conv = regula_falsi(f, args.a, args.b, args.tol, args.maxiter)
        elif args.method == "secant":
            if args.x0 is None or args.x1 is None:
                raise ValueError("Secant needs --x0 and --x1")
            root, history, conv = secant(f, args.x0, args.x1, args.tol, args.maxiter)
        elif args.method == "newton":
            if args.x0 is None:
                raise ValueError("Newton needs --x0")
            root, history, conv = newton_raphson(f, args.x0, args.tol, args.maxiter, df=df)
        elif args.method == "fixedpoint":
            if g is None or args.x0 is None:
                raise ValueError("Fixed-Point needs --g and --x0")
            root, history, conv = fixed_point(g, args.x0, args.tol, args.maxiter)
        elif args.method == "modifiedsecant":
            if args.x0 is None:
                raise ValueError("Modified Secant needs --x0 and optionally --delta")
            root, history, conv = modified_secant(f, args.x0, args.delta, args.tol, args.maxiter)
        else:
            raise ValueError("Unknown method")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Iteration history (latest entries last) ---")
    pretty_print(history)
    final_iter = history[-1] if history else {}
    final_error = final_iter.get("error", None)
    print(f"\nFinal estimated root: {root}")
    print(f"Converged: {conv}")
    print(f"Final error estimate: {final_error}")
    print(f"Total iterations: {len(history)}")

if __name__ == "__main__":
    main()
