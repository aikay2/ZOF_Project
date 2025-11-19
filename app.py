from flask import Flask, render_template, request, redirect, url_for, flash
import math
from typing import Callable
from ZOF_CLI import (make_func, bisection, regula_falsi, secant, newton_raphson, fixed_point, modified_secant)

app = Flask(__name__)
app.secret_key = "change-this-secret-in-production"

METHOD_MAP = {
    "bisection": {"name":"Bisection", "requires": ["a","b"]},
    "regulafalsi": {"name":"Regula Falsi", "requires": ["a","b"]},
    "secant": {"name":"Secant", "requires": ["x0","x1"]},
    "newton": {"name":"Newton-Raphson", "requires": ["x0"]},
    "fixedpoint": {"name":"Fixed Point", "requires": ["g","x0"]},
    "modifiedsecant": {"name":"Modified Secant", "requires": ["x0","delta"]}
}

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    history = None
    form = {}
    if request.method == "POST":
        method = request.form.get("method")
        f_expr = request.form.get("f")
        df_expr = request.form.get("df","").strip()
        g_expr = request.form.get("g","").strip()
        # parse floats with defaults
        def tofloat(k, default=None):
            v = request.form.get(k,"").strip()
            return float(v) if v != "" else default

        a = tofloat("a")
        b = tofloat("b")
        x0 = tofloat("x0")
        x1 = tofloat("x1")
        delta = tofloat("delta", 1e-3)
        tol = tofloat("tol", 1e-6)
        maxiter = int(request.form.get("maxiter", "50"))

        form = request.form.to_dict()

        try:
            if method not in METHOD_MAP:
                flash("Invalid method selected", "danger")
                return redirect(url_for("index"))
            f = make_func(f_expr)
            df = make_func(df_expr) if df_expr else None
            g = make_func(g_expr) if g_expr else None

            if method == "bisection":
                root, history, conv = bisection(f, a, b, tol, maxiter)
            elif method == "regulafalsi":
                root, history, conv = regula_falsi(f, a, b, tol, maxiter)
            elif method == "secant":
                root, history, conv = secant(f, x0, x1, tol, maxiter)
            elif method == "newton":
                root, history, conv = newton_raphson(f, x0, tol, maxiter, df=df)
            elif method == "fixedpoint":
                root, history, conv = fixed_point(g, x0, tol, maxiter)
            elif method == "modifiedsecant":
                root, history, conv = modified_secant(f, x0, delta, tol, maxiter)
            else:
                raise ValueError("Unsupported method")

            result = {"root": root, "converged": conv, "iterations": len(history)}
        except Exception as e:
            flash(f"Computation error: {e}", "danger")
    return render_template("index.html", methods=METHOD_MAP, result=result, history=history, form=form)

if __name__ == "__main__":
    # Development server
    app.run(debug=True, host="0.0.0.0", port=5000)
