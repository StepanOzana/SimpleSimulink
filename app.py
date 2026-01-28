from __future__ import annotations

import mimetypes
mimetypes.add_type("text/javascript", ".mjs")

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional, Set
from flask import Flask, request, jsonify
from flask import send_from_directory
from werkzeug.exceptions import BadRequest
import traceback
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import tf2ss
import types

# ----------------------------
# Python S-Function sandbox helpers
# ----------------------------

_ALLOWED_IMPORTS: Set[str] = {"math", "numpy", "scipy", "random", "time", "statistics","control"}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import '{top}' is not allowed in PythonSFunction.")
    return __import__(name, globals, locals, fromlist, level)

_SAFE_BUILTINS = {
    # basics
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "enumerate": enumerate,
    "int": int, "float": float, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "zip": zip,
    # errors
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    # import (restricted)
    "__import__": _safe_import,
}

app = Flask(__name__)

# ----------------------------
# Block definitions (MVP)
# ----------------------------

@dataclass(frozen=True)
class PortRef:
    block: str
    port: str

Wire = Dict[str, Dict[str, str]]  # {"from": {"block":..,"port":..}, "to": {...}}

BLOCK_SPECS = {
    "Constant": {
        "inputs": [],
        "outputs": ["y"],
        "params": {"value": 0.0},
    },
    "Sine": {
        "inputs": [],
        "outputs": ["y"],
        "params": {"amp": 1.0, "freq_hz": 1.0, "phase": 0.0},
    },
    "Sum": {
        "inputs": ["u1", "u2"],
        "outputs": ["y"],
        # Simulink-like signs string, e.g. "++" or "+-"
        "params": {"signs": "++"},
    },
    "Gain": {
        "inputs": ["u"],
        "outputs": ["y"],
        "params": {"k": 1.0},
    },
    "Product": {
        "inputs": ["u1", "u2"],
        "outputs": ["y"],
        "params": {},
    },
    "Integrator": {
        "inputs": ["u"],
        "outputs": ["y"],
        "params": {"x0": 0.0},
    },

    # Discontinuities
    "Saturation": {
        "inputs": ["u"],
        "outputs": ["y"],
        "params": {"lower": -1.0, "upper": 1.0},
    },

    # Continuous-time Transfer Function block (strictly proper in MVP).
    # Parameters are CSV strings like "1" and "1,1" (highest power first).
    "TransferFunction": {
        "inputs": ["u"],
        "outputs": ["y"],
        "params": {"num": "1", "den": "1,1"},
    },

    # User-programmable continuous-time block (Simulink S-Function-like, simplified)
    # Provide Python code that defines:
    #   def outputs(t, x, u, p):
    #       return y
    #   def derivatives(t, x, u, p):
    #       return dx
    # where:
    #   t: float
    #   x: numpy array (n_states,)
    #   u: float (input)
    #   p: dict (params)
    # Return values may be scalar or 1D array/list.
    "PythonSFunction": {
        "inputs": ["u"],
        "outputs": ["y"],
        "params": {
            "n_states": 1,
            "x0": "0",
            "code": (
                "# Define two functions: outputs(t, x, u, p) and derivatives(t, x, u, p)\n"
                "# Example: first-order system x' = -a*x + b*u; y = x\n"
                "import numpy as np\n"
                "def outputs(t, x, u, p):\n"
                "    return float(x[0])\n"
                "def derivatives(t, x, u, p):\n"
                "    a = float(p.get('a', 1.0))\n"
                "    b = float(p.get('b', 1.0))\n"
                "    return np.array([-a*x[0] + b*u], dtype=float)\n"
            ),
            # optional extra user params can be added and will be passed in p
            "a": 1.0,
            "b": 1.0,
        },
    },
}

def get_param(block: Dict[str, Any], name: str, default: Any) -> Any:
    return block.get("params", {}).get(name, default)

def out_key(block_id: str, port: str) -> Tuple[str, str]:
    return (block_id, port)

# ----------------------------
# Graph parsing & validation
# ----------------------------

class GraphError(Exception):
    pass

def parse_graph(g: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Wire], List[PortRef]]:
    blocks = g.get("blocks", [])
    wires = g.get("wires", [])
    scopes = g.get("scopes", [])

    blocks_by_id: Dict[str, Dict[str, Any]] = {}
    for b in blocks:
        bid = b.get("id")
        btype = b.get("type")
        if not bid or not btype:
            raise GraphError("Each block must have id and type.")
        if btype not in BLOCK_SPECS:
            raise GraphError(f"Unknown block type: {btype}")
        if bid in blocks_by_id:
            raise GraphError(f"Duplicate block id: {bid}")
        blocks_by_id[bid] = b

    # Parse scope signal refs
    scope_refs: List[PortRef] = []
    for s in scopes:
        for sig in s.get("signals", []):
            scope_refs.append(PortRef(block=sig["block"], port=sig["port"]))

    return blocks_by_id, wires, scope_refs

def validate_ports(blocks_by_id: Dict[str, Dict[str, Any]], wires: List[Wire]) -> None:
    """Ensure all wire endpoints exist and match port directions/spec.

    Note: Some blocks have dynamic inputs (e.g., Sum: u1..uN).
    """

    def _inputs_for(b: Dict[str, Any]) -> List[str]:
        btype = b["type"]
        if btype == "Sum":
            p = b.get("params", {}) or {}
            n = int(p.get("n", 0) or 0)
            if n <= 0:
                # fallback to signs length or default 2
                signs = str(p.get("signs", "++"))
                n = max(2, len(signs) if signs else 2)
            n = max(2, n)
            return [f"u{i}" for i in range(1, n + 1)]
        return list(BLOCK_SPECS[btype]["inputs"])

    def _outputs_for(b: Dict[str, Any]) -> List[str]:
        return list(BLOCK_SPECS[b["type"]]["outputs"])

    for w in wires:
        fr = w.get("from", {})
        to = w.get("to", {})
        a_id, a_port = fr.get("block"), fr.get("port")
        b_id, b_port = to.get("block"), to.get("port")

        if not a_id or not a_port or not b_id or not b_port:
            raise GraphError("Wire endpoints must have block and port.")

        if a_id not in blocks_by_id:
            raise GraphError(f"Wire source block missing: {a_id}")
        if b_id not in blocks_by_id:
            raise GraphError(f"Wire target block missing: {b_id}")

        a = blocks_by_id[a_id]
        b = blocks_by_id[b_id]

        if a_port not in _outputs_for(a):
            raise GraphError(f"Invalid source port: {a_id}({a['type']}).{a_port}")
        if b_port not in _inputs_for(b):
            raise GraphError(f"Invalid target port: {b_id}({b['type']}).{b_port}")

def build_input_map(wires: List[Wire]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Map (dst_block, dst_port) -> (src_block, src_port)
    Enforce single wire per input port in MVP.
    """
    m: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for w in wires:
        fr = w["from"]
        to = w["to"]
        dst = (to["block"], to["port"])
        src = (fr["block"], fr["port"])
        if dst in m:
            raise GraphError(f"Multiple wires into input {dst[0]}.{dst[1]} not allowed in MVP.")
        m[dst] = src
    return m

def validate_required_inputs(blocks_by_id: Dict[str, Dict[str, Any]], input_map: Dict[Tuple[str, str], Tuple[str, str]]) -> None:
    for bid, b in blocks_by_id.items():
        btype = b["type"]
        if btype == "Sum":
            p = b.get("params", {}) or {}
            n = int(p.get("n", 0) or 0)
            if n <= 0:
                signs = str(p.get("signs", "++"))
                n = max(2, len(signs) if signs else 2)
            n = max(2, n)
            req = [f"u{i}" for i in range(1, n + 1)]
        else:
            req = list(BLOCK_SPECS[btype]["inputs"])

        for inp in req:
            if (bid, inp) not in input_map:
                raise GraphError(f"Missing required input: {bid}({btype}).{inp}")

# ----------------------------
# Topological ordering (algebraic only) (algebraic only)
# ----------------------------

def build_dependency_graph(blocks_by_id: Dict[str, Dict[str, Any]], wires: List[Wire]) -> Dict[str, Set[str]]:
    """
    deps[B] contains set of blocks that B depends on (incoming edges from)
    """
    deps: Dict[str, Set[str]] = {bid: set() for bid in blocks_by_id.keys()}
    for w in wires:
        a = w["from"]["block"]
        b = w["to"]["block"]
        deps[b].add(a)
    return deps

def topo_sort_algebraic(blocks_by_id: Dict[str, Dict[str, Any]], deps: Dict[str, Set[str]]) -> List[str]:
    """
    Topologically sort only non-integrator blocks.
    Integrators are treated as state sources (their output is available immediately).
    Reject cycles among algebraic blocks ("pure algebraic cycles").
    """
    # Treat stateful blocks as sources (they break algebraic loops): Integrator + TransferFunction
    algebraic = {
        bid for bid, b in blocks_by_id.items()
        if b["type"] not in ("Integrator", "TransferFunction")
    }
    # compute in-degrees within algebraic subgraph, but ignore deps coming from integrators
    indeg: Dict[str, int] = {bid: 0 for bid in algebraic}
    forward: Dict[str, Set[str]] = {bid: set() for bid in algebraic}

    for b in algebraic:
        for a in deps[b]:
            if a in algebraic:
                indeg[b] += 1
                forward[a].add(b)

    q = [bid for bid, d in indeg.items() if d == 0]
    order: List[str] = []

    while q:
        n = q.pop()
        order.append(n)
        for m in forward[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    if len(order) != len(algebraic):
        # cycle among algebraic blocks
        raise GraphError("Pure algebraic cycle detected (not allowed in MVP). Add an Integrator/Delay to break the loop.")

    return order

# ----------------------------
# Runtime evaluation
# ----------------------------

def eval_block(
    bid: str,
    block: Dict[str, Any],
    t: float,
    signals: Dict[Tuple[str, str], float],
    input_map: Dict[Tuple[str, str], Tuple[str, str]],
    *,
    x: Optional[np.ndarray] = None,
    state_slices: Optional[Dict[str, slice]] = None,
    sfunc_exec: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    btype = block["type"]

    def inp(port: str) -> float:
        src = input_map[(bid, port)]
        return signals[out_key(src[0], src[1])]

    if btype == "Constant":
        value = float(get_param(block, "value", BLOCK_SPECS[btype]["params"]["value"]))
        signals[out_key(bid, "y")] = value

    elif btype == "Sine":
        amp = float(get_param(block, "amp", 1.0))
        freq_hz = float(get_param(block, "freq_hz", 1.0))
        phase = float(get_param(block, "phase", 0.0))
        signals[out_key(bid, "y")] = amp * np.sin(2.0 * np.pi * freq_hz * t + phase)

    elif btype == "Sum":
        p = block.get("params", {}) or {}
        signs = str(get_param(block, "signs", "++")).strip()
        n = int(p.get("n", 0) or 0)
        if n <= 0:
            n = max(2, len(signs) if signs else 2)
        n = max(2, n)

        # normalize signs to length n
        if not signs:
            signs = "+" * n
        if len(signs) < n:
            signs = signs + "+" * (n - len(signs))
        if len(signs) > n:
            signs = signs[:n]

        acc = 0.0
        for i in range(1, n + 1):
            sgn = -1.0 if signs[i - 1] == "-" else 1.0
            acc += sgn * float(inp(f"u{i}"))
        signals[out_key(bid, "y")] = float(acc)

    elif btype == "Gain":
        k = float(get_param(block, "k", 1.0))
        signals[out_key(bid, "y")] = float(k * inp("u"))

    elif btype == "Product":
        signals[out_key(bid, "y")] = float(inp("u1") * inp("u2"))

    elif btype == "Saturation":
        lo = float(get_param(block, "lower", -1.0))
        hi = float(get_param(block, "upper", 1.0))
        u = float(inp("u"))
        if lo > hi:
            lo, hi = hi, lo
        signals[out_key(bid, "y")] = float(min(max(u, lo), hi))

    elif btype == "PythonSFunction":
        if x is None or state_slices is None or sfunc_exec is None:
            raise GraphError("PythonSFunction evaluation requires state")
        sl = state_slices[bid]
        x_sf = x[sl].reshape(-1)
        u = float(inp("u"))
        p = dict(block.get("params", {}) or {})
        out_fn = sfunc_exec[bid]["outputs"]
        try:
            y = out_fn(float(t), x_sf, u, p)
        except Exception as e:
            raise GraphError(f"PythonSFunction {bid}: outputs() error: {type(e).__name__}: {e}")
        # accept scalar or length-1 vector
        if isinstance(y, (list, tuple, np.ndarray)):
            yv = float(np.array(y, dtype=float).reshape(-1)[0])
        else:
            yv = float(y)
        signals[out_key(bid, "y")] = yv

    else:
        raise GraphError(f"eval_block not implemented for type {btype}")

def compile_model(graph: Dict[str, Any]) -> Dict[str, Any]:
    blocks_by_id, wires, scope_refs = parse_graph(graph)
    validate_ports(blocks_by_id, wires)
    input_map = build_input_map(wires)
    validate_required_inputs(blocks_by_id, input_map)

    deps = build_dependency_graph(blocks_by_id, wires)
    algebraic_order = topo_sort_algebraic(blocks_by_id, deps)

    # Stateful blocks: Integrator + TransferFunction + PythonSFunction
    integrators = [bid for bid, b in blocks_by_id.items() if b["type"] == "Integrator"]
    tfs = [bid for bid, b in blocks_by_id.items() if b["type"] == "TransferFunction"]
    sfuncs = [bid for bid, b in blocks_by_id.items() if b["type"] == "PythonSFunction"]

    # Build a packed state vector: all integrators first, then each TF's internal states,
    # then PythonSFunction internal states.
    tf_ss: Dict[str, Dict[str, Any]] = {}
    sfunc_exec: Dict[str, Dict[str, Any]] = {}

    # helper to parse CSV coefficient lists
    def _parse_coeff_list(s: Any) -> List[float]:
        parts = [p.strip() for p in str(s or "").split(",") if p.strip()]
        if not parts:
            raise GraphError("TransferFunction: empty coefficient list")
        try:
            return [float(x) for x in parts]
        except Exception:
            raise GraphError("TransferFunction: coefficients must be numbers like '1, 2, 3'")

    # Integrator states
    state_slices: Dict[str, slice] = {}
    cursor = 0
    for bid in integrators:
        state_slices[bid] = slice(cursor, cursor + 1)
        cursor += 1

    # TransferFunction states (continuous-time)
    for bid in tfs:
        b = blocks_by_id[bid]
        num = _parse_coeff_list(get_param(b, "num", "1"))
        den = _parse_coeff_list(get_param(b, "den", "1,1"))
        if len(den) < 2:
            raise GraphError(f"TransferFunction {bid}: den must have order >= 1 (e.g. '1,1')")

        # Normalize to monic denominator for numerical stability
        if den[0] == 0:
            raise GraphError(f"TransferFunction {bid}: den[0] must be nonzero")
        d0 = den[0]
        den = [x / d0 for x in den]
        num = [x / d0 for x in num]

        # MVP restriction: strictly proper (no direct feedthrough), so deg(num) < deg(den)
        if len(num) >= len(den):
            raise GraphError(
                f"TransferFunction {bid}: numerator degree must be < denominator degree (strictly proper) in MVP."
            )

        # Pad numerator so tf2ss gets a proper polynomial format
        pad = (len(den) - 1) - len(num)
        if pad > 0:
            num = [0.0] * pad + num

        A, B, C, D = tf2ss(num, den)
        # Enforce D ~ 0 (strictly proper). We still accept tiny numerical noise.
        if np.max(np.abs(D)) > 1e-9:
            raise GraphError(
                f"TransferFunction {bid}: direct feedthrough (D != 0) not supported in MVP. Make the TF strictly proper."
            )

        n = A.shape[0]
        state_slices[bid] = slice(cursor, cursor + n)
        cursor += n
        tf_ss[bid] = {"A": A, "B": B, "C": C}

    # PythonSFunction states + code compilation
    def _parse_x0_any(x0v: Any, n_states: int) -> np.ndarray:
        if n_states <= 0:
            return np.zeros((0,), dtype=float)
        # allow scalar, list, or CSV string
        if isinstance(x0v, (int, float)):
            return np.full((n_states,), float(x0v), dtype=float)
        if isinstance(x0v, (list, tuple)):
            arr = np.array([float(v) for v in x0v], dtype=float).reshape(-1)
            if arr.size == 1 and n_states > 1:
                return np.full((n_states,), float(arr[0]), dtype=float)
            if arr.size != n_states:
                raise GraphError(f"PythonSFunction: x0 size {arr.size} does not match n_states={n_states}")
            return arr
        parts = [p.strip() for p in str(x0v or "").split(",") if p.strip()]
        if not parts:
            return np.zeros((n_states,), dtype=float)
        arr = np.array([float(v) for v in parts], dtype=float).reshape(-1)
        if arr.size == 1 and n_states > 1:
            return np.full((n_states,), float(arr[0]), dtype=float)
        if arr.size != n_states:
            raise GraphError(f"PythonSFunction: x0 size {arr.size} does not match n_states={n_states}")
        return arr

    def _compile_sfunc(code: str) -> Dict[str, Any]:
        """Compile user code with a small, explicit API."""
        if not isinstance(code, str) or not code.strip():
            raise GraphError("PythonSFunction: code is empty")

        # Restrict builtins: enough for basic math, but avoid file/network.
        safe_builtins = dict(_SAFE_BUILTINS)
        env: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "np": np,
        }
        loc: Dict[str, Any] = {}
        try:
            compiled = compile(code, "<PythonSFunction>", "exec")
            exec(compiled, env, loc)
        except Exception as e:
            raise GraphError(f"PythonSFunction: code error: {type(e).__name__}: {e}")

        out_fn = loc.get("outputs") or env.get("outputs")
        der_fn = loc.get("derivatives") or env.get("derivatives")
        if not callable(out_fn) or not callable(der_fn):
            raise GraphError("PythonSFunction: code must define callables 'outputs' and 'derivatives'")
        return {"outputs": out_fn, "derivatives": der_fn}

    for bid in sfuncs:
        b = blocks_by_id[bid]
        n_states = int(get_param(b, "n_states", 1) or 1)
        if n_states < 0:
            raise GraphError(f"PythonSFunction {bid}: n_states must be >= 0")
        state_slices[bid] = slice(cursor, cursor + n_states)
        cursor += n_states

        code = str(get_param(b, "code", ""))
        sfunc_exec[bid] = _compile_sfunc(code)

    state_index = {bid: i for i, bid in enumerate(integrators)}

    # initial state vector from x0 params
    x0 = np.zeros(cursor, dtype=float)
    for bid in integrators:
        sl = state_slices[bid]
        x0[sl.start] = float(get_param(blocks_by_id[bid], "x0", 0.0))
    # TF initial states default to zeros; you can add x0 later as an enhancement

    # PythonSFunction initial states
    for bid in sfuncs:
        sl = state_slices[bid]
        n_states = sl.stop - sl.start
        if n_states <= 0:
            continue
        b = blocks_by_id[bid]
        x0_vec = _parse_x0_any(get_param(b, "x0", "0"), n_states)
        x0[sl] = x0_vec

    # validate scopes (optional)
    for ref in scope_refs:
        if ref.block not in blocks_by_id:
            raise GraphError(f"Scope references unknown block {ref.block}")
        btype = blocks_by_id[ref.block]["type"]
        if ref.port not in BLOCK_SPECS[btype]["outputs"]:
            raise GraphError(f"Scope references invalid output port {ref.block}.{ref.port}")

    compiled = {
        "blocks_by_id": blocks_by_id,
        "input_map": input_map,
        "algebraic_order": algebraic_order,
        "integrators": integrators,
        "tfs": tfs,
        "sfuncs": sfuncs,
        "state_index": state_index,
        "state_slices": state_slices,
        "tf_ss": tf_ss,
        "sfunc_exec": sfunc_exec,
        "x0": x0,
        "scope_refs": scope_refs,
    }
    return compiled

def make_rhs(compiled: Dict[str, Any]) -> Callable[[float, np.ndarray], np.ndarray]:
    blocks_by_id = compiled["blocks_by_id"]
    input_map = compiled["input_map"]
    algebraic_order = compiled["algebraic_order"]
    integrators = compiled["integrators"]
    tfs = compiled["tfs"]
    sfuncs = compiled.get("sfuncs", [])
    state_slices = compiled["state_slices"]
    tf_ss = compiled["tf_ss"]
    sfunc_exec = compiled.get("sfunc_exec", {})

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        signals: Dict[Tuple[str, str], float] = {}

        # Integrator outputs from state
        for bid in integrators:
            sl = state_slices[bid]
            signals[out_key(bid, "y")] = float(x[sl.start])

        # TransferFunction outputs from state (strictly proper: y = Cx)
        for bid in tfs:
            sl = state_slices[bid]
            x_tf = x[sl]
            C = tf_ss[bid]["C"]
            y = float((C @ x_tf.reshape(-1, 1))[0, 0])
            signals[out_key(bid, "y")] = y

        # compute algebraic blocks in topo order
        for bid in algebraic_order:
            eval_block(
                bid,
                blocks_by_id[bid],
                t,
                signals,
                input_map,
                x=x,
                state_slices=state_slices,
                sfunc_exec=sfunc_exec,
            )

        # compute derivatives for integrators
        dx = np.zeros_like(x)

        # Integrator derivatives: x' = u
        for bid in integrators:
            sl = state_slices[bid]
            src = input_map[(bid, "u")]
            u = float(signals[out_key(src[0], src[1])])
            dx[sl.start] = u

        # TransferFunction derivatives: x' = A x + B u
        for bid in tfs:
            sl = state_slices[bid]
            src = input_map[(bid, "u")]
            u = float(signals[out_key(src[0], src[1])])
            A = tf_ss[bid]["A"]
            B = tf_ss[bid]["B"]
            x_tf = x[sl].reshape(-1, 1)
            dx_tf = (A @ x_tf) + (B * u)
            dx[sl] = dx_tf.flatten()

        # PythonSFunction derivatives: dx = derivatives(t, x_sf, u, p)
        for bid in sfuncs:
            sl = state_slices[bid]
            n_states = sl.stop - sl.start
            if n_states <= 0:
                continue
            src = input_map[(bid, "u")]
            u = float(signals[out_key(src[0], src[1])])
            x_sf = x[sl].reshape(-1)
            p = dict(blocks_by_id[bid].get("params", {}) or {})
            der_fn = sfunc_exec[bid]["derivatives"]
            try:
                dx_sf = der_fn(float(t), x_sf, u, p)
            except Exception as e:
                raise GraphError(f"PythonSFunction {bid}: derivatives() error: {type(e).__name__}: {e}")
            arr = np.array(dx_sf, dtype=float).reshape(-1)
            if arr.size == 1 and n_states > 1:
                arr = np.full((n_states,), float(arr[0]), dtype=float)
            if arr.size != n_states:
                raise GraphError(f"PythonSFunction {bid}: derivatives() returned size {arr.size}, expected {n_states}")
            dx[sl] = arr
        return dx

    return rhs


def _integrate_fixed_step(rhs: Callable[[float, np.ndarray], np.ndarray],
                          x0: np.ndarray,
                          t_eval: np.ndarray,
                          method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Simple fixed-step explicit integrators.

    Returns: (t, Y) where Y shape is (n_states, n_times) like solve_ivp.y.
    """
    method = (method or "rk4").lower()
    n = x0.size
    m = t_eval.size
    Y = np.zeros((n, m), dtype=float)
    Y[:, 0] = x0
    for k in range(m - 1):
        t = float(t_eval[k])
        h = float(t_eval[k + 1] - t_eval[k])
        x = Y[:, k]

        if method in ("euler", "fe", "forwardeuler"):
            k1 = rhs(t, x)
            x_next = x + h * k1

        elif method in ("rk2", "heun", "improvedeuler"):
            k1 = rhs(t, x)
            k2 = rhs(t + h, x + h * k1)
            x_next = x + (h * 0.5) * (k1 + k2)

        elif method in ("rk4",):
            k1 = rhs(t, x)
            k2 = rhs(t + 0.5 * h, x + 0.5 * h * k1)
            k3 = rhs(t + 0.5 * h, x + 0.5 * h * k2)
            k4 = rhs(t + h, x + h * k3)
            x_next = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        else:
            raise GraphError(f"Unknown fixed-step solver: {method}")

        Y[:, k + 1] = x_next

    return t_eval, Y

def simulate(graph: Dict[str, Any], t0: float, t1: float, dt: float, solver: str = 'rk45') -> Dict[str, Any]:
    compiled = compile_model(graph)
    rhs = make_rhs(compiled)

    x0 = compiled["x0"]
    t_eval = np.arange(t0, t1 + 1e-12, dt)

    solver_l = (solver or 'rk45').lower()
    if solver_l in ('euler','rk2','heun','rk4'):
        t_out, y_out = _integrate_fixed_step(rhs, x0, t_eval, solver_l)
        sol = type('Sol', (), {})()
        sol.t = t_out
        sol.y = y_out
        sol.success = True
        sol.message = f'fixed-step {solver_l}'
    else:
        # Delegate to SciPy solve_ivp (variable-step). SciPy expects specific method names
        # like 'RK45', 'Radau', 'BDF', ... (case-sensitive).
        method_map = {
            "rk23": "RK23",
            "rk45": "RK45",
            "dop853": "DOP853",
            "radau": "Radau",
            "bdf": "BDF",
            "lsoda": "LSODA",
        }
        method = method_map.get(solver_l, solver)
        sol = solve_ivp(rhs, (t0, t1), x0, t_eval=t_eval, method=method, rtol=1e-6, atol=1e-9)


    # Recompute logged signals at each t_eval using the same evaluation
    # (Simple approach; for MVP size it’s fine.)
    blocks_by_id = compiled["blocks_by_id"]
    input_map = compiled["input_map"]
    algebraic_order = compiled["algebraic_order"]
    integrators = compiled["integrators"]
    tfs = compiled["tfs"]
    sfuncs = compiled.get("sfuncs", [])
    state_slices = compiled["state_slices"]
    tf_ss = compiled["tf_ss"]
    sfunc_exec = compiled.get("sfunc_exec", {})
    scope_refs: List[PortRef] = compiled["scope_refs"]

    logs: Dict[str, List[float]] = {}
    def sig_name(ref: PortRef) -> str:
        return f"{ref.block}.{ref.port}"

    for ref in scope_refs:
        logs[sig_name(ref)] = []

    for k, t in enumerate(sol.t):
        x = sol.y[:, k]
        signals: Dict[Tuple[str, str], float] = {}

        # Integrator outputs
        for bid in integrators:
            sl = state_slices[bid]
            signals[out_key(bid, "y")] = float(x[sl.start])

        # TransferFunction outputs (strictly proper)
        for bid in tfs:
            sl = state_slices[bid]
            x_tf = x[sl]
            C = tf_ss[bid]["C"]
            y = float((C @ x_tf.reshape(-1, 1))[0, 0])
            signals[out_key(bid, "y")] = y

        for bid in algebraic_order:
            eval_block(
                bid,
                blocks_by_id[bid],
                float(t),
                signals,
                input_map,
                x=x,
                state_slices=state_slices,
                sfunc_exec=sfunc_exec,
            )

        for ref in scope_refs:
            logs[sig_name(ref)].append(float(signals[out_key(ref.block, ref.port)]))

    return {
        "solver": solver,
        "t": sol.t.tolist(),
        "x": sol.y.T.tolist(),  # list of state vectors over time
        "logs": logs,
        "integrators": integrators,
        "success": bool(sol.success),
        "message": sol.message,
    }


# ----------------------------
# Simple Python Exec endpoint (sandboxed)
# ----------------------------

import io
import ast
import contextlib

def _run_user_code(code: str) -> Dict[str, Any]:
    """Execute user code in a restricted environment.

    Returns dict with: stdout, result (repr), error (optional), ok (bool).
    """
    if not isinstance(code, str):
        raise BadRequest("Field 'code' must be a string.")
    if len(code) > 20000:
        raise BadRequest("Code too large (max 20000 characters).")

    # Restricted builtins + allowed imports (same as PythonSFunction)
    env: Dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS, **{
            "print": print,  # allow printing
        }),
        "np": np,
    }
    # Provide math as a convenience module (allowed by _safe_import)
    try:
        import math  # noqa: F401
        env["math"] = math
    except Exception:
        pass

    out = io.StringIO()

    # Emulate REPL: if the last statement is an expression, evaluate it and return its repr
    result_obj = None
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop().value
        body_mod = ast.Module(body=tree.body, type_ignores=[])
        compiled_body = compile(body_mod, "<pyexec>", "exec")
        compiled_expr = compile(ast.Expression(last_expr), "<pyexec>", "eval") if last_expr is not None else None

        with contextlib.redirect_stdout(out):
            exec(compiled_body, env, env)
            if compiled_expr is not None:
                result_obj = eval(compiled_expr, env, env)

        return {
            "ok": True,
            "stdout": out.getvalue(),
            "result": None if result_obj is None else repr(result_obj),
        }
    except Exception as e:
        return {
            "ok": False,
            "stdout": out.getvalue(),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


# ----------------------------
# Flask routes
# ----------------------------

@app.route("/compile", methods=["POST"])
def api_compile():
    try:
        graph = request.get_json(force=True, silent=False)
        c = compile_model(graph)
        return jsonify({
            "integrators": c["integrators"],
            "state_index": c["state_index"],
            "algebraic_order": c["algebraic_order"],
            "x0": c["x0"].tolist(),
        })
    except BadRequest as e:
        # Happens when request body is not valid JSON or Content-Type mismatches
        return jsonify({"error": f"Bad JSON request: {e}"}), 400
    except GraphError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

@app.route("/simulate", methods=["POST"])
def api_simulate():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request JSON must be an object."}), 400

        if "graph" not in payload:
            return jsonify({"error": "Missing field: graph"}), 400

        graph = payload["graph"]
        t0 = float(payload.get("t0", 0.0))
        t1 = float(payload.get("t1", 10.0))
        dt = float(payload.get("dt", 0.01))

        # basic validation
        if dt <= 0:
            return jsonify({"error": "dt must be > 0"}), 400
        if t1 < t0:
            return jsonify({"error": "t1 must be >= t0"}), 400

        solver = str(payload.get('solver', 'rk45'))

        return jsonify(simulate(graph, t0, t1, dt, solver=solver))

    except BadRequest as e:
        return jsonify({"error": f"Bad JSON request: {e}"}), 400
    except GraphError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400


@app.route("/pyexec", methods=["POST"])
def api_pyexec():
    """Execute small Python snippets from the UI (sandboxed)."""
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request JSON must be an object."}), 400
        code = payload.get("code", "")
        res = _run_user_code(code)
        return jsonify(res)
    except BadRequest as e:
        return jsonify({"ok": False, "error": f"Bad request: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}), 400


from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory(app.root_path, "index.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")

if __name__ == "__main__":
    app.run(debug=True)
    
    