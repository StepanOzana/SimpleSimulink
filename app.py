from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional, Set
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import traceback
import numpy as np
from scipy.integrate import solve_ivp

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
        "params": {},
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
    # ensure all wire endpoints exist and match port directions/spec
    for w in wires:
        fr = w.get("from", {})
        to = w.get("to", {})
        a_id, a_port = fr.get("block"), fr.get("port")
        b_id, b_port = to.get("block"), to.get("port")
        if a_id not in blocks_by_id or b_id not in blocks_by_id:
            raise GraphError("Wire references unknown block id.")
        a_type = blocks_by_id[a_id]["type"]
        b_type = blocks_by_id[b_id]["type"]
        if a_port not in BLOCK_SPECS[a_type]["outputs"]:
            raise GraphError(f"Invalid output port {a_port} on block {a_id}:{a_type}")
        if b_port not in BLOCK_SPECS[b_type]["inputs"]:
            raise GraphError(f"Invalid input port {b_port} on block {b_id}:{b_type}")

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
        spec = BLOCK_SPECS[b["type"]]
        for inp in spec["inputs"]:
            if (bid, inp) not in input_map:
                raise GraphError(f"Missing required input: {bid}({b['type']}).{inp}")

# ----------------------------
# Topological ordering (algebraic only)
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
    algebraic = {bid for bid, b in blocks_by_id.items() if b["type"] != "Integrator"}
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
        signals[out_key(bid, "y")] = float(inp("u1") + inp("u2"))

    elif btype == "Gain":
        k = float(get_param(block, "k", 1.0))
        signals[out_key(bid, "y")] = float(k * inp("u"))

    elif btype == "Product":
        signals[out_key(bid, "y")] = float(inp("u1") * inp("u2"))

    else:
        raise GraphError(f"eval_block not implemented for type {btype}")

def compile_model(graph: Dict[str, Any]) -> Dict[str, Any]:
    blocks_by_id, wires, scope_refs = parse_graph(graph)
    validate_ports(blocks_by_id, wires)
    input_map = build_input_map(wires)
    validate_required_inputs(blocks_by_id, input_map)

    deps = build_dependency_graph(blocks_by_id, wires)
    algebraic_order = topo_sort_algebraic(blocks_by_id, deps)

    integrators = [bid for bid, b in blocks_by_id.items() if b["type"] == "Integrator"]
    state_index = {bid: i for i, bid in enumerate(integrators)}

    # initial state vector from x0 params
    x0 = np.zeros(len(integrators), dtype=float)
    for bid, i in state_index.items():
        x0[i] = float(get_param(blocks_by_id[bid], "x0", 0.0))

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
        "state_index": state_index,
        "x0": x0,
        "scope_refs": scope_refs,
    }
    return compiled

def make_rhs(compiled: Dict[str, Any]) -> Callable[[float, np.ndarray], np.ndarray]:
    blocks_by_id = compiled["blocks_by_id"]
    input_map = compiled["input_map"]
    algebraic_order = compiled["algebraic_order"]
    integrators = compiled["integrators"]
    state_index = compiled["state_index"]

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        signals: Dict[Tuple[str, str], float] = {}

        # integrator outputs from state
        for bid in integrators:
            i = state_index[bid]
            signals[out_key(bid, "y")] = float(x[i])

        # compute algebraic blocks in topo order
        for bid in algebraic_order:
            eval_block(bid, blocks_by_id[bid], t, signals, input_map)

        # compute derivatives for integrators
        dx = np.zeros_like(x)
        for bid in integrators:
            i = state_index[bid]
            # integrator input u must be wired
            src = input_map[(bid, "u")]
            u = signals[out_key(src[0], src[1])]
            dx[i] = float(u)
        return dx

    return rhs

def simulate(graph: Dict[str, Any], t0: float, t1: float, dt: float) -> Dict[str, Any]:
    compiled = compile_model(graph)
    rhs = make_rhs(compiled)

    x0 = compiled["x0"]
    t_eval = np.arange(t0, t1 + 1e-12, dt)

    sol = solve_ivp(rhs, (t0, t1), x0, t_eval=t_eval, rtol=1e-6, atol=1e-9)

    # Recompute logged signals at each t_eval using the same evaluation
    # (Simple approach; for MVP size it’s fine.)
    blocks_by_id = compiled["blocks_by_id"]
    input_map = compiled["input_map"]
    algebraic_order = compiled["algebraic_order"]
    integrators = compiled["integrators"]
    state_index = compiled["state_index"]
    scope_refs: List[PortRef] = compiled["scope_refs"]

    logs: Dict[str, List[float]] = {}
    def sig_name(ref: PortRef) -> str:
        return f"{ref.block}.{ref.port}"

    for ref in scope_refs:
        logs[sig_name(ref)] = []

    for k, t in enumerate(sol.t):
        x = sol.y[:, k]
        signals: Dict[Tuple[str, str], float] = {}

        for bid in integrators:
            signals[out_key(bid, "y")] = float(x[state_index[bid]])

        for bid in algebraic_order:
            eval_block(bid, blocks_by_id[bid], float(t), signals, input_map)

        for ref in scope_refs:
            logs[sig_name(ref)].append(float(signals[out_key(ref.block, ref.port)]))

    return {
        "t": sol.t.tolist(),
        "x": sol.y.T.tolist(),  # list of state vectors over time
        "logs": logs,
        "integrators": integrators,
        "success": bool(sol.success),
        "message": sol.message,
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

        return jsonify(simulate(graph, t0, t1, dt))

    except BadRequest as e:
        return jsonify({"error": f"Bad JSON request: {e}"}), 400
    except GraphError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400

from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
    
    
