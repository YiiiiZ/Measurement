from __future__ import annotations
from typing import Dict, Any, Iterable
import json

class ResultSaver:
    def __init__(self):
        # Internal layout: {"Q1": {...}, "Q2": {...}}
        self.results: Dict[str, Dict[str, Any]] = {}

    # ---------- write ----------
    def save_qubit(self, label: str, **vals) -> None:
        """Create or update results for a qubit (Q1, Q2, ...)."""
        cur = self.results.get(label, {})
        cur.update(vals)
        self.results[label] = cur

    # ---------- read ----------
    def get_qubit(self, label: str) -> Dict[str, Any]:
        """Return a *copy* of the stored dict for the qubit."""
        if label not in self.results:
            raise KeyError(f"No results for {label}")
        return dict(self.results[label])

    def getf(self, label: str, field: str) -> Any:
        """Fetch a single field from a qubit's results."""
        return self.get_qubit(label)[field]

    def inputs(self, label: str, fields: Iterable[str]) -> Dict[str, Any]:
        """Return {field: value} for a qubit."""
        d = self.get_qubit(label)
        return {k: d[k] for k in fields}

    def exists(self, label: str) -> bool:
        """Check if a qubit label is present."""
        return label in self.results

    def list_qubits(self) -> list[str]:
        """List all stored qubit labels, e.g., ['Q1', 'Q2']."""
        return sorted(self.results.keys())

    def require(self, label: str, fields: Iterable[str]) -> None:
        """Raise if any of the requested fields are missing for a qubit."""
        d = self.get_qubit(label)
        missing = [f for f in fields if f not in d]
        if missing:
            raise KeyError(f"{label} missing fields: {missing}")

    # ---------- optional: disk persistence ----------
    def save(self, path: str) -> None:
        """Save all results to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ResultSaver":
        """Load results from a JSON file and return a ResultSaver."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        # trust keys/values as-is (you can add sanity checks here if desired)
        obj.results = {str(k): dict(v) for k, v in data.items()}
        return obj


# -------------------------
# Example/Usage
# -------------------------
# rs = ResultSaver()

# # store/update
# rs.save_qubit(
#     "Q1",
#     qb_freq=5.123e9,
#     res_freq_dressed=6.789e9,
#     res_freq=6.800e9,
#     pi_pulse_range=-10,
#     pi_pulse_amp=0.18,
#     pi_pulse_length=80e-9,
#     T1=22e-6,
#     RamseyT2=9e-6,
#     EchoT2=14e-6,
# )
# rs.save_qubit("Q1", qb_freq=5.1245e9)  # partial update

# # use as inputs to a new experiment
# fq = rs.getf("Q1", "qb_freq")
# fr = rs.getf("Q1", "res_freq")
# params = rs.inputs("Q1", ["qb_freq", "res_freq"])
# print("Q1 qb_freq:", fq)
# print("Q1 res_freq:", fr)
# print("Q1 inputs:", params)

# # optional persistence
# rs.save("cal_results.json")
# rs2 = ResultSaver.load("cal_results.json")
# print("Loaded qubits:", rs2.list_qubits())