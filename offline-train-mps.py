#!/usr/bin/env python3
"""Train a Pine v8 *challenger* locally with PyTorch on Apple Silicon.

The browser stays the source of truth: this script only prepares a challenger
checkpoint. Importing it in Pine Autopilot v8 starts a fresh, blind tournament
against the browser's frozen champion before the challenger can be promoted.
Nothing is uploaded or sent to a game server.

Example:
  python3 offline-train-mps.py checkpoint.json training.json \
    --output pine-autopilot-challenger.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit(
        "This trainer needs NumPy and PyTorch. Install a current PyTorch build, "
        "then re-run this command. Original error: " + str(exc)
    ) from exc


PHASES = ("early", "mid", "late", "hell")
PHASE_SHARES = {"early": 0.30, "mid": 0.30, "late": 0.23, "hell": 0.17}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def unpack_f32(value) -> np.ndarray:
    """Decode the Float32/base64 representation emitted by the userscript."""
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if not isinstance(value, dict) or not isinstance(value.get("float32"), str):
        raise ValueError("missing packed Float32 tensor")
    return np.frombuffer(base64.b64decode(value["float32"]), dtype="<f4").copy()


def pack_f32(values: np.ndarray) -> dict:
    raw = np.asarray(values, dtype="<f4").tobytes(order="C")
    return {"float32": base64.b64encode(raw).decode("ascii")}


class DuelingNet(nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int):
        super().__init__()
        self.hidden = nn.Linear(inputs, hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(x))
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=1, keepdim=True)


def network_from_json(record: dict, device: torch.device) -> DuelingNet:
    inputs = int(record["inputs"])
    hidden = int(record["hidden"])
    outputs = int(record["outputs"])
    model = DuelingNet(inputs, hidden, outputs).to(device)
    with torch.no_grad():
        model.hidden.weight.copy_(torch.tensor(unpack_f32(record["w1"]).reshape(hidden, inputs), device=device))
        model.hidden.bias.copy_(torch.tensor(unpack_f32(record["b1"]), device=device))
        model.value.weight.copy_(torch.tensor(unpack_f32(record["wValue"]).reshape(1, hidden), device=device))
        model.value.bias.copy_(torch.tensor([float(record["bValue"])], device=device))
        model.advantage.weight.copy_(torch.tensor(unpack_f32(record["wAdvantage"]).reshape(outputs, hidden), device=device))
        model.advantage.bias.copy_(torch.tensor(unpack_f32(record["bAdvantage"]), device=device))
    return model


def network_to_json(record: dict, model: DuelingNet, added_steps: int) -> None:
    """Write tensors in exactly the shape and ordering used by the JS network."""
    with torch.no_grad():
        record["w1"] = pack_f32(model.hidden.weight.detach().cpu().numpy().reshape(-1))
        record["b1"] = pack_f32(model.hidden.bias.detach().cpu().numpy())
        record["wValue"] = pack_f32(model.value.weight.detach().cpu().numpy().reshape(-1))
        record["bValue"] = float(model.value.bias.detach().cpu().item())
        record["wAdvantage"] = pack_f32(model.advantage.weight.detach().cpu().numpy().reshape(-1))
        record["bAdvantage"] = pack_f32(model.advantage.bias.detach().cpu().numpy())
        record["samples"] = int(record.get("samples", 0)) + added_steps


def head_for(model: dict, phase: str) -> dict:
    return model if phase == "early" else model[f"{phase}Head"]


def clean_transition(row: dict, state_inputs: int, action_count: int) -> dict | None:
    if not isinstance(row, dict) or row.get("kind") != "movement":
        return None
    state = row.get("input")
    if not isinstance(state, list) or len(state) != state_inputs:
        return None
    action = row.get("actionIndex")
    reward = row.get("reward")
    if not isinstance(action, int) or not 0 <= action < action_count or not isinstance(reward, (int, float)):
        return None
    next_state = row.get("nextState")
    if not isinstance(next_state, list) or len(next_state) != state_inputs:
        next_state = state
    values = list(state) + list(next_state)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return None
    return {
        "state": state,
        "next": next_state,
        "action": action,
        "reward": float(reward),
        "done": bool(row.get("done")),
        "steps": max(1, int(row.get("steps", 1))),
        "phase": row.get("phase") if row.get("phase") in PHASES else "early",
    }


def load_transitions(training: dict, state_inputs: int, action_count: int) -> dict[str, list[dict]]:
    by_phase = {phase: [] for phase in PHASES}
    # Elite trajectories deliberately appear twice: this mirrors v8's 45%
    # elite sampling while keeping each browser phase head separate.
    rows = list(training.get("recentReplay") or []) + list(training.get("eliteReplay") or []) * 2
    for row in rows:
        transition = clean_transition(row, state_inputs, action_count)
        if transition:
            by_phase[transition["phase"]].append(transition)
    return by_phase


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is unavailable to this PyTorch build.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_phase(
    phase: str,
    rows: list[dict],
    head: dict,
    steps: int,
    batch_size: int,
    learning_rate: float,
    discount: float,
    device: torch.device,
) -> tuple[float, int]:
    if not rows or steps <= 0:
        return 0.0, 0
    online = network_from_json(head["movementNet"], device)
    target_record = head.get("movementTarget") or head["movementNet"]
    target = network_from_json(target_record, device).eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=learning_rate, weight_decay=1e-5)
    total_loss = 0.0
    gamma_cache = {n: discount ** n for n in range(1, 33)}

    for step in range(steps):
        batch = random.choices(rows, k=batch_size)
        states = torch.tensor(np.asarray([row["state"] for row in batch], dtype=np.float32), device=device)
        next_states = torch.tensor(np.asarray([row["next"] for row in batch], dtype=np.float32), device=device)
        actions = torch.tensor([row["action"] for row in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([row["reward"] for row in batch], dtype=torch.float32, device=device)
        done = torch.tensor([row["done"] for row in batch], dtype=torch.bool, device=device)
        powers = torch.tensor([gamma_cache.get(row["steps"], discount ** row["steps"]) for row in batch], dtype=torch.float32, device=device)

        values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = online(next_states).argmax(dim=1)
            next_values = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected = rewards + powers * next_values * (~done)
        loss = F.smooth_l1_loss(values, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        # The browser uses a slowly updated target. Refresh it twice during a
        # short offline pass rather than allowing a stale target to drift.
        if (step + 1) % max(1, steps // 2) == 0:
            target.load_state_dict(online.state_dict())

    network_to_json(head["movementNet"], online, steps)
    # Set the imported target equal to the new online model. The browser will
    # resume its normal 1,500-step target synchronization thereafter.
    head["movementTarget"] = copy.deepcopy(head["movementNet"])
    return total_loss / steps, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Pine Autopilot v8 challenger locally with PyTorch/MPS.")
    parser.add_argument("checkpoint", type=Path, help="MPS checkpoint downloaded from the v8 panel")
    parser.add_argument("training", type=Path, help="training JSON downloaded from the v7 or v8 panel")
    parser.add_argument("--output", type=Path, default=Path("pine-autopilot-challenger.json"))
    parser.add_argument("--steps", type=int, default=1200, help="total gradient steps across phase heads (default: 1200)")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("--steps, --batch-size, and --learning-rate must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = read_json(args.checkpoint)
    training = read_json(args.training)
    if checkpoint.get("format") != "pine-autopilot-checkpoint-v8" or not isinstance(checkpoint.get("model"), dict):
        raise SystemExit("Checkpoint must be a pine-autopilot-checkpoint-v8 file from the v8 panel.")
    if training.get("format") not in {"pine-autopilot-training-v7", "pine-autopilot-training-v8"}:
        raise SystemExit("Training file must be a pine-autopilot-training-v7 or v8 export.")

    contract = checkpoint.get("contract") or {}
    state_inputs = int(contract.get("stateInputs", 0))
    actions = contract.get("movementActions") or []
    if state_inputs < 1 or not actions:
        raise SystemExit("Checkpoint is missing its model contract.")
    model = checkpoint["model"]
    for phase in PHASES:
        head = head_for(model, phase)
        network = head.get("movementNet") if isinstance(head, dict) else None
        if not isinstance(network, dict) or int(network.get("inputs", -1)) != state_inputs or int(network.get("outputs", -1)) != len(actions):
            raise SystemExit(f"Checkpoint's {phase} movement head is incompatible.")

    by_phase = load_transitions(training, state_inputs, len(actions))
    available = [phase for phase in PHASES if by_phase[phase]]
    if not available:
        raise SystemExit("The training export has no usable movement transitions.")
    device = choose_device(args.device)
    print(f"Device: {device.type}; usable transitions: {dict(Counter({p: len(v) for p, v in by_phase.items()}))}")
    share_total = sum(PHASE_SHARES[phase] for phase in available)
    allocations = {
        phase: max(1, round(args.steps * PHASE_SHARES[phase] / share_total)) if phase in available else 0
        for phase in PHASES
    }
    # Correct rounding while preserving the phase intent.
    while sum(allocations.values()) > args.steps:
        phase = max(available, key=lambda p: allocations[p])
        allocations[phase] -= 1
    while sum(allocations.values()) < args.steps:
        phase = max(available, key=lambda p: PHASE_SHARES[p])
        allocations[phase] += 1

    discount = float(contract.get("discount", 0.985))
    for phase in PHASES:
        loss, completed = train_phase(
            phase, by_phase[phase], head_for(model, phase), allocations[phase], args.batch_size,
            args.learning_rate, discount, device,
        )
        if completed:
            print(f"{phase:>5}: {completed:4d} steps on {len(by_phase[phase]):4d} transitions; mean Huber loss {loss:.5f}")
        else:
            print(f"{phase:>5}: skipped (no phase-matched transitions)")

    checkpoint["exportedAt"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    checkpoint["offlineTraining"] = {
        "tool": "offline-train-mps.py",
        "device": device.type,
        "sourceTrainingFormat": training["format"],
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "phaseTransitions": {phase: len(by_phase[phase]) for phase in PHASES},
        "phaseSteps": allocations,
    }
    args.output.write_text(json.dumps(checkpoint, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote challenger checkpoint: {args.output}")
    print("Import it through the central v8 Pine tab. It will be tested against the frozen champion before promotion.")


if __name__ == "__main__":
    main()
