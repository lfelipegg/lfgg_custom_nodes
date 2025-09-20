import random
from decimal import Decimal, getcontext
from typing import List, Sequence, Tuple

getcontext().prec = 12


class ModelMergeCombos:
    """Generate deterministic combinations of three floats for model merging.

    The node iterates over a 3D grid of values in the range [0, 1] defined by
    the chosen step. Instead of materialising the full cartesian product, we
    index into the grid directly which is far more memory efficient. Extra
    controls expose sequencing direction, skipping, batching, normalisation and
    basic sum-range filtering. Metadata such as the total number of combinations
    and the current index help downstream flow-control nodes.
    """

    def __init__(self):
        self._state = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_c": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9_999_999, "step": 1}),
                "direction": ("STRING", {"default": "ascending", "choices": ["ascending", "descending", "shuffle"]}),
                "skip": ("INT", {"default": 1, "min": 1, "max": 999}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 20}),
                "normalize": ("BOOLEAN", {"default": False}),
                "min_sum": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "max_sum": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("a", "b", "c", "seed", "index", "total", "looped", "batch")
    FUNCTION = "next_combo"
    CATEGORY = "utils"

    # region public entrypoint -------------------------------------------------
    def next_combo(
        self,
        start_a: float,
        start_b: float,
        start_c: float,
        step: float,
        seed: int,
        direction: str,
        skip: int,
        batch_size: int,
        normalize: bool,
        min_sum: float,
        max_sum: float,
    ):
        try:
            return self._next_combo_impl(
                start_values=(start_a, start_b, start_c),
                step=step,
                seed=seed,
                direction=direction,
                skip=skip,
                batch_size=batch_size,
                normalize=normalize,
                min_sum=min_sum,
                max_sum=max_sum,
            )
        except ValueError as exc:  # input validation fallback
            message = f"ModelMergeCombos error: {exc}"
            self._state = None
            safe_a = self._clamp(start_a)
            safe_b = self._clamp(start_b)
            safe_c = self._clamp(start_c)
            return safe_a, safe_b, safe_c, seed, 0, 0, False, message

    # endregion -----------------------------------------------------------------

    # region implementation helpers --------------------------------------------
    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _precision_from_step(step: float) -> int:
        step_str = f"{step:.8f}".rstrip("0").rstrip(".")
        if "." not in step_str:
            return 0
        precision = len(step_str.split(".")[1])
        return max(0, min(precision, 6))

    @classmethod
    def _grid_from_step(cls, step: float) -> Tuple[List[float], int]:
        if step <= 0:
            raise ValueError("step must be greater than zero")

        precision = cls._precision_from_step(step)
        step_dec = Decimal(str(step))
        one = Decimal("1")
        tol = Decimal("0.0000005")
        values: List[float] = []
        current = Decimal("0")
        guard = 0
        while current <= one + tol:
            numeric = cls._clamp(float(current))
            values.append(round(numeric, precision))
            current += step_dec
            guard += 1
            if guard > 10000:
                raise ValueError("step value results in too many grid points")

        if values[-1] < 1.0 - float(tol):
            values.append(1.0)

        unique_sorted = sorted(dict.fromkeys(values))
        return unique_sorted, precision

    @staticmethod
    def _index_from_coords(a_idx: int, b_idx: int, c_idx: int, base: int) -> int:
        return (a_idx * base * base) + (b_idx * base) + c_idx

    @staticmethod
    def _coords_from_index(index: int, base: int) -> Tuple[int, int, int]:
        a_idx = index // (base * base)
        remainder = index % (base * base)
        b_idx = remainder // base
        c_idx = remainder % base
        return a_idx, b_idx, c_idx

    @classmethod
    def _snap_to_grid(cls, value: float, grid: Sequence[float]) -> Tuple[float, int]:
        clamped = cls._clamp(value)
        closest = min(grid, key=lambda candidate: abs(candidate - clamped))
        return closest, grid.index(closest)

    @staticmethod
    def _apply_linear_offset(index: int, skip: int, steps: int, total: int, direction: str) -> int:
        if total == 0:
            return 0
        delta = (skip * steps) % total
        if direction == "descending":
            return (index - delta) % total
        return (index + delta) % total

    @staticmethod
    def _normalize(values: Tuple[float, float, float], precision: int) -> Tuple[float, float, float]:
        total = sum(values)
        if total <= 0:
            return values
        return tuple(round(v / total, precision) for v in values)

    @staticmethod
    def _within_sum_range(values: Tuple[float, float, float], min_sum: float, max_sum: float) -> bool:
        total = sum(values)
        return min_sum - 1e-8 <= total <= max_sum + 1e-8

    def _should_reset_state(self, signature: Tuple, seed: int) -> bool:
        if not self._state:
            return True
        state = self._state
        if state["signature"] != signature:
            return True
        previous_input = state.get("seed_input")
        previous_output = state.get("seed_output")
        if previous_input is not None and seed == previous_input:
            return False
        if previous_output is not None and seed == previous_output:
            return False
        return True

    def _initialise_state(
        self,
        signature: Tuple,
        seed: int,
        base_index: int,
        total: int,
        direction: str,
        skip: int,
        grid: Sequence[float],
        precision: int,
    ) -> None:
        self._state = {
            "signature": signature,
            "index": base_index,
            "total": total,
            "direction": direction,
            "skip": skip,
            "grid": list(grid),
            "precision": precision,
            "exposed_seed": seed,
            "rng": random.Random(seed) if direction == "shuffle" else None,
            "seed_input": seed,
            "seed_output": seed,
        }

    def _advance_index(self, current_index: int, total: int) -> int:
        direction = self._state["direction"]
        skip = self._state["skip"]
        if direction == "ascending":
            return (current_index + skip) % total
        if direction == "descending":
            return (current_index - skip) % total
        return current_index

    def _next_combo_impl(
        self,
        start_values: Tuple[float, float, float],
        step: float,
        seed: int,
        direction: str,
        skip: int,
        batch_size: int,
        normalize: bool,
        min_sum: float,
        max_sum: float,
    ) -> Tuple[float, float, float, int, int, int, bool, str]:
        if min_sum > max_sum:
            raise ValueError("min_sum must be less than or equal to max_sum")

        grid, precision = self._grid_from_step(step)
        base = len(grid)
        total = base ** 3
        if total <= 0:
            raise ValueError("no combinations available for the chosen step")

        snapped_values: List[float] = []
        snapped_indices: List[int] = []
        for value in start_values:
            snapped, idx = self._snap_to_grid(value, grid)
            snapped_values.append(snapped)
            snapped_indices.append(idx)
        snapped_tuple = tuple(snapped_values)
        start_index = self._index_from_coords(*snapped_indices, base)

        signature = (
            tuple(grid),
            tuple(snapped_indices),
            direction,
            int(skip),
            bool(normalize),
            round(min_sum, 6),
            round(max_sum, 6),
        )

        if self._should_reset_state(signature, seed):
            current_index = start_index
            if direction != "shuffle":
                current_index = self._apply_linear_offset(start_index, skip, seed, total, direction)
            self._initialise_state(signature, seed, current_index, total, direction, skip, grid, precision)
        elif direction != "shuffle":
            self._state["index"] %= total
            self._state["total"] = total
            self._state["grid"] = list(grid)

        state = self._state
        state["total"] = total
        state["grid"] = list(grid)
        state["precision"] = precision
        state["seed_input"] = seed

        combos: List[Tuple[float, float, float]] = []
        combo_indices: List[int] = []
        looped = False

        attempts = 0
        max_attempts = total if direction != "shuffle" else max(total, batch_size * 4)

        while len(combos) < batch_size and attempts < max_attempts:
            if direction == "shuffle":
                rng = state.get("rng")
                if rng is None:
                    rng = random.Random(state.get("seed_input", seed))
                    state["rng"] = rng
                idx = rng.randrange(total) if total > 0 else 0
            else:
                idx = state["index"]

            a_idx, b_idx, c_idx = self._coords_from_index(idx, base)
            triple = (grid[a_idx], grid[b_idx], grid[c_idx])

            if self._within_sum_range(triple, min_sum, max_sum):
                processed = self._normalize(triple, precision) if normalize else triple
                combos.append(processed)
                combo_indices.append(idx)

            attempts += 1

            if direction != "shuffle":
                next_index = self._advance_index(idx, total)
                if attempts >= total and len(combos) < batch_size:
                    looped = True
                    break
                state["index"] = next_index

        if len(combos) < batch_size and attempts >= max_attempts:
            looped = True

        if combos:
            primary = combos[0]
            first_index = combo_indices[0]
        else:
            primary = snapped_tuple
            first_index = start_index

        precision_fmt = f"{{:.{precision}f}}"
        combos_str = (
            ", ".join(
                f"({precision_fmt.format(a)}, {precision_fmt.format(b)}, {precision_fmt.format(c)})"
                for a, b, c in combos
            )
            if combos
            else "No combinations matched the filters"
        )

        increment = attempts if direction != "shuffle" else len(combos)
        next_seed = state["exposed_seed"] + increment
        state["exposed_seed"] = next_seed
        state["seed_output"] = next_seed
        state["index"] %= total

        return (
            primary[0],
            primary[1],
            primary[2],
            next_seed,
            first_index,
            total,
            looped,
            combos_str,
        )

    # endregion -----------------------------------------------------------------


NODE_CLASS_MAPPINGS = {
    "ModelMergeCombos": ModelMergeCombos
}