import itertools

class ModelMergeCombos:
    def __init__(self):
        # internal counter
        self.internal_seed = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "start_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "start_c": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                # expose seed so you can see and adjust it
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("a", "b", "c", "seed")
    FUNCTION = "next_combo"
    CATEGORY = "utils"

    def next_combo(self, start_a, start_b, start_c, step, seed):
        # if UI seed changed manually, reset internal counter
        if seed != self.internal_seed:
            self.internal_seed = seed

        n = int(round(1.0 / step)) + 1
        values = [round(i * step, 6) for i in range(n)]
        combos = list(itertools.product(values, repeat=3))

        try:
            start_index = combos.index((round(start_a, 6), round(start_b, 6), round(start_c, 6)))
        except ValueError:
            start_index = 0

        idx = (start_index + self.internal_seed) % len(combos)
        a, b, c = combos[idx]

        # increment seed after use
        self.internal_seed += 1

        return (a, b, c, self.internal_seed)

NODE_CLASS_MAPPINGS = {
    "ModelMergeCombos": ModelMergeCombos
}
