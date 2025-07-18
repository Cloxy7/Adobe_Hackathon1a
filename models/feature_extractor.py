import torch

def build_feature_vector(line):
    return torch.tensor([
        1.0 if line["is_bold"] else 0.0,
        1.0 if line["is_all_caps"] else 0.0,
        float(line["font_size"]) / 20.0,
        float(line["line_length"]) / 10.0,
        float(line["line_gap"]) / 20.0,
        1.0 if line["ends_with_dot"] else 0.0,
        float(line["x_position"]) / 600.0
    ])
