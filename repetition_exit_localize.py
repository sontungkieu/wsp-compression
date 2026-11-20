"""Causal last-position prefill patching for the Qwen reasoning-stop signal."""
from contextlib import contextmanager


def decoder_layers(model, torch):
    containers = [list(layerlist) for name, layerlist in model.named_modules()
                  if name.endswith(".layers") and isinstance(layerlist, torch.nn.ModuleList)
                  and len(layerlist) and type(layerlist[0]).__name__ == "Qwen3_5DecoderLayer"]
    if len(containers) != 1:
        raise ValueError("could not resolve a unique Qwen3.5 text decoder")
    return containers[0]


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


def odds(logits, token, prior, torch):
    scores = logits.float().clone()
    for i, item in enumerate(prior):
        index = torch.tensor(sorted(set(item)), dtype=torch.long, device=scores.device)
        scores[i, index] -= 1.5
    scores /= .6
    z = scores[:, token].clone()
    scores[:, token] = -torch.inf
    return z - scores.logsumexp(dim=-1)


def capture(model, model_inputs, layers):
    captures = {}
    hooks = []

    def hook(layer):
        def store(_module, _args, output):
            captures[layer] = hidden(output)[:, -1].detach().clone()
        return store

    try:
        for index, module in layers.items():
            hooks.append(module.register_forward_hook(hook(index)))
        logits = model(**model_inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
    finally:
        for registration in hooks:
            registration.remove()
    if captures.keys() != layers.keys():
        raise ValueError("prefill capture missed one or more layers")
    return logits, captures


@contextmanager
def replace_first_position(module, replacement):
    applied = [0]

    def hook(_module, _args, output):
        if applied[0]:
            return output
        tensor = hidden(output)
        if tensor.shape[0] != replacement.shape[0] or tensor.shape[-1] != replacement.shape[-1]:
            raise ValueError("replacement hidden state has the wrong shape")
        updated = tensor.clone()
        updated[:, -1] = replacement
        applied[0] = 1
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    registration = module.register_forward_hook(hook)
    try:
        yield
    finally:
        registration.remove()
    if not applied[0]:
        raise ValueError("prefill hook was never reached")


def scan(model, inputs, layers, token, prior, torch):
    baseline = {}
    activations = {}
    for arm, arm_inputs in inputs.items():
        logits, activations[arm] = capture(model, arm_inputs, layers)
        baseline[arm] = odds(logits, token, prior, torch)
    deltas = {}
    for layer, module in layers.items():
        tests = {}
        for name, receiver, donor in (("forward", "blank", "left"), ("reverse", "left", "blank"),
                                      ("round", "blank", "round")):
            with replace_first_position(module, activations[donor][layer]):
                result = model(**inputs[receiver], use_cache=False, logits_to_keep=1).logits[:, -1]
            tests[name] = odds(result, token, prior, torch) - baseline[receiver]
        deltas[layer] = tests
    return baseline, deltas


def random_replacement(original, donor, seed, torch):
    generator = torch.Generator(device=original.device).manual_seed(seed)
    delta = donor.float() - original.float()
    vector = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
    vector *= delta.norm(dim=-1, keepdim=True) / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (original.float() + vector).to(original.dtype)
