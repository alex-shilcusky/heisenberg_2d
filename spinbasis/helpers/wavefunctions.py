"""
Wavefunction factory: turns the [model] section of the config into a Flax module.

    [model]
    type = "jastrow" | "rbm" | "vit" | "gcnn"

and each type reads its own subtable, [model.jastrow], [model.rbm], [model.gcnn] or
[model.vit]. All three subtables live in the base config at once, so a sweep
over `model.type` works without editing anything else; the subtables of the
types that are not selected are simply ignored.

Adding a wavefunction = one more branch in build_model plus a [model.<name>]
table in config_spinbasis.toml. Everything downstream (sampler, driver, checkpointing)
only ever sees a Flax module mapping (batch, N) spins in {-1, +1} to log(psi).
"""
import jax.numpy as jnp
import netket as nk
from flax import linen as nn

from helpers.vit import ViT

MODEL_TYPES = ("jastrow", "rbm", "vit", "gcnn")

_DTYPES = {"float64": jnp.float64, "complex128": jnp.complex128}
_SYMMETRIES = ("none", "translation", "space_group")


def _dtype(name, table):
    if name not in _DTYPES:
        raise ValueError(f"[{table}] param_dtype must be one of {list(_DTYPES)}, "
                         f"got {name!r}")
    return _DTYPES[name]


def _symmetry_group(name, graph, table):
    if name not in _SYMMETRIES:
        raise ValueError(f"[{table}] symmetries must be one of {list(_SYMMETRIES)}, "
                         f"got {name!r}")
    if name == "translation":
        return graph.translation_group()
    if name == "space_group":
        return graph.space_group()          # translations x C4v
    return None


def _kernel_init(sub):
    """Optional `init_std` -> normal(init_std) initializer kwargs; absent -> {},
    i.e. NetKet's own default (normal(0.01) for RBM/Jastrow, 0.1 for RBMSymm)."""
    std = sub.get("init_std")
    if std is None:
        return {}, "default"
    return {"kernel_init": nn.initializers.normal(stddev=float(std))}, f"{float(std):g}"


def build_model(model_cfg, graph, L):
    """Return (flax module, one-line description for the log)."""
    kind = str(model_cfg["type"]).lower()
    if kind not in MODEL_TYPES:
        raise ValueError(f"[model] type must be one of {list(MODEL_TYPES)}, got {kind!r}")
    sub = model_cfg.get(kind, {})
    table = f"model.{kind}"

    if kind == "jastrow":
        # log psi = sum_{i<j} W_ij s_i s_j. No one-body term: in the Sz = 0
        # sector sum_i s_i is fixed, and a site-dependent field would only break
        # translation symmetry.
        dtype = _dtype(sub.get("param_dtype", "complex128"), table)
        init, std = _kernel_init(sub)
        model = nk.models.Jastrow(param_dtype=dtype, **init)
        return model, f"Jastrow(param_dtype={dtype.__name__}, init_std={std})"

    if kind == "rbm":
        dtype = _dtype(sub.get("param_dtype", "complex128"), table)
        alpha = sub.get("alpha", 1)
        use_vb = bool(sub.get("use_visible_bias", True))
        init, std = _kernel_init(sub)
        symm_name = sub.get("symmetries", "none")
        group = _symmetry_group(symm_name, graph, table)
        if group is None:
            model = nk.models.RBM(alpha=alpha, param_dtype=dtype,
                                  use_visible_bias=use_vb, **init)
            return model, (f"RBM(alpha={alpha}, param_dtype={dtype.__name__}, "
                           f"use_visible_bias={use_vb}, init_std={std})")
        # RBMSymm has alpha * N / |G| feature maps, each shared across the group.
        # NetKet raises if that is below 1, but only at init time and without
        # saying which config key to change, so check it here.
        n_feat = alpha * graph.n_nodes / len(group)
        if n_feat < 1:
            raise ValueError(
                f"[{table}] alpha={alpha} gives {n_feat:g} feature maps with "
                f"symmetries={symm_name!r} (|G|={len(group)}, N={graph.n_nodes}); "
                f"need alpha >= {len(group) / graph.n_nodes:g}.")
        # RBMSymm's visible bias is ONE scalar b multiplying sum_i s_i, which is
        # identically 0 at total_sz = 0. The parameter does not enter psi at all:
        # its log-derivative is exactly 0 on every sample, so it only adds a
        # null direction to S and makes the per-parameter SNR that auto_alpha
        # uses 0/0 = NaN. Dropping it leaves the wavefunction unchanged.
        model = nk.models.RBMSymm(symmetries=group, alpha=alpha, param_dtype=dtype,
                                  use_visible_bias=False, **init)
        return model, (f"RBMSymm(alpha={alpha} -> {n_feat:g} feature maps, "
                       f"symmetries={symm_name} |G|={len(group)}, "
                       f"param_dtype={dtype.__name__}, init_std={std}, visible bias "
                       f"{'dropped (identically 0 at Sz=0)' if use_vb else 'off'})")

    if kind == "gcnn":
        # Group-convolutional network (Roth & MacDonald 2021): exactly invariant
        # under the chosen group, single-site translations included. Default
        # characters (all +1) select the fully symmetric sector, which is where
        # the 4x4 ground state lies at J2 = 0 and 0.5 (checked by ED, both sign
        # rules). With param_dtype = "float64" log psi is real, so psi > 0: use
        # sign_rule = "TF" (Marshall), whose J2 = 0 ground state is single-signed.
        layers = int(sub["layers"])
        features = sub["features"]
        features = int(features) if isinstance(features, (int, float)) else tuple(int(f) for f in features)
        dtype = _dtype(sub.get("param_dtype", "float64"), table)
        symm_name = sub.get("symmetries", "space_group")
        if symm_name == "none":
            raise ValueError(f"[{table}] symmetries must be 'translation' or 'space_group'.")
        group = _symmetry_group(symm_name, graph, table)
        parity = int(sub.get("parity", 0))
        if parity not in (-1, 0, 1):
            raise ValueError(f"[{table}] parity must be +1, -1 or 0 (off), got {parity}.")
        # Passing the graph (not the group) lets NetKet use its FFT over translations.
        model = nk.models.GCNN(symmetries=graph if symm_name == "space_group" else group,
                               layers=layers, features=features, param_dtype=dtype,
                               parity=parity or None, complex_output=True)
        return model, (f"GCNN(layers={layers}, features={features}, symmetries={symm_name} "
                       f"|G|={len(group)}, parity={parity or 'off'}, param_dtype={dtype.__name__})")

    # vit
    num_layers = int(sub["num_layers"])
    d_model = int(sub["d_model"])
    n_heads = int(sub["n_heads"])
    patch_size = int(sub["patch_size"])
    transl_invariant = bool(sub.get("transl_invariant", True))
    if d_model % n_heads:
        raise ValueError(f"[{table}] d_model={d_model} must be divisible by "
                         f"n_heads={n_heads}.")
    if L % patch_size:
        raise ValueError(f"[{table}] L={L} must be a multiple of patch_size={patch_size}.")
    model = ViT(num_layers=num_layers, d_model=d_model, n_heads=n_heads,
                patch_size=patch_size, transl_invariant=transl_invariant)
    n_patches = (L // patch_size) ** 2
    return model, (f"ViT(num_layers={num_layers}, d_model={d_model}, n_heads={n_heads}, "
                   f"patch_size={patch_size} -> {n_patches} patches, "
                   f"transl_invariant={transl_invariant})")
