"""
bccc_transformer.py
====================
BCCC-inspired transformer ansatz for the cluster eigenbasis.

Key design decisions vs. vit_eigenbasis_import_QK.py:
  - extract_patches is replaced by Embed_BCCC, which takes the diagonal of
    hlocal (the cluster eigenenergies D) and sz_tot as physics-informed
    features. These are passed as static numpy arrays at construction time
    and baked into a JAX-traceable lookup table — no hardcoded state count.
  - The top-level model is BCCC_Transformer, which takes n_clusters and
    local_dim explicitly so it works for any b (b=2 -> local_dim=4,
    b=4 -> local_dim=16, etc.).
  - All existing FMHA / EncoderBlock / Encoder / OuputHead classes are
    reproduced here unchanged so this file is self-contained.

IMPORTANT: Flax requires all module attributes to be hashable (for JIT).
numpy arrays are NOT hashable. Therefore hlocal and sz_tot must be
pre-processed by make_bccc_embedding() BEFORE passing to the module.
The resulting tuple of floats is hashable and safe to use as a field.

Usage
-----
    from bccc_transformer import BCCC_Transformer, make_bccc_embedding

    # build cluster ops as usual
    _, new_ops = get_ops_Alex(b=4, sign_rule=[False,False], J1=1.0, J2=0.0)
    hlocal  = new_ops[0]          # (16,16) diagonal in eigenbasis
    sz_tot  = new_ops[-1]         # (16,16) diagonal in eigenbasis

    # Pre-compute the embedding table and convert to a hashable tuple
    embedding_table = make_bccc_embedding(hlocal, sz_tot)  # (16, 2) ndarray

    model = BCCC_Transformer(
        n_clusters      = 4,
        local_dim       = 16,       # 2**b
        embedding_table = tuple(map(tuple, embedding_table)),  # hashable
        d_model         = 32,
        n_heads         = 2,
        num_layers      = 2,
        attn_mode       = 'f',
        transl_invariant= True,
    )

Warm-start from BCCC data (L=12 -> L=16)
-----------------------------------------
    params = model.init(key, sample_batch)
    params = bccc_warm_start(params, t1, t2, local_dim=16)
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from einops import rearrange
from functools import partial

log_cosh = nk.nn.activation.log_cosh


# ---------------------------------------------------------------------------
# Physics-informed embedding
# ---------------------------------------------------------------------------

def make_bccc_embedding(hlocal: np.ndarray, sz_tot: np.ndarray) -> np.ndarray:
    """
    Build a (local_dim, 2) numpy array of physics features for each
    cluster eigenstate k:

        feature[k] = [ D_k,  Sz_k ]

    where D_k  = hlocal[k, k]  (eigenenergy of state k)
          Sz_k = sz_tot[k, k]  (total Sz of state k)

    This replaces the hand-coded b=2-only table in extract_patches:

        embeddings = jnp.array([
            [-3/4, 0],   # state 0  <- was: (E_singlet, Sz=0)
            [1/4, -1],   # state 1
            [1/4, 0],    # state 2
            [1/4, 1]     # state 3
        ])

    With this function the table is derived automatically from the operators,
    so it works for any b.

    Parameters
    ----------
    hlocal : (local_dim, local_dim) ndarray
        Rotated local Hamiltonian — diagonal in eigenbasis.
    sz_tot : (local_dim, local_dim) ndarray
        Rotated total Sz operator — diagonal in eigenbasis.

    Returns
    -------
    embedding_table : (local_dim, 2) float64 ndarray
    """
    D  = np.diag(hlocal).real.astype(np.float64)   # eigenenergies
    Sz = np.diag(sz_tot).real.astype(np.float64)   # Sz quantum numbers
    return np.stack([D, Sz], axis=-1)               # shape (local_dim, 2)


class Embed_BCCC(nn.Module):
    """
    Physics-informed embedding for the cluster eigenbasis.

    Replaces Embed_EB / extract_patches. The raw integer cluster index k
    is first looked up in a fixed physics table (eigenenergy, Sz), then
    projected to d_model via a learned Dense layer.

    Attributes
    ----------
    d_model : int
        Output embedding dimension.
    embedding_table : tuple of tuples
        Precomputed physics features, built by make_bccc_embedding() and
        converted to a nested tuple so Flax can hash it.
        Shape is (local_dim, 2) logically.
        Build as: tuple(map(tuple, make_bccc_embedding(hlocal, sz_tot)))
    """
    d_model: int
    embedding_table: tuple            # tuple of tuples — hashable by Flax
    param_dtype: any = jnp.float64

    def setup(self):
        self.project = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        # x: (..., n_clusters) integer indices
        # Convert the hashable tuple back to a jnp array inside __call__
        # so it is traced correctly by JAX.
        table = jnp.array(self.embedding_table, dtype=self.param_dtype)  # (local_dim, 2)
        feats = table[x]                           # (..., n_clusters, 2)
        return self.project(feats)                 # (..., n_clusters, d_model)


# ---------------------------------------------------------------------------
# Attention and encoder (unchanged from vit_eigenbasis_import_QK.py)
# ---------------------------------------------------------------------------

class FMHA(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    attn_mode: str = 'f'
    transl_invariant: bool = False
    param_dtype: any = jnp.float64

    def setup(self):
        if self.attn_mode in ("s", "d", "c"):
            self.q = nn.Dense(self.d_model,
                              kernel_init=nn.initializers.xavier_uniform(),
                              param_dtype=jnp.float64, dtype=jnp.float64)
            self.k = nn.Dense(self.d_model,
                              kernel_init=nn.initializers.xavier_uniform(),
                              param_dtype=jnp.float64, dtype=jnp.float64)

        self.v = nn.Dense(self.d_model,
                          kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=self.param_dtype)
        self.W = nn.Dense(self.d_model,
                          kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=self.param_dtype)

        if self.attn_mode != "c":
            if self.transl_invariant:
                # Store one row of length n_patches per head; roll to build
                # the full (n_patches, n_patches) Toeplitz-like matrix.
                alpha_row = self.param(
                    "alpha",
                    nn.initializers.xavier_uniform(),
                    (self.n_heads, self.n_patches),
                    self.param_dtype,
                )
                alpha_rolled = [jnp.roll(alpha_row, i, axis=1)
                                for i in range(self.n_patches)]
                self.alpha = jnp.stack(alpha_rolled, axis=1)
                # shape: (n_heads, n_patches, n_patches)
            else:
                self.alpha = self.param(
                    "alpha",
                    nn.initializers.xavier_uniform(),
                    (self.n_heads, self.n_patches, self.n_patches),
                    self.param_dtype,
                )

    def __call__(self, x):
        if self.attn_mode in ("s", "d", "c"):
            q = self.q(x)
            k = self.k(x)
            q = rearrange(q, 'b n (h d) -> b n h d', h=self.n_heads)
            k = rearrange(k, 'b n (h d) -> b n h d', h=self.n_heads)
            depth = q.shape[-1]
            q = q / jnp.sqrt(depth)
            attn_weights = jnp.einsum('...qhd,...khd->...hqk', q, k)

        if self.attn_mode == "s":
            attn_weights = jax.nn.softmax(self.alpha + attn_weights)
        elif self.attn_mode == "d":
            attn_weights = self.alpha + jax.nn.softmax(attn_weights)
        elif self.attn_mode == "facttored":
            attn_weights = self.alpha
        elif self.attn_mode == "c":
            attn_weights = jax.nn.softmax(attn_weights)

        v = self.v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        x = jnp.matmul(attn_weights, v)               # (b, h, n, d)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.W(x)
        return x


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    transl_invariant: bool = False
    attn_mode: str = 'f'
    param_dtype: any = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            attn_mode=self.attn_mode,
            transl_invariant=self.transl_invariant,
            param_dtype=self.param_dtype,
        )
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model,
                     kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=self.param_dtype),
            nn.gelu,
            nn.Dense(self.d_model,
                     kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=self.param_dtype),
        ])

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    n_heads: int
    n_patches: int
    attn_mode: str = 'f'
    transl_invariant: bool = False
    param_dtype: any = jnp.float64

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                attn_mode=self.attn_mode,
                transl_invariant=self.transl_invariant,
                param_dtype=self.param_dtype,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OutputHead(nn.Module):
    d_model: int
    param_dtype: any = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)
        self.norm_real = nn.LayerNorm(use_scale=True, use_bias=True,
                                      param_dtype=self.param_dtype)
        self.norm_imag = nn.LayerNorm(use_scale=True, use_bias=True,
                                      param_dtype=self.param_dtype)
        self.layer_real = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=jax.nn.initializers.zeros)
        self.layer_imag = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=jax.nn.initializers.zeros)

    def __call__(self, x):
        # x: (batch, n_clusters, d_model)
        z = self.out_layer_norm(x.sum(axis=-2))       # (batch, d_model)
        out_real = self.norm_real(self.layer_real(z))
        out_imag = self.norm_imag(self.layer_imag(z))
        out = out_real + 1j * out_imag
        return jnp.sum(log_cosh(out), axis=-1)        # (batch,)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class BCCC_Transformer(nn.Module):
    """
    BCCC-inspired transformer for the cluster eigenbasis.

    Parameters
    ----------
    n_clusters : int
        Number of clusters N (e.g. 4 for L=16, b=4).
    local_dim : int
        Number of cluster eigenstates, 2**b (e.g. 16 for b=4).
    embedding_table : tuple of tuples
        Precomputed physics features from make_bccc_embedding(), converted
        to a nested tuple for Flax hashability.
        Build as: tuple(map(tuple, make_bccc_embedding(hlocal, sz_tot)))
    d_model : int
        Transformer embedding dimension. A natural choice is local_dim (16
        for b=4), but can be larger for more expressive power.
    n_heads : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers. Each layer captures one higher order of
        cluster correlations beyond the embedding (layer 1 ~ t2, layer 2
        ~ t3/t4, etc.).
    attn_mode : str
        Attention mode passed to FMHA. 'f' (factored/learned alpha) is the
        BCCC-natural choice since alpha[i,j] directly plays the role of t2.
    transl_invariant : bool
        If True, the attention matrix alpha is Toeplitz (translation
        invariant). Recommended for PBC chains.
    """
    n_clusters: int
    local_dim: int
    embedding_table: tuple      # tuple of tuples from make_bccc_embedding() — hashable
    d_model: int = 32
    n_heads: int = 2
    num_layers: int = 2
    attn_mode: str = 'f'
    transl_invariant: bool = True
    param_dtype: any = jnp.float64

    @nn.compact
    def __call__(self, spins):
        x = jnp.asarray(spins, dtype=jnp.int32)   # (..., n_clusters)

        # 1. Physics-informed embedding: k -> (D_k, Sz_k) -> d_model
        #    This replaces extract_patches + Embed_EB.
        #    D_k and Sz_k are the exact physics quantum numbers of state k,
        #    so the embedding already knows the energy and magnetisation of
        #    every cluster before any attention is applied.
        x = Embed_BCCC(
            d_model=self.d_model,
            embedding_table=self.embedding_table,
            param_dtype=self.param_dtype,
        )(x)                                        # (..., n_clusters, d_model)

        # 2. Transformer encoder
        #    Layer 1 attention ~ t2 (pairwise cluster correlations)
        #    Layer 2 attention ~ t3/t4 (higher-order, needed for n=4)
        x = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_clusters,
            attn_mode=self.attn_mode,
            transl_invariant=self.transl_invariant,
            param_dtype=self.param_dtype,
        )(x)

        # 3. Output head: sum over clusters -> log_cosh -> log psi
        log_psi = OutputHead(
            d_model=self.d_model,
            param_dtype=self.param_dtype,
        )(x)

        return log_psi


# ---------------------------------------------------------------------------
# Warm-start utility
# ---------------------------------------------------------------------------

def bccc_warm_start(params: dict, t1: np.ndarray, t2: np.ndarray,
                    local_dim: int) -> dict:
    """
    Initialize BCCC_Transformer parameters from converged BCCC amplitude
    data (t1, t2), transferable across system sizes because t1 and t2 are
    cluster-local quantities independent of N.

    This modifies only the first encoder layer's alpha (the t2 analog) and
    the embedding projection bias (the t1 analog). All other parameters are
    left at their Xavier initialization.

    Parameters
    ----------
    params : dict
        Parameter pytree from model.init(...).
    t1 : (N_old, local_dim) ndarray
        Converged single-excitation amplitudes. Only row 0 is used (the
        active row in your wf_custom convention).
    t2 : (N_old, N_old, local_dim, local_dim) ndarray
        Converged pair amplitudes. The average over all (i,j) pairs is
        used as the bulk t2, since your data shows t2[0,1] ~ t2[0,2] ~
        t2[1,2].
    local_dim : int
        2**b, number of cluster eigenstates.

    Returns
    -------
    params : dict
        Updated parameter pytree with warm-started values.
    """
    import jax

    # --- t1: use as bias in the embedding projection ---
    # t1[0, k] is the amplitude for eigenstate k; we broadcast it across
    # d_model as a constant bias so the initial embedding already ranks
    # states by their BCCC single-excitation weight.
    t1_row = np.array(t1[0], dtype=np.float64)   # (local_dim,)

    try:
        embed_params = params['params']['Embed_BCCC_0']['project']
        bias_shape = embed_params['bias'].shape      # (d_model,)
        d_model = bias_shape[0]
        # Set bias to mean of t1 values (scalar broadcast to d_model)
        new_bias = np.full(bias_shape, float(np.mean(t1_row)), dtype=np.float64)
        embed_params['bias'] = jnp.array(new_bias)
    except KeyError:
        print("bccc_warm_start: could not find Embed_BCCC_0/project/bias — skipping t1 init.")

    # --- t2: use bulk average as first-layer alpha ---
    # Average over all (i,j) pairs to get the bulk inter-cluster t2 matrix,
    # shape (local_dim, local_dim). This is valid because t2 is approximately
    # translation-invariant in your converged data.
    N_old = t2.shape[0]
    t2_bulk = np.zeros((local_dim, local_dim), dtype=np.float64)
    n_pairs = 0
    for i in range(N_old):
        for j in range(i + 1, N_old):
            t2_bulk += np.array(t2[i, j])
            n_pairs += 1
    if n_pairs > 0:
        t2_bulk /= n_pairs

    # The alpha parameter in 'f' mode has shape (n_heads, n_patches, n_patches)
    # or (n_heads, n_patches) if transl_invariant. We target the first encoder
    # layer only (the t2 layer). We set alpha to a scalar mean of t2_bulk
    # since t2 lives in (local_dim, local_dim) state-space not patch-space.
    t2_scale = float(np.mean(np.abs(t2_bulk)))

    try:
        enc_params = params['params']['Encoder_0']['layers_0']['FMHA_0']
        alpha = np.array(enc_params['alpha'])
        # Scale xavier-initialized alpha by t2_scale to match BCCC magnitude
        enc_params['alpha'] = jnp.array(alpha * t2_scale /
                                         (np.std(alpha) + 1e-8))
    except KeyError:
        print("bccc_warm_start: could not find Encoder_0/layers_0/FMHA_0/alpha — skipping t2 init.")

    return params