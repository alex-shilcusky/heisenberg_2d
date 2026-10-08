"""
multilayer_2D_vit_Eigenbasis.py
===============================
2D ViT wave function for the CLUSTER EIGENBASIS (2x2 plaquette clusters, b=4).

Drop-in for helpers/. Self-contained: depends only on jax / flax / netket /
einops (no other helpers modules).

Token model
-----------
A token is one 2x2 plaquette, represented by which of its `local_dim` (= 2**b
= 16 for b=4) cluster eigenstates it occupies -- an integer in [0, local_dim).
The input `spins` is therefore already the length-N_clusters sequence of these
integers (NO patch extraction).

Two ways to turn that integer into a d_model vector, selected by `embedding`:

  'learnable' (default)
      A free nn.Embed lookup table, local_dim x d_model trainable parameters.
      The 16 local states are unrelated as far as the network is concerned; any
      physical similarity between them has to be learned from scratch.

  'qn'
      The token is a learnable LINEAR FUNCTION of the state's physical quantum
      numbers (E, Sz, S~, sigma_d, C2), supplied as `qn_table`:

          table = Dense(d_model)(standardise(qn_table))   # (local_dim, d_model)
          x     = table[spins]

      so 5*d_model + d_model parameters instead of local_dim*d_model, and two
      states with similar quantum numbers necessarily start with similar
      tokens. Get the table from
      run_2D_eigenbasis.get_ops_2x2(..., return_qn=True). Columns are
      standardised to zero mean / unit variance across the local states inside
      the model, so raw physical units can be passed in; a constant column is
      left at zero rather than divided by a zero standard deviation.

Translation invariance
-----------------------
`transl_invariant=True` builds a 2D-circulant attention matrix (roll2d) over
the SQUARE Cx x Cy cluster grid, i.e. invariance under translations that
preserve the 2x2 tiling (multiples of 2 physical sites). This assumes the
N_clusters tokens are ordered row-major as n = cx + Cx*cy (Cx == Cy), which is
the ordering produced by get_cluster_graph_2D_J2 -- so the network's symmetry
and the Hamiltonian's bonds refer to the same cluster indexing. For an LxL
lattice with 2x2 clusters, N_clusters = (L/2)^2 is a perfect square.

Requires: d_model % n_heads == 0.

python 3.13 / netket 3.16.1.post1
"""

import netket as nk
import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange
from functools import partial


# ---------------------------------------------------------------------------
# 2D translation over the (square) cluster grid
# ---------------------------------------------------------------------------
@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(vecs, i, j):
    """Roll a (n_heads, n_patches) array as a 2D field on a side x side grid.
    For a non-square Cx x Cy grid, reshape to (Cy, Cx) instead and roll
    arange(Cy) on axis -2 and arange(Cx) on axis -1."""
    side = int(round(vecs.shape[-1] ** 0.5))
    v = vecs.reshape(vecs.shape[0], side, side)
    v = jnp.roll(jnp.roll(v, i, axis=-2), j, axis=-1)
    return v.reshape(vecs.shape[0], -1)


# ---------------------------------------------------------------------------
# Factored multi-head attention (2D-translation-invariant alpha via roll2d)
# ---------------------------------------------------------------------------
class FMHA(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    attn_mode: str = 'f'             # 'f' factored, 's' T5, 'd' decoupled, 'c' plain
    transl_invariant: bool = True
    param_dtype = jnp.float64

    def setup(self):
        if self.attn_mode in ("s", "d", "c"):
            self.q = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                              param_dtype=jnp.float64, dtype=jnp.float64)
            self.k = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                              param_dtype=jnp.float64, dtype=jnp.float64)

        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=self.param_dtype)
        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                          param_dtype=self.param_dtype)

        if self.attn_mode != "c":
            if self.transl_invariant:
                self.alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                                        (self.n_heads, self.n_patches), self.param_dtype)
                side = int(round(self.n_patches ** 0.5))
                assert side * side == self.n_patches, (
                    f"roll2d needs a square cluster grid; got n_patches="
                    f"{self.n_patches} (use the non-square roll noted in roll2d).")
                self.alpha = roll2d(self.alpha, jnp.arange(side), jnp.arange(side))
                self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
            else:
                self.alpha = self.param("alpha", nn.initializers.xavier_uniform(),
                                        (self.n_heads, self.n_patches, self.n_patches),
                                        self.param_dtype)

    def __call__(self, x):
        if self.attn_mode in ("s", "d", "c"):
            q = self.q(x); k = self.k(x)
            q = rearrange(q, 'b n (h d) -> b n h d', h=self.n_heads)
            k = rearrange(k, 'b n (h d) -> b n h d', h=self.n_heads)
            q = q / jnp.sqrt(q.shape[-1])
            attn_weights = jnp.einsum('...qhd,...khd->...hqk', q, k)

        if self.attn_mode == "s":
            attn_weights = jax.nn.softmax(self.alpha + attn_weights)
        elif self.attn_mode == "d":
            attn_weights = self.alpha + jax.nn.softmax(attn_weights)
        elif self.attn_mode in ("f", "factored"):      # 'f' is the declared default
            attn_weights = self.alpha
        elif self.attn_mode == "c":
            attn_weights = jax.nn.softmax(attn_weights)

        v = self.v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        x = jnp.matmul(attn_weights, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.W(x)
        return x


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_patches: int
    transl_invariant: bool = True
    attn_mode: str = 'f'
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(d_model=self.d_model, n_heads=self.n_heads,
                         n_patches=self.n_patches, attn_mode=self.attn_mode,
                         transl_invariant=self.transl_invariant)
        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.ff = nn.Sequential([
            nn.Dense(4 * self.d_model, kernel_init=nn.initializers.xavier_uniform(),
                     param_dtype=self.param_dtype),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(),
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
    transl_invariant: bool = True

    def setup(self):
        self.layers = [EncoderBlock(d_model=self.d_model, n_heads=self.n_heads,
                                    n_patches=self.n_patches, attn_mode=self.attn_mode,
                                    transl_invariant=self.transl_invariant)
                       for _ in range(self.num_layers)]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


log_cosh = nk.nn.activation.log_cosh


class OutputHead(nn.Module):
    d_model: int
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)
        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
        self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
        self.output_layer0 = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                                      kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=jax.nn.initializers.zeros)
        self.output_layer1 = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                                      kernel_init=nn.initializers.xavier_uniform(),
                                      bias_init=jax.nn.initializers.zeros)

    def __call__(self, x):
        z = self.out_layer_norm(x.sum(axis=1))          # sum over cluster tokens
        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))
        out = out_real + 1j * out_imag
        return jnp.sum(log_cosh(out), axis=-1)


class AmpPhaseHead(nn.Module):
    """log psi = a(z) + i * phi(z), with a separate MLP for the log-amplitude a
    and the phase phi of the pooled token z.

    The log-cosh head above makes amplitude and phase share one set of
    log-cosh units. In the plaquette eigenbasis the ground state's SIGN is
    what the network fails to learn (sign_diagnostics_4x4.py: given the exact
    signs, the same network fits as well as the spin-basis ViT), so this gives
    the phase capacity of its own. Opt-in via ViT_2D_EB(output_head='amp_phase').
    """
    d_model: int
    param_dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        z = nn.LayerNorm(param_dtype=self.param_dtype, name="pool_norm")(x.sum(axis=1))

        def mlp(name):
            h = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                         kernel_init=nn.initializers.xavier_uniform(), name=f"{name}_hidden")(z)
            h = nn.gelu(h)
            return nn.Dense(1, param_dtype=self.param_dtype,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=jax.nn.initializers.zeros, name=f"{name}_out")(h)[..., 0]

        return mlp("amp") + 1j * mlp("phase")


# ---------------------------------------------------------------------------
# Top-level 2D eigenbasis ViT wave function (learnable embedding)
# ---------------------------------------------------------------------------
class ViT_2D_EB(nn.Module):
    """
    Parameters
    ----------
    num_layers, d_model, n_heads : usual transformer sizes (d_model % n_heads == 0)
    local_dim  : number of cluster eigenstates = 2**b (16 for a 2x2 plaquette)
    attn_mode  : 'f' (factored, recommended), 's', 'd', or 'c'
    transl_invariant : 2D-circulant attention over the cluster grid (needs a
                       square grid; see module docstring)
    embedding  : 'learnable' -> free nn.Embed table (local_dim x d_model params)
                 'qn'        -> learnable linear map of the physical quantum
                                numbers in `qn_table` (5*d_model + d_model
                                params). See the module docstring.
    qn_table   : required when embedding='qn'. The per-local-state quantum
                 numbers as a HASHABLE nested tuple of shape
                 (local_dim, n_features), e.g. the (16, 5) table
                 (E, Sz, S~, sigma_d, C2) returned by
                 get_ops_2x2(..., return_qn=True). Static, not a parameter.
    """
    num_layers: int
    d_model: int
    n_heads: int
    local_dim: int = 16
    attn_mode: str = 'f'
    transl_invariant: bool = True
    embedding: str = 'learnable'
    qn_table: tuple = ()
    output_head: str = 'logcosh'     # 'logcosh' (original) | 'amp_phase' (see AmpPhaseHead)
    param_dtype = jnp.float64

    @nn.compact
    def __call__(self, spins):
        x = jnp.asarray(spins, dtype=jnp.int32)          # (batch, N_clusters)
        n_patches = x.shape[-1]                          # one token per cluster

        if self.embedding == 'learnable':
            # free lookup table: the 16 local states are a priori unrelated
            x = nn.Embed(num_embeddings=self.local_dim, features=self.d_model,
                         param_dtype=self.param_dtype)(x)  # (batch, N_clusters, d_model)
        elif self.embedding == 'qn':
            if not self.qn_table:
                raise ValueError("embedding='qn' needs qn_table=... ; get it from "
                                 "get_ops_2x2(sign_rule, J1, J2, return_qn=True).")
            qn = jnp.asarray(self.qn_table, dtype=self.param_dtype)
            if qn.shape[0] != self.local_dim:
                raise ValueError(f"qn_table has {qn.shape[0]} rows but local_dim="
                                 f"{self.local_dim}.")
            # standardise per feature; a constant column stays at 0 rather than
            # blowing up on a zero standard deviation
            mu = qn.mean(axis=0)
            sd = qn.std(axis=0)
            qn = jnp.where(sd > 1e-12, (qn - mu) / jnp.where(sd > 1e-12, sd, 1.0), 0.0)
            # learnable LINEAR map (n_features -> d_model), shared by all states
            table = nn.Dense(self.d_model, param_dtype=self.param_dtype,
                             kernel_init=nn.initializers.xavier_uniform(),
                             name='qn_embed')(qn)          # (local_dim, d_model)
            x = table[x]                                   # (batch, N_clusters, d_model)
        else:
            raise ValueError(f"embedding must be 'learnable' or 'qn', got "
                             f"{self.embedding!r}")

        y = Encoder(num_layers=self.num_layers, d_model=self.d_model,
                    n_heads=self.n_heads, n_patches=n_patches,
                    attn_mode=self.attn_mode,
                    transl_invariant=self.transl_invariant)(x)
        if self.output_head == 'logcosh':
            return OutputHead(d_model=self.d_model)(y)
        if self.output_head == 'amp_phase':
            return AmpPhaseHead(d_model=self.d_model)(y)
        raise ValueError(f"output_head must be 'logcosh' or 'amp_phase', got "
                         f"{self.output_head!r}")
