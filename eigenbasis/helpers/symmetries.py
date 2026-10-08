"""
symmetries.py
=============
Point-group (C4v) and spin-flip symmetry projection for the 2x2 cluster
eigenbasis, in the form of a Flax wrapper around any log-amplitude module.

Why a projection and not an architectural constraint
----------------------------------------------------
Translations BETWEEN clusters are already exact in the ansatz: FMHA builds a
2D-circulant attention matrix (see roll2d in multilayer_2D_vit_Eigenbasis),
so the state is invariant under the Cx x Cy cluster-translation group for
free. Everything else has to be imposed on the wave function itself:

    Psi~(sigma) = (1/|G|) sum_{g in G} Psi(g^-1 sigma)                    (A1)

which is Eq. (A1) of Viteritti et al. 2025. The sum is over a FIXED number of
group elements and does not grow with system size -- but it does cost |G|
forward passes per sample, so see "Cost" below.

Why this is cheap in the eigenbasis (and why that is not obvious)
------------------------------------------------------------------
A configuration here is a length-N_clusters vector of integers in [0, 16),
labelling which 2x2-plaquette EIGENSTATE each cluster occupies. A lattice
symmetry g acts in two places at once:

  1. it permutes the clusters on the Cx x Cy grid, and
  2. inside every cluster it permutes the four physical sites,

and (2) is the part that could have been expensive: a site permutation is a
16x16 matrix, and in the eigenbasis V it becomes V^T P V, which is in general
DENSE -- in which case g would map one basis configuration to a superposition
of many and no projection of the form (A1) would be possible at all.

Numerically, for the basis built by get_ops_2x2 (hierarchical simultaneous
diagonalisation against Sz_tot, S^2, C2, sigma_d), every element of C4v and
the global spin flip come out MONOMIAL (exactly one nonzero per column, of
modulus 1) for both sign_rule = "FF" and "TF" at J1=1, J2=0:

    sigma_d, C2   diagonal, eigenvalues +-1
    C4, C4^3      signed permutation
    spin flip     signed permutation

sigma_d and C2 are diagonal because the basis is adapted to them by
construction. C4 is NOT diagonal -- it does not commute with sigma_d -- but it
merely EXCHANGES the two partners of each 2D E-irrep (sigma_d = +1 <-> -1),
and an exchange is a permutation. So each g acts on a configuration as

    g : |n_0, ..., n_{M-1}>  ->  s_g(n) |n'_{p(0)}, ..., n'_{p(M-1)}>

with p a cluster permutation, n' = local_map[n] a relabelling of the 16 local
states, and s_g(n) = prod_k local_sign[n_k] an overall +-1. That is an index
gather plus a sign -- no linear algebra at sample time.

build_symmetry_tables() verifies the monomial property and raises if it fails,
so a parameter regime where this breaks down is an error rather than silently
wrong physics.

What is still NOT recoverable
------------------------------
Translations by an ODD number of sites. Those re-tile the lattice into a
DIFFERENT 2x2 partition, which does not map cluster-eigenbasis configurations
to cluster-eigenbasis configurations at all. For an LxL lattice the ansatz
therefore carries (L/2)^2 of the L^2 translations, i.e. 25%. This is a genuine
cost of the eigenbasis and is not fixable by projection.

(Contrast Viteritti et al., where the 4-site patch IS the crystallographic unit
cell of the Shastry-Sutherland lattice, so patch translations exhaust the full
Bravais group and nothing is lost.)

Cost
----
|G| forward passes per sample. The full C4v x Z2(spin) group is 16x. Follow the
staged protocol of Appendix D of the paper -- optimise unsymmetrised first,
then switch symmetries on for a short final phase -- rather than paying 16x
for the whole run. Pick a subgroup with `elements=`; useful ladders:

    ("I",)                                  1x   off
    ("I", "sigma_d")                        2x
    ("I", "C2", "sigma_d", "sigma_a")       4x   Klein four-group
    C4V_ELEMENTS                            8x   full point group
    C4V_ELEMENTS + spin_flip=True          16x   everything available

Conventions (must match the rest of the code)
----------------------------------------------
  cluster index   n = cx + Cx*cy          (get_cluster_graph_2D_J2)
  intra-cluster   k = dx + 2*dy           (site() in get_ops_2x2)
  basis index     b = s0*8 + s1*4 + s2*2 + s3   (site 0 most significant)

The C4v elements are taken about the centre of cluster (0,0), at (0.5, 0.5) in
site coordinates, so every element maps the 2x2 tiling onto itself.

python 3.13 / netket 3.16.1.post1
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# The eight elements of C4v, as maps on global site coordinates (X, Y) of an
# L x L torus, about the centre of cluster (0,0).
# ---------------------------------------------------------------------------
_C4V = {
    "I":       lambda X, Y, L: (X, Y),
    "C4":      lambda X, Y, L: ((1 - Y) % L, X),
    "C2":      lambda X, Y, L: ((1 - X) % L, (1 - Y) % L),
    "C4_3":    lambda X, Y, L: (Y, (1 - X) % L),
    "sigma_d": lambda X, Y, L: (Y, X),                      # about y = x
    "sigma_a": lambda X, Y, L: ((1 - Y) % L, (1 - X) % L),  # anti-diagonal
    "sigma_x": lambda X, Y, L: (X, (1 - Y) % L),            # about y = 1/2
    "sigma_y": lambda X, Y, L: ((1 - X) % L, Y),            # about x = 1/2
}

C4V_ELEMENTS = ("I", "C4", "C2", "C4_3", "sigma_d", "sigma_a", "sigma_x", "sigma_y")


def _perm_op_2x2(pi):
    """16x16 permutation moving the content of site k to site pi[k].

    Identical to the helper of the same name in run_2D_eigenbasis.py; kept
    local so this module has no import-time dependency on the entry point.
    """
    P = np.zeros((16, 16))
    for b in range(16):
        s = [(b >> (3 - k)) & 1 for k in range(4)]
        t = [0] * 4
        for k in range(4):
            t[pi[k]] = s[k]
        P[sum(t[k] << (3 - k) for k in range(4)), b] = 1.0
    return P


def _spin_flip_op():
    """Global spin flip on a plaquette: product of sigma_x over the 4 sites.

    In the computational basis this is the bit complement b -> b XOR 0b1111.
    """
    F = np.zeros((16, 16))
    for b in range(16):
        F[b ^ 0b1111, b] = 1.0
    return F


def _split(X, Y, Cx):
    """Global site (X, Y) -> (cluster index, intra-cluster site index)."""
    cx, dx = divmod(X, 2)
    cy, dy = divmod(Y, 2)
    return (cx % Cx) + Cx * (cy % Cx), dx + 2 * dy


def _element_action(name, Cx):
    """Cluster permutation and intra-cluster site permutation for element g.

    Returns (cperm, site_perm) where cperm[n] is the cluster that cluster n is
    sent to, and site_perm[k] is the site that intra-cluster site k is sent to.

    Asserts that the site permutation is the SAME for every cluster -- true
    because each element is an affine map and the tiling is regular, but cheap
    to check and a sharp tripwire if a convention ever drifts.
    """
    L = 2 * Cx
    f = _C4V[name]
    cperm = np.full(Cx * Cx, -1, dtype=int)
    site_perm = None
    for cy in range(Cx):
        for cx in range(Cx):
            for dy in (0, 1):
                for dx in (0, 1):
                    X, Y = 2 * cx + dx, 2 * cy + dy
                    n, k = _split(X, Y, Cx)
                    Xn, Yn = f(X, Y, L)
                    n2, k2 = _split(Xn, Yn, Cx)
                    if cperm[n] == -1:
                        cperm[n] = n2
                    elif cperm[n] != n2:
                        raise RuntimeError(
                            f"element {name!r} does not map cluster {n} to a "
                            f"single cluster ({cperm[n]} vs {n2}); the tiling "
                            f"is not preserved.")
                    if site_perm is None:
                        site_perm = np.full(4, -1, dtype=int)
                    if site_perm[k] == -1:
                        site_perm[k] = k2
                    elif site_perm[k] != k2:
                        raise RuntimeError(
                            f"element {name!r} has a cluster-dependent site "
                            f"permutation at site {k}: {site_perm[k]} vs {k2}.")
    if (cperm < 0).any() or len(set(cperm.tolist())) != len(cperm):
        raise RuntimeError(f"element {name!r} gave a non-bijective cluster map.")
    return cperm, site_perm


def _monomial(M, what, tol=1e-9):
    """Decompose a monomial matrix into (index map, signs).

    Raises if M is not monomial -- i.e. if the symmetry would map one basis
    configuration into a superposition, in which case the projection is not
    representable as an index gather and this whole approach is invalid.
    """
    nz = np.abs(M) > tol
    counts = nz.sum(axis=0)
    if not (counts == 1).all():
        worst = int(counts.max())
        raise RuntimeError(
            f"{what} is not monomial in the cluster eigenbasis (a column has "
            f"{worst} nonzero entries). The local basis does not support an "
            f"index-permutation projection for this element; check J1/J2/"
            f"sign_rule, or drop this element from `elements`.")
    idx = np.argmax(nz, axis=0)
    val = M[idx, np.arange(M.shape[1])]
    if not np.allclose(np.abs(val), 1.0, atol=1e-8):
        raise RuntimeError(f"{what} has non-unimodular entries: "
                           f"|v| in [{np.abs(val).min()}, {np.abs(val).max()}].")
    return idx.astype(np.int32), np.sign(val).astype(np.float64)


def build_symmetry_tables(v, Cx, elements=C4V_ELEMENTS, spin_flip=False):
    """Build the gather tables that define the projection group.

    Parameters
    ----------
    v : (16, 16) ndarray
        The cluster eigenbasis, columns = local eigenstates. Obtain it from
        get_ops_2x2(..., return_basis=True).
    Cx : int
        Clusters per side (= L // 2). The cluster grid is Cx x Cx.
    elements : tuple of str
        Which C4v elements to include. See C4V_ELEMENTS.
    spin_flip : bool
        Also include the global spin flip, doubling the group. Only a symmetry
        of the Heisenberg Hamiltonian in the Sz = 0 sector.

    Returns
    -------
    (inv_cperms, local_maps, local_signs, names) as nested tuples, hashable so
    they can be Flax static attributes. inv_cperms is the INVERSE cluster
    permutation, which is what the gather in SymmProject needs.
    """
    F = _spin_flip_op()
    inv_cperms, local_maps, local_signs, names = [], [], [], []

    for name in elements:
        cperm, site_perm = _element_action(name, Cx)
        P = _perm_op_2x2(site_perm.tolist())
        for flip in ((False, True) if spin_flip else (False,)):
            A = F @ P if flip else P
            tag = f"{name}*flip" if flip else name
            lmap, lsign = _monomial(v.T @ A @ v, f"element {tag!r}")
            inv = np.argsort(cperm)
            inv_cperms.append(tuple(int(i) for i in inv))
            local_maps.append(tuple(int(i) for i in lmap))
            local_signs.append(tuple(float(s) for s in lsign))
            names.append(tag)

    return (tuple(inv_cperms), tuple(local_maps),
            tuple(local_signs), tuple(names))


# ---------------------------------------------------------------------------
# The Flax wrapper
# ---------------------------------------------------------------------------
class SymmProject(nn.Module):
    """Symmetry-project a complex log-amplitude module over a finite group.

    Computes, stably in log space,

        Log Psi~(sigma) = Log [ (1/|G|) sum_g s_g(sigma) Psi(g^-1 sigma) ]

    Each -1 sign is carried as an added i*pi in the log-amplitude, which is
    exact because the wrapped module is complex-valued; that avoids needing a
    signed logsumexp.

    Structurally this is the same pattern as FlipExpSum in
    importance_sampling_nqs/packages/deepnets/nn/blocks/symmetrize.py, with
    |G| signed index-permutations in place of a single global sign flip.

    Costs |G| evaluations of `module` per sample.
    """
    module: nn.Module
    inv_cperms: tuple
    local_maps: tuple
    local_signs: tuple

    @nn.compact
    def __call__(self, spins):
        x = jnp.asarray(spins, dtype=jnp.int32)
        squeeze_out = x.ndim == 1
        if squeeze_out:
            x = x[None, :]
        B, M = x.shape

        inv_cperms = jnp.asarray(self.inv_cperms, dtype=jnp.int32)    # (G, M)
        local_maps = jnp.asarray(self.local_maps, dtype=jnp.int32)    # (G, 16)
        local_signs = jnp.asarray(self.local_signs, dtype=jnp.float64)
        G = inv_cperms.shape[0]

        # Build all |G| images at once.
        #   xr[g,b,k] = x[b, inv_cperms[g,k]]          undo the cluster permutation
        #   xg[g,b,k] = local_maps[g, xr[g,b,k]]       relabel the local eigenstates
        #   sg[g,b]   = prod_k local_signs[g, xr[g,b,k]]
        xr = jnp.transpose(x[:, inv_cperms], (1, 0, 2))               # (G, B, M)
        gidx = jnp.arange(G)[:, None, None]
        xg = local_maps[gidx, xr]                                     # (G, B, M)
        sg = jnp.prod(local_signs[gidx, xr], axis=-1)                 # (G, B)

        # ONE call on a (G*B, M) batch, not G calls on (B, M).
        #
        # This matters much more than the FLOP count suggests: the network is
        # tiny (n_clusters tokens, d_model ~ 72) and float64, so a step is
        # dominated by kernel-launch overhead, not arithmetic -- which is why a
        # B200 clocks the same as an H200 here. G separate invocations would pay
        # the launch overhead G times over; one fused batch pays it once.
        #
        # Flax shares parameters across calls to the same submodule instance
        # either way, so this is purely a scheduling change, not a change to the
        # parameter tree.
        #
        # It does multiply the effective batch by G, so scale [run] CHUNK_SIZE
        # down by roughly |G| if you hit memory pressure.
        lp = self.module(xg.reshape(G * B, M)).reshape(G, B)

        # Stable complex log-mean-exp, shifting by the max REAL part.
        #
        # The +-1 is applied as a REAL multiplication rather than as a +i*pi
        # shift of the log-amplitude. Both are algebraically exact, but
        # exp(i*pi) evaluates to -1 + 1.22e-16j in float64, and that residue
        # dominates wherever the group sum nearly cancels -- precisely the
        # configurations whose amplitude is most delicate. A real +-1 cancels
        # exactly.
        #
        # NOTE the sum genuinely vanishes for some configurations: if sigma is
        # invariant under some g with s_g(sigma) = -1 then Psi~(sigma) = 0 and
        # this returns -inf. That is the correct answer -- a node FORCED by
        # symmetry, which is also why an "invariant by construction" ansatz
        # cannot represent this ground state. It is harmless downstream:
        # |Psi~|^2 = 0 so the sampler never visits such a configuration, and as
        # a connected state in a local energy it contributes a ratio of exactly
        # 0. The Jacobian is only ever taken at sampled configurations, which by
        # definition have nonzero amplitude.
        m = jnp.max(lp.real, axis=0, keepdims=True)
        out = jnp.squeeze(m, 0) + jnp.log(
            jnp.mean(sg * jnp.exp(lp - m), axis=0))
        return out[0] if squeeze_out else out
