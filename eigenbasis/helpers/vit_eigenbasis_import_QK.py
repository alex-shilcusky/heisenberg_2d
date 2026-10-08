import numpy as np
import matplotlib.pyplot as pl
import netket as nk
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from einops import rearrange

# class Embed_EB(nn.Module): # this is the learnable embedding test version
#     d_model : int
#     patch_size: int
#     embedding_table: tuple  # kept for API compatibility
#     param_dtype = jnp.float64

#     def setup(self):
#         n_eigenstates = len(self.embedding_table)
#         self.embed = nn.Embed(
#             num_embeddings=n_eigenstates,
#             features=self.d_model,
#             param_dtype=self.param_dtype,
#         )

#     def __call__(self, x):
#         x = jnp.asarray(x, dtype=jnp.int32)
#         return self.embed(x)

class Embed_EB(nn.Module):
    d_model : int           # dimensionality of the embedding space
    patch_size: int         # linear patch size (kept for API compatibility)
    embedding_table: tuple  # tuple of tuples, shape (local_dim, n_features)
                            # build as: tuple(map(tuple, emb_table))
                            # where emb_table = np.stack([D, Sz], axis=-1)
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)

    def __call__(self, x):
        # Convert the hashable tuple back to a traced jnp array at call time.
        # This replaces the old hardcoded extract_patches lookup.
        table = jnp.array(self.embedding_table, dtype=self.param_dtype)  # (local_dim, n_features)
        x = table[x]   # (..., n_patches, n_features)
        x = self.embed(x)
        return x
  

##########
from functools import partial
@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll1d(spins, i):
    return jnp.roll(spins, i, axis=-1)


class FMHA(nn.Module):
    d_model : int       # dimensionality of the embedding space
    n_heads: int        # number of heads
    n_patches: int      # lenght of the input sequence
    attn_mode: str = 'f' # can be s, d, f, or c
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):

        if self.attn_mode == "s" or self.attn_mode == "d" or self.attn_mode == "c":
            self.q = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), 
            param_dtype=jnp.float64, dtype=jnp.float64)
            self.k = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), 
            param_dtype=jnp.float64, dtype=jnp.float64)


        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)
        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)

        if self.attn_mode != "c":
            if self.transl_invariant:

                self.alpha = self.param("alpha", nn.initializers.xavier_uniform(), (self.n_heads, self.n_patches), self.param_dtype)
                
                # We need to adjust how we handle the translation invariance
                # Create a list to store rolled versions
                alpha_rolled = []
                
                # Manually create the rolled versions for each position
                for i in range(self.n_patches):
                    rolled = jnp.roll(self.alpha, i, axis=1)
                    alpha_rolled.append(rolled)
                    
                # Stack the rolled versions
                self.alpha = jnp.stack(alpha_rolled, axis=1)
            else:
                self.alpha = self.param("alpha", nn.initializers.xavier_uniform(), (self.n_heads, self.n_patches, self.n_patches), self.param_dtype)

    def __call__(self, x):
        if self.attn_mode == "s" or self.attn_mode == "d" or self.attn_mode == "c":
            q = self.q(x)
            k = self.k(x)
            q = rearrange(q, 'batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff', n_heads=self.n_heads)
            k = rearrange(k, 'batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff', n_heads=self.n_heads)

            # calculate attention matrix
            depth = q.shape[-1]
            q = q / jnp.sqrt(depth)
            # attn weight shape is (batch..., num_heads, q_length, kv_length)
            attn_weights = jnp.einsum('...qhd,...khd->...hqk', q, k)
            # attn_weights = jax.nn.softmax(attn_weights)
            
        if self.attn_mode == "s":
            attn_weights = jax.nn.softmax(self.alpha + attn_weights)
        elif self.attn_mode == 'd':
            attn_weights = self.alpha + jax.nn.softmax(attn_weights)
        elif self.attn_mode == 'f':
            attn_weights = self.alpha
        elif self.attn_mode == 'c':
            attn_weights = jax.nn.softmax(attn_weights)

        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(v, 'batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff', n_heads=self.n_heads)

        # factored attention mechanism
        v = rearrange(v, 'batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff')

        # x = jnp.matmul(self.alpha, v)
        x = jnp.matmul(attn_weights, v)
        x = rearrange(x, 'batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff')

        # concatenate the different heads
        x = rearrange(x, 'batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)')

        # the representations of the different heads are combined together
        x = self.W(x)

        return x

########################
class EncoderBlock(nn.Module):
    d_model : int      # dimensionality of the embedding space
    n_heads: int       # number of heads
    n_patches: int     # lenght of the input sequence
    transl_invariant: bool = False
    attn_mode: str = 'f'
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(d_model=self.d_model, n_heads=self.n_heads, n_patches=self.n_patches,attn_mode=self.attn_mode, transl_invariant=self.transl_invariant)

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential([
            nn.Dense(4*self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype),
            nn.gelu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype),
        ])


    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff( self.layer_norm_2(x) )
        return x
    
    
################################################################################
class Encoder(nn.Module):
    num_layers: int     # number of layers
    d_model : int       # dimensionality of the embedding space
    n_heads: int        # number of heads
    n_patches: int      # lenght of the input sequence
    attn_mode: str = 'f'
    transl_invariant: bool = False

    def setup(self):
        self.layers = [EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, n_patches=self.n_patches, 
        attn_mode=self.attn_mode, transl_invariant=self.transl_invariant) for _ in range(self.num_layers)]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x
################################################################################

log_cosh = nk.nn.activation.log_cosh # Logarithm of the hyperbolic cosine, implemented in a more stable way

class OuputHead(nn.Module):
    d_model : int # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
        self.norm3 = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)

        self.output_layer0 = nn.Dense(self.d_model, param_dtype=self.param_dtype, kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)
        self.output_layer1 = nn.Dense(self.d_model, param_dtype=self.param_dtype, kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)

    def __call__(self, x):

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j*out_imag

        return jnp.sum(log_cosh(out), axis=-1)
    
class ViT_EB_QK(nn.Module):
    num_layers: int     # number of layers
    d_model : int       # dimensionality of the embedding space
    n_heads: int        # number of heads
    patch_size: int     # linear patch size
    embedding_table: tuple  # tuple of tuples, build as:
                            #   D = np.diag(hlocal).real
                            #   Sz = np.diag(sz_tot).real
                            #   emb_table = np.stack([D, Sz], axis=-1)
                            #   embedding_table = tuple(map(tuple, emb_table))
    attn_mode: str = 'f' # 'f': factored; 's': T5 attention; 'd': decoupled
    transl_invariant: bool = False

    @nn.compact
    def __call__(self, spins):
        x = spins
        x = jnp.asarray(x, dtype=jnp.int32) # essential change for eigenbasis embedding

        Ns = x.shape[-1]                        # number of sites
        n_patches = Ns    # lenght of the input sequence

        x = Embed_EB(d_model=self.d_model,
                  patch_size=self.patch_size,
                  embedding_table=self.embedding_table)(x)

        y = Encoder(num_layers=self.num_layers, 
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    n_patches=n_patches, 
                    attn_mode=self.attn_mode,
                    transl_invariant=self.transl_invariant)(x)

        log_psi = OuputHead(d_model=self.d_model)(y)

        return log_psi