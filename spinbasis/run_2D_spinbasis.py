"""
run_2D_spinbasis.py
===================
2D square-lattice J1-J2 Heisenberg model in the plain sigma^z SPIN BASIS: VMC
with a choice of wavefunction (Jastrow, RBM, ViT), plus an exact-diagonalization
reference for small L.

Companion to ../eigenbasis/run_2D_eigenbasis.py. Same config layout, same
driver (overdispersed VMC_NG with optional adaptive alpha), same offset-aware
lr / diag_shift schedules, same .progress / .done / .timeout sentinels, and the
same run.* output files, so run.sh auto-resubmits it unchanged and
the notebooks can read its sweeps (they live in data/spinbasis/ of this repo).

What differs from the eigenbasis script
---------------------------------------
1. BASIS. Hilbert space is nk.hilbert.Spin(1/2, N=L^2, total_sz=0); there are
   no clusters. The Hamiltonian is NetKet's Heisenberg on a periodic Hypercube
   with max_neighbor_order=2, J=[J1, J2], and the sampler is MetropolisExchange.

2. UNITS. nk.operator.Heisenberg is written with Pauli matrices, i.e. it is
   4x the S = sigma/2 Hamiltonian (the ViT tutorial divides by 4 when it plots).
   Here the operator is multiplied by 1/4 at construction, so every logged
   energy is in the same S units as the eigenbasis runs and as E0_PER_SITE_REF
   in the plotting notebook. Consequence: the SR gradient is 1/4 of the Pauli
   one, so learning rates are directly comparable to config_eigenbasis.toml
   and roughly 4x the tutorials' values.

3. WAVEFUNCTION. Chosen by [model] type = "jastrow" | "rbm" | "vit", see
   helpers/wavefunctions.py. Each type has its own [model.<type>] table.

4. CHECKPOINTS stay consistent with .progress. The eigenbasis script writes
   .progress every SAVE_EVERY steps but run.mpack only when a window ends, and
   lets JsonLog write a second, differently named .mpack per window. Here one
   function writes run.mpack and then .progress, both atomically, at the same
   step, every SAVE_EVERY steps and at the end; JsonLog saves no parameters.
   A hard kill or scancel therefore loses at most SAVE_EVERY steps, and a plain
   resubmit of the array task resumes correctly with no manual repair.

5. LOG FILE NAMES are zero-padded and carry the window's start step,
   "run step 000000.log", "run step 004400.log", ..., so sorting them by name
   sorts them by time. The notebook merges them keeping the last occurrence of
   each iteration, so after a hard kill the resumed window's values win over the
   discarded tail of the killed one.

python 3.13 / netket 3.19
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import netket as nk

import jax
import optax
import flax
import scipy.sparse.linalg as sla
from netket.callbacks import InvalidLossStopping

import advanced_drivers as advd

from helpers.wavefunctions import build_model

job_start = time.time()
print('\n#############')
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(jax.devices())
print('#############\n', flush=True)

DATA_ROOT = sys.argv[2]


# ===========================================================================
# args
# ===========================================================================
def parse_config():
    global config
    with open(sys.argv[1], "rb") as f:
        config = json.load(f)

    # System
    global L, J1, J2, sign_rule
    system = config["system"]
    # One config per basis: refuse a config written for the other basis. A
    # config without the key (written before the field existed) is accepted.
    if system.get("basis", "spin") != "spin":
        raise SystemExit(f"this is the spin-basis entry point, but [system] basis = "
                         f"{system['basis']!r}; use eigenbasis/run_2D_eigenbasis.py")
    L = system["L"]
    J1 = system["J1"]
    J2 = system["J2"]
    sign_rule = system["sign_rule"]

    # Sampler
    global SAMPLER, MONTECARLO, seed, chains, discard_per_chain, d_max
    sampler = config["sampler"]
    SAMPLER = sampler["SAMPLER"]
    MONTECARLO = sampler["MONTECARLO"]
    seed = sampler["seed"]
    chains = sampler["chains"]
    discard_per_chain = sampler["discard_per_chain"]
    d_max = sampler["d_max"]

    # Distribution: `alpha` is the INITIAL overdispersion exponent of
    # q = |psi|^alpha; with auto_alpha=True VMC_NG moves it every step along the
    # gradient of the gradient-SNR. alpha = 2 with auto_alpha = false is plain SR.
    global alpha, auto_alpha, use_ntk, on_the_fly
    distr = config["distribution"]
    alpha = distr["alpha"]
    auto_alpha = bool(distr["auto_alpha"])
    use_ntk = distr["use_ntk"]
    on_the_fly = distr["on_the_fly"]

    # Model: iteration target here (as in the eigenbasis config); the
    # wavefunction itself is built from the whole section by build_model.
    global iters, model_cfg
    model_cfg = config["model"]
    iters = model_cfg["iter"]

    # Optimizer
    global learn_rate1, learn_rate2, diag_shift1, diag_shift2, lr_step, ds_step
    opt = config["optimizer"]
    learn_rate1 = opt["initial_learning_rate"]
    learn_rate2 = opt["final_learning_rate"]
    diag_shift1 = opt["initial_diagonal_shift"]
    diag_shift2 = opt["final_diagonal_shift"]
    lr_step = opt["lr_transition_steps"]
    ds_step = opt["ds_transition_steps"]

    # Run
    global max_runtime, save_every, print_every, diag_every, chunk_size
    global freeze_patience
    run = config["run"]
    max_runtime = run["MAX_RUNTIME_MIN"]
    save_every = run["SAVE_EVERY"]
    print_every = run["PRINT_EVERY"]
    diag_every = run["DIAG_EVERY"]
    chunk_size = run["CHUNK_SIZE"]
    # .get: runs resumed from a sweep submitted before this key existed read
    # their own config.json, which does not have it.
    freeze_patience = int(run.get("FREEZE_PATIENCE", 3))

    # Misc
    global EXACT_SAMPLER_MAX_STATES
    EXACT_SAMPLER_MAX_STATES = config["misc"]["EXACT_SAMPLER_MAX_STATES"]

parse_config()

# Config checks and derived parameters
sign_rule = [c.upper() == 'T' for c in sign_rule]
assert len(sign_rule) == 2, "sign_rule must be 2 chars T/F, e.g. 'FF' for [J1,J2]"
sr_tag = ''.join('T' if s else 'F' for s in sign_rule)
N_MONTECARLO = int(2048 * MONTECARLO)          # integer multiple of 2048
n_discard = discard_per_chain if discard_per_chain >= 0 else None
N_sites = L * L

print(f'L={L}x{L}  N={N_sites}  J1={J1} J2={J2}  sign_rule={sr_tag} {sign_rule}  iters={iters} (total)')
print(f'N_samples={N_MONTECARLO} (=2048*{MONTECARLO})  seed={seed}  '
      f'n_chains={chains}  n_discard_per_chain={n_discard if n_discard is not None else "auto"}  '
      f'd_max={d_max}')
print(f'lr=[{learn_rate1},{learn_rate2}] ds=[{diag_shift1},{diag_shift2}] trans(lr/ds)={lr_step}/{ds_step}')
# Adaptive alpha: on_the_fly=True routes the update through `srt_onthefly`,
# which does not accept the `is_jac` argument the SNR derivative is carried in.
# That fails deep inside a jit, so check it here instead.
if auto_alpha and on_the_fly:
    raise ValueError("auto_alpha=True is incompatible with on_the_fly=True "
                     "(srt_onthefly takes no is_jac argument). Set "
                     "[distribution] on_the_fly = false.")
print(f'driver: overdispersed VMC_NG  ALPHA={alpha} '
      f'({"initial, auto_is tunes it" if auto_alpha else "fixed"})  use_ntk={use_ntk}  '
      f'on_the_fly={on_the_fly}  chunk_size={chunk_size}', flush=True)


# ===========================================================================
# lattice, Hilbert space, Hamiltonian
# ===========================================================================
graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)

# The ViT's patch extraction reshapes the flat configuration as an L x L grid,
# so it needs site i at (i // L, i % L). Hypercube does this today; check it,
# because a different ordering would not error, just scramble the patches.
_expected = np.stack(np.divmod(np.arange(N_sites), L), axis=1)
if not np.allclose(np.asarray(graph.positions), _expected):
    raise RuntimeError("Hypercube site ordering is not row-major (i // L, i % L); "
                       "helpers/vit.py extract_patches2d would scramble the patches.")

hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
# 1/4: Pauli -> S = sigma/2 units (see point 2 of the module docstring).
ha = 0.25 * nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[J1, J2],
                                   sign_rule=sign_rule)
ha_jax = ha.to_jax_operator()

try:
    n_states = hi.n_states
except Exception:            # RuntimeError: too large to be indexed
    n_states = None
print('n_states (Sz=0) =', n_states if n_states is not None else 'too large to index')

if SAMPLER == 'auto':
    sampler_name = 'exact' if (n_states is not None and n_states <= EXACT_SAMPLER_MAX_STATES) else 'metro'
else:
    sampler_name = SAMPLER
if sampler_name not in ('exact', 'metro'):
    raise ValueError(f"[sampler] SAMPLER must be 'auto', 'exact' or 'metro', got {SAMPLER!r}")


# ===========================================================================
# ED reference (small systems only)
# ===========================================================================
E0 = None
if n_states is not None and n_states <= EXACT_SAMPLER_MAX_STATES:
    t = time.time()
    E0 = float(sla.eigsh(ha.to_sparse(), k=1, which='SA', return_eigenvectors=False)[0])
    print(f'ED ground-state energy  E0 = {E0:.8f}  E0/site = {E0 / N_sites:.8f}   '
          f'({time.time()-t:.1f}s)', flush=True)
else:
    print('ED skipped (sector too large).')


# ===========================================================================
# output paths + sentinels
# ===========================================================================
fname = os.path.join(DATA_ROOT, "run")
print(f'\n{fname=}')

PROGRESS = fname + '.progress'
DONE = fname + '.done'
CKPT = fname + '.mpack'
TIMEOUT = fname + '.timeout'
FROZEN = fname + '.frozen'
DIAG = fname + '.diag.jsonl'
for _stale in (TIMEOUT, FROZEN):
    if os.path.exists(_stale):
        os.remove(_stale)

def _read_progress_json():
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def read_progress():
    return int(_read_progress_json().get('cum_steps', 0))

def read_progress_alpha():
    """Adaptive alpha lives OUTSIDE vstate.variables, so the .mpack does not
    carry it; without this it would re-anneal from the config value on every
    resubmission. None when absent (fresh run, or fixed alpha)."""
    a = _read_progress_json().get('alpha', None)
    return None if a is None else float(a)

def write_progress(cum, alpha_now=None):
    tmp = PROGRESS + '.tmp'
    rec = {'cum_steps': int(cum)}
    if alpha_now is not None:
        rec['alpha'] = float(alpha_now)
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, PROGRESS)

jobid = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
with open(os.path.join(DATA_ROOT, f'.runinfo_{jobid}'), 'w') as f:
    f.write(fname + '\n')


# ===========================================================================
# fresh vs resume
# ===========================================================================
do_resume = os.path.exists(CKPT)

# .diag.jsonl is append-only and keyed on the global step; start it empty on a
# fresh run.
if not do_resume and os.path.exists(DIAG):
    os.remove(DIAG)

step_offset = read_progress() if do_resume else 0
print(f'do_resume={do_resume}  step_offset={step_offset}')
remaining = max(0, iters - step_offset)
if remaining == 0:
    print(f'\n*** Target {iters} already reached (cum={step_offset}). Writing .done. ***')
    open(DONE, 'w').close(); sys.exit(0)
print(f'remaining this submission = {remaining}')


# ===========================================================================
# OFFSET-AWARE linear schedules: a resubmit continues from the global step
# instead of re-annealing from the initial values.
# ===========================================================================
_lr = optax.linear_schedule(init_value=learn_rate1, end_value=learn_rate2, transition_steps=lr_step)
_ds = optax.linear_schedule(init_value=diag_shift1, end_value=diag_shift2, transition_steps=ds_step)
lr = lambda c: _lr(c + step_offset)
diag_shift = lambda c: _ds(c + step_offset)

print(f'\nSchedules @ global step {step_offset}:')
print(f'  lr         = {float(lr(0)):.4e}   ({learn_rate1} -> {learn_rate2} over {lr_step})')
print(f'  diag_shift = {float(diag_shift(0)):.4e}   ({diag_shift1} -> {diag_shift2} over {ds_step})', flush=True)


# ===========================================================================
# model + sampler + variational state
# ===========================================================================
model, model_desc = build_model(model_cfg, graph=graph, L=L)
print(f'\nwavefunction: {model_desc}', flush=True)

if sampler_name == 'exact':
    sampler = nk.sampler.ExactSampler(hilbert=hi)
else:
    # d_max=2 lets the exchange move swap next-nearest neighbours too, which is
    # what the J2 bonds connect.
    sampler = nk.sampler.MetropolisExchange(hilbert=hi, graph=graph, d_max=d_max,
                                            n_chains=chains)

# The sampler seed is shifted by step_offset so each resubmission draws a fresh
# Markov-chain stream rather than replaying the first window's random numbers.
# (The parameter seed does not matter on a resume: the checkpoint overwrites it.)
mcstate_kwargs = dict(sampler=sampler, model=model, n_samples=N_MONTECARLO,
                      seed=seed, sampler_seed=seed + step_offset)
if sampler_name != 'exact' and n_discard is not None:
    mcstate_kwargs['n_discard_per_chain'] = n_discard
vstate = nk.vqs.MCState(**mcstate_kwargs)
if chunk_size > 0:
    vstate.chunk_size = chunk_size
n_params = nk.jax.tree_size(vstate.parameters)
print(f'sampler={sampler_name}  n_params={n_params}  '
      f'n_discard_per_chain={getattr(vstate, "n_discard_per_chain", "n/a")}', flush=True)

if do_resume:
    print('\n### resuming from checkpoint ###')
    with open(CKPT, 'rb') as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())


# ===========================================================================
# driver: overdispersed VMC_NG (advanced_drivers), as in the eigenbasis script
# ===========================================================================
opt = optax.sgd(learning_rate=lr)
modulus_distribution = advd.driver.overdispersed_distribution(alpha=alpha)

# Resume the exponent where the previous submission left it. Only for an
# adaptive run: with a fixed alpha the config value is the truth and must win.
if auto_alpha and do_resume:
    _a_prev = read_progress_alpha()
    if _a_prev is not None:
        modulus_distribution.q_variables = {"alpha": jax.numpy.array([_a_prev])}
        print(f'resumed adaptive alpha = {_a_prev:.4f} (config initial was {alpha})')
    else:
        print(f'no alpha in .progress; adaptive alpha restarts from {alpha}')

driver = advd.driver.VMC_NG(hamiltonian=ha_jax,
                            optimizer=opt,
                            sampling_distribution=modulus_distribution,
                            variational_state=vstate,
                            diag_shift=diag_shift,
                            auto_is=auto_alpha,
                            use_ntk=use_ntk,
                            on_the_fly=on_the_fly)

# The SNR derivative behind auto_alpha reads the jacobian as interleaved
# real/imag rows, which is only right in mode='complex'. The mode follows the
# model: complex output (the ViT, or param_dtype = "complex128") gives
# 'complex'; a float64 RBM/Jastrow gives 'real'.
if auto_alpha and driver.mode != 'complex':
    raise ValueError(f"auto_alpha=True needs the driver in mode='complex', but "
                     f"{model_cfg['type']!r} gave mode={driver.mode!r}. Either set "
                     f"[model.{model_cfg['type']}] param_dtype = \"complex128\", or "
                     f"set [distribution] auto_alpha = false.")
print(f'driver mode = {driver.mode}', flush=True)


def current_alpha():
    """Live overdispersion exponent as a scalar."""
    try:
        return float(np.asarray(modulus_distribution.q_variables["alpha"]).mean())
    except Exception:
        return float(alpha)


def save_checkpoint(cum):
    """Write run.mpack, THEN .progress, each atomically (tmp file + rename).

    `cum` must be the number of parameter updates contained in vstate.variables.
    Order matters: killed between the two writes, the .mpack is at most one
    SAVE_EVERY ahead of .progress, so the resumed run just repeats a few steps
    from slightly better parameters. The reverse order could leave .progress
    ahead of the parameters, which silently shortens the run."""
    tmp = CKPT + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(flax.serialization.to_bytes(vstate.variables))
    os.replace(tmp, CKPT)
    write_progress(cum, current_alpha() if auto_alpha else None)


# ===========================================================================
# callbacks
#
# advanced_drivers runs these AFTER computing step s's update but BEFORE
# applying it. So inside a callback at step s, vstate holds exactly s updates,
# and a callback returning False stops the run with step s never applied.
# ===========================================================================
class ProgressWriter:
    def __init__(self, every, print_every, E0):
        self.every = every
        self.print_every = print_every
        self.E0 = E0

    def __call__(self, step, log_data, driver):
        if step > step_offset and step % self.every == 0:
            save_checkpoint(step)
        if step % self.print_every == 0:
            try:
                e = float(np.real(log_data["Energy"].mean))
                a = current_alpha() if auto_alpha else None
                tail = f'  alpha={a:.4f}' if a is not None else ''
                msg = f'[{step:6d}] E={e:.8f}  E/site={e / N_sites:.8f}'
                if self.E0 is not None:
                    msg += f'  rel.err={abs(e - self.E0) / abs(self.E0):.3e}'
                print(msg + tail, flush=True)
            except Exception:
                pass
        return True

class TimeBudget:
    def __init__(self, minutes):
        self.deadline = time.time() + minutes * 60 if minutes > 0 else None
        self.stopped = False
    def __call__(self, step, log_data, driver):
        if self.deadline is not None and time.time() > self.deadline:
            self.stopped = True
            print(f'\n*** wall-clock budget reached at step {step}; stopping gracefully ***', flush=True)
            return False
        return True

class SamplingDiag:
    """Cheap sampling-health diagnostics -> <fname>.diag.jsonl (append-only,
    keyed on the same global iteration numbers as the .log). One extra forward
    pass (log|psi| of the current samples) every `every` steps.

    Logged fields:
      ess_w / ess_w_frac : effective sample size of the importance weights
                           w ~ |psi|^(2-alpha); ~n_samples at alpha = 2.
      rhat_logpsi        : split-R-hat of log|psi| across Markov chains
                           (~1.0 = chains agree; >>1 = not mixing).
      acceptance         : Metropolis acceptance fraction (none for exact).
      var_eloc / _per_site : Var(E_loc), and divided by N_sites.

    A broken diagnostic must never kill a run: every part is guarded -- the one
    deliberate exception is the freeze guard below, which stops the driver when
    the wavefunction has demonstrably collapsed (see `frozen`)."""

    # Collapse signature. Healthy runs sit orders of magnitude away from both:
    # the 2026-09-28 sweep held acc ~ 0.3 with Var/site between 1e-2 (RBM) and
    # 3e-5 (ViT), while the frozen Jastrow read acc = 0.00, Var/site = 9e-32.
    FREEZE_ACC = 1e-3
    FREEZE_VAR = 1e-12

    def __init__(self, alpha, every, diag_path, n_sites, patience=0):
        # fallback only: the weights must use the exponent the driver is
        # sampling at right now, or ess_w drifts once adaptive alpha moves
        self.alpha = float(alpha)
        self.every = max(1, int(every))
        self.path = diag_path
        self.patience = int(patience)
        self.streak = 0            # consecutive diags matching the signature
        self.frozen = False        # set when the guard fires; read at finalize
        self.n_sites = int(n_sites)

    def _alpha(self, driver):
        try:
            return float(np.asarray(
                driver.sampling_distribution.q_variables["alpha"]).mean())
        except Exception:
            return self.alpha

    @staticmethod
    def _split_rhat(x):
        # x: (n_chains, n_per_chain) real -> standard split-R-hat (Gelman-Rubin).
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4:
            return float('nan')
        m = x.shape[1] // 2
        h = np.concatenate([x[:, :m], x[:, m:2 * m]], axis=0)   # (2*n_chains, m)
        n = h.shape[1]
        means = h.mean(axis=1); vars = h.var(axis=1, ddof=1)
        W = float(means.size and vars.mean())
        B = float(n * means.var(ddof=1))
        if W <= 0:
            return float('nan')
        var_hat = (n - 1) / n * W + B / n
        return float(np.sqrt(var_hat / W))

    def __call__(self, step, log_data, driver):
        if step % self.every:
            return True
        rec = {"step": int(step), "alpha": self._alpha(driver)}
        try:
            v = float(np.real(log_data["Energy"].variance))
            rec["var_eloc"] = v
            rec["var_eloc_per_site"] = v / self.n_sites
        except Exception:
            pass
        vs = driver.state
        # Diagnose the chain the gradient used, q = |psi|^alpha, not vs.samples
        # (the |psi|^2 chain, which would trigger an extra MCMC pass here).
        chain = getattr(driver.sampling_distribution, 'name', None)
        try:
            afun_q, vars_q = driver.sampling_distribution(vs._apply_fun, vs.variables)
            samp = np.asarray(vs.samples_distribution(afun_q, variables=vars_q,
                                                      chain_name=chain))
            rec["chain"] = chain
        except Exception as ex:
            samp = np.asarray(vs.samples)
            rec["chain"] = "default"
            rec["chain_error"] = repr(ex)
        try:
            flat = samp.reshape(-1, samp.shape[-1])
            logpsi = np.asarray(vs.log_value(flat)).real
            wl = (2.0 - rec["alpha"]) * logpsi
            wl = wl - wl.max()
            w = np.exp(wl)
            sw = float(w.sum()); sw2 = float(np.square(w).sum())
            rec["ess_w"] = (sw * sw / sw2) if sw2 > 0 else float('nan')
            rec["ess_w_frac"] = rec["ess_w"] / w.size
            rec["n_samples"] = int(w.size)
            if samp.ndim >= 3:
                nchains = int(np.prod(samp.shape[:-2])); nper = samp.shape[-2]
                rec["rhat_logpsi"] = self._split_rhat(logpsi.reshape(nchains, nper))
        except Exception as ex:
            rec["logpsi_error"] = repr(ex)
        try:
            ss = getattr(vs, 'sampler_states', {}).get(chain) or vs.sampler_state
            acc = getattr(ss, "acceptance", None)
            if acc is None:
                na = getattr(ss, "n_accepted", None); ns = getattr(ss, "n_steps", None)
                if na is not None and ns:
                    acc = float(np.sum(na)) / float(np.sum(ns))
            if acc is not None:
                rec["acceptance"] = float(np.asarray(acc).mean())
        except Exception:
            pass
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        msg = f'    [diag {step:6d}]'
        if "alpha" in rec:             msg += f' alpha={rec["alpha"]:.4f}'
        if "ess_w_frac" in rec:        msg += f' ESS_w={rec["ess_w"]:.0f} ({100 * rec["ess_w_frac"]:.1f}%)'
        if "rhat_logpsi" in rec:       msg += f' Rhat|psi|={rec["rhat_logpsi"]:.3f}'
        if "acceptance" in rec:        msg += f' acc={rec["acceptance"]:.2f}'
        if "var_eloc_per_site" in rec: msg += f' Var/site={rec["var_eloc_per_site"]:.3e}'
        print(msg, flush=True)

        return self._check_freeze(step, rec)

    def _check_freeze(self, step, rec):
        """False => stop the driver, the wavefunction has collapsed.

        Acceptance at zero AND a machine-precision E_loc variance means psi has
        collapsed onto one configuration: E_loc is then constant, so the sampled
        gradient vanishes and SR is parked at a false stationary point. It never
        recovers, so stop instead of burning the rest of the window.

        Both quantities must be present and finite -- a missing or broken
        diagnostic must not trigger a stop -- and the signature must hold for
        `patience` diagnostics in a row, so one unlucky sample cannot end a run.
        """
        if self.patience <= 0:
            return True
        acc = rec.get("acceptance")
        var = rec.get("var_eloc_per_site")
        hit = (acc is not None and var is not None
               and np.isfinite(acc) and np.isfinite(var)
               and acc < self.FREEZE_ACC and var < self.FREEZE_VAR)
        self.streak = self.streak + 1 if hit else 0
        if self.streak < self.patience:
            return True
        self.frozen = True
        print(f'\n*** collapsed wavefunction at step {step}: '
              f'acceptance {acc:.2e} < {self.FREEZE_ACC:g} and '
              f'Var/site {var:.2e} < {self.FREEZE_VAR:g} for '
              f'{self.streak} consecutive diagnostics. Stopping; '
              f'no resubmission. ***', flush=True)
        return False

progress_cb = ProgressWriter(save_every, print_every, E0)
time_cb = TimeBudget(max_runtime)

# sampling-health diagnostics: -1 => follow print_every, 0 => off
diag_every = print_every if diag_every < 0 else diag_every
diag_cb = None
if diag_every > 0:
    diag_cb = SamplingDiag(alpha=alpha, every=diag_every, diag_path=DIAG,
                           n_sites=N_sites, patience=freeze_patience)
    print(f'sampling diagnostics -> {os.path.basename(DIAG)} every {diag_every} steps', flush=True)
    if freeze_patience > 0:
        print(f'freeze guard: stop after {freeze_patience} consecutive diags with '
              f'acc < {SamplingDiag.FREEZE_ACC:g} and Var/site < '
              f'{SamplingDiag.FREEZE_VAR:g}', flush=True)
elif freeze_patience > 0:
    print('freeze guard requested but DIAG_EVERY = 0 disables the diagnostics '
          'it reads; guard is OFF.', flush=True)

# ---- one continuous log across resubmissions --------------------------------
# The driver's step counter is seeded with the global offset, so the .log
# records global iterations; each window writes its own zero-padded file (see
# point 5 of the module docstring). save_params=False: run.mpack is written by
# save_checkpoint only, so there is exactly one checkpoint and it always
# matches .progress.
driver._step_count = int(step_offset)
logger = nk.logging.JsonLog(f'{fname} step {step_offset:06d}',
                            mode='write',
                            save_params=False,
                            write_every=save_every)

driver.run(n_iter=remaining, out=logger,
           callback=[InvalidLossStopping(monitor='mean', patience=5), progress_cb, time_cb]
                    + ([diag_cb] if diag_cb is not None else []))


# ===========================================================================
# finalize: checkpoint + progress at the true number of applied updates, then
# the summary and the sentinel run.sh branches on.
#
# Only a CLEAN stop (target reached, or wall-clock budget) overwrites the
# checkpoint. Any other stop is InvalidLossStopping, i.e. divergence, where the
# parameters are likely NaN: keep the last periodic checkpoint instead, so the
# run can be inspected or restarted from before it blew up.
# ===========================================================================
cum = int(driver.step_count)        # exact: counts updates actually applied
clean_stop = cum >= iters or time_cb.stopped
if clean_stop:
    save_checkpoint(cum)
elif os.path.exists(CKPT):
    print(f'\n*** unclean stop at step {cum}: NOT overwriting {os.path.basename(CKPT)}; '
          f'it still holds step {read_progress()} ***', flush=True)
else:
    print(f'\n*** unclean stop at step {cum}, before the first checkpoint '
          f'(SAVE_EVERY={save_every}): nothing saved ***', flush=True)
alpha_final = current_alpha()

Ef = float(np.real(vstate.expect(ha_jax).mean))
print('\n===== summary =====')
print(f'wavefunction: {model_desc}  n_params={n_params}')
if auto_alpha:
    print(f'alpha: {alpha} (initial) -> {alpha_final:.4f} (final, adaptive)')
else:
    print(f'alpha={alpha} (fixed)')
print(f'final VMC energy E = {Ef:.8f}   E/site = {Ef / N_sites:.8f}')
if E0 is not None:
    print(f'ED reference    E0 = {E0:.8f}   E0/site = {E0 / N_sites:.8f}   '
          f'rel.err = {abs(Ef - E0) / abs(E0):.3e}')

with open(fname + '.summary.json', 'w') as f:
    json.dump({"basis": "spin", "L": L, "J1": J1, "J2": J2, "sign_rule": sr_tag,
               "model_type": model_cfg["type"], "model": model_desc, "n_params": n_params,
               "E_vmc": Ef, "E_vmc_per_site": Ef / N_sites, "E0": E0,
               "rel_err": (abs(Ef - E0) / abs(E0)) if E0 is not None else None,
               "alpha": alpha, "alpha_final": alpha_final, "auto_alpha": auto_alpha,
               "use_ntk": bool(use_ntk), "on_the_fly": bool(on_the_fly),
               "chunk_size": chunk_size, "n_discard_per_chain": n_discard,
               "lr_step": lr_step, "ds_trans": ds_step,
               "cum_steps": cum,
               "checkpoint_step": read_progress() if os.path.exists(CKPT) else None,
               "N_iter_target": iters}, f, indent=2)

if cum >= iters:
    open(DONE, 'w').close()
    print(f'\n*** Target reached: cum={cum} >= {iters}. Wrote .done ***')
elif time_cb.stopped:
    open(TIMEOUT, 'w').close()
    print(f'\n*** Clean wall-clock stop at cum={cum}/{iters}. Wrote .timeout (resubmit). ***')
elif diag_cb is not None and diag_cb.frozen:
    open(FROZEN, 'w').close()
    print(f'\n*** Stopped early at cum={cum}/{iters}: collapsed wavefunction '
          f'(acceptance ~ 0 with zero E_loc variance). Wrote .frozen; '
          f'no resubmission. ***')
else:
    print(f'\n*** Stopped early at cum={cum}/{iters} WITHOUT clean time-stop '
          f'(divergence?). No resubmission. ***')

print('\nRuntime:', (time.time() - job_start) / 60, 'mins')
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
jax.clear_caches()
