import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
from IPython.display import display, clear_output
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import time
import pandas as pd
import random





from .Envirnonment import Environment, make_landscape, make_anti_correlated_2good
from .helper_functions import get_truncated_normal, initialize_bundle, generate_lorenz_and_gini
from .Agents import Agent

gini = generate_lorenz_and_gini




def initialize(n_agents, env_size, resource_fns, max_resource,
               metabolic_rate_mean, preferences_mean, vision,
               n_goods, bundle_mode, rng):
    """Build the environment and a list of agents with random positions and parameters.

    Parameters
    ----------
    n_agents : int
        Number of agents to create.
    env_size : tuple[int, int]
        (H, W) grid dimensions.
    resource_fns : list[callable]
        One landscape function per good.
    max_resource : float
        Peak resource value.
    metabolic_rate_mean : float
        Mean metabolic rate used when sampling per-agent, per-good rates.
    preferences_mean : list[float] or None
        Mean preference weights per good.  If None, defaults to uniform [0.5]*n_goods.
    vision : int
        Agent vision radius (Chebyshev).
    n_goods : int
        Number of goods.
    bundle_mode : str
        Initial bundle mode; passed to initialize_bundle.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    tuple[list[Agent], Environment]
    """
    env = Environment(env_size[0], env_size[1], resource_fns, max_resource)

    if preferences_mean is None:
        preferences_mean = [0.5] * n_goods

    agents = []
    occupied = set()
    for agent_id in range(n_agents):
        # Random non-colliding starting position
        while True:
            r = rng.integers(0, env.H)
            c = rng.integers(0, env.W)
            if (r, c) not in occupied:
                occupied.add((r, c))
                break
        position = (int(r), int(c))

        stochastic_preferences = np.array([
            get_truncated_normal(preferences_mean[g], sd=0.1, low=0.0, high=1.0, rng=rng)
            for g in range(n_goods)
        ])

        metabolic_rate = np.array([
            get_truncated_normal(metabolic_rate_mean, sd=0.3, low=0.1, high=4.0, rng=rng)
            for _ in range(n_goods)
        ])

        bundle = initialize_bundle(n_goods, mode=bundle_mode, rng=rng)

        agents.append(Agent(agent_id, stochastic_preferences, bundle,
                            vision, position, metabolic_rate))
    return agents, env




def update(agents, env, regrowth_rate, rng, reproduction_enabled=False):
    """One simulation step.

    Steps
    -----
    1. Each agent moves (and harvests resources at the destination).
    2. Each agent metabolizes; dead agents are collected.
    3. Dead agents are removed from the list.
    4. Resources regrow.
    5. Reproduction is skipped while reproduction_enabled=False.

    Parameters
    ----------
    agents : list[Agent]
        Living agents (mutated in-place: dead agents are removed).
    env : Environment
        The environment.
    regrowth_rate : float
        Fraction of resource deficit recovered per step.
    rng : np.random.Generator
        Random number generator.
    reproduction_enabled : bool
        If True, enable reproduction (not yet implemented).

    Returns
    -------
    int
        Number of deaths this step.
    """
    randomized_agents = [agents[i] for i in rng.permutation(len(agents))]

    # 1. Movement + harvest
    for a in randomized_agents:
        a.move(env, agents, rng)
    for a in agents:
        a.trade_history = {'Price': [], 'Quantity Bought': [], 'Quantity Sold': [], 'Good Given': [], 'Good Received': [], 'Partner ID': []}
    for a in randomized_agents:
        for neighbor in a.get_neighbors_in_vision(agents):
            if neighbor.id > a.id:  # Avoid double-counting pairs
                a.trade(neighbor, good_i=0, good_j=1)  # Example: trade good 0 for good 1

    # 2. Metabolism — collect deaths without mutating the list mid-loop
    to_remove = []
    for a in randomized_agents:
        if a.metabolism():
            to_remove.append(a)

    # 3. Apply deaths
    for a in to_remove:
        agents.remove(a)

    # 4. Regrow resources
    env.regrow(regrowth_rate)

    # 5. Reproduction (disabled by default)
    if reproduction_enabled:
        pass  # placeholder for future implementation

    return len(to_remove)


def _make_fig(env):
    """Build a figure with one subplot per good, ready for animation."""
    cmaps = ['YlOrBr', 'BuGn', 'PuRd', 'Blues', 'Greens', 'Oranges', 'Purples']
    fig, axes = plt.subplots(1, env.n_goods, figsize=(5 * env.n_goods, 5),
                             squeeze=False)
    axes = axes[0]
    ims, scatters = [], []
    for g, ax in enumerate(axes):
        im = ax.imshow(env.landscape[g], cmap=cmaps[g % len(cmaps)],
                       vmin=0, vmax=env.max_landscape[g].max(),
                       interpolation='nearest', origin='upper')
        sc = ax.scatter([], [], s=30, c=[], cmap='plasma',
                        edgecolors='black', linewidths=0.5,
                        vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Good {g}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ims.append(im)
        scatters.append(sc)
    # Shared colorbar for agent wealth at the far right
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.04, label='Agent wealth')
    fig.subplots_adjust(right=0.8, wspace=0.25)
    return fig, axes, ims, scatters


def animate(df_log, agents_history, env_history, save_path=None):
    """Return a FuncAnimation replaying the recorded simulation history.

    Parameters
    ----------
    df_log : pd.DataFrame
        Summary DataFrame returned by run().
    agents_history : list of (positions, wealths) tuples
        As captured by run(record=True).
    env_history : list of np.ndarray
        Landscape snapshots as captured by run(record=True).
    save_path : str or None
        If given, save the animation (use .gif for pillow, .mp4 for ffmpeg).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    # Build a throw-away env shell to pass to _make_fig (only needs n_goods,
    # max_landscape, landscape shape — no live simulation state required).
    class _FakeEnv:
        pass
    fake_env = _FakeEnv()
    fake_env.n_goods = env_history[0].shape[0]
    fake_env.landscape = env_history[0].copy()
    fake_env.max_landscape = env_history[0].copy()
    for snap in env_history:
        fake_env.max_landscape = np.maximum(fake_env.max_landscape, snap)

    fig, axes, ims, scatters = _make_fig(fake_env)
    title = fig.suptitle("t = 0")

    def _update(frame):
        snap = env_history[frame]
        positions, wealths = agents_history[frame]
        vmax = float(wealths.max()) if len(wealths) > 0 else 1.0
        for g, (im, sc) in enumerate(zip(ims, scatters)):
            im.set_data(snap[g])
            if len(positions) > 0:
                sc.set_offsets(positions[:, ::-1])  # (col, row) = (x, y)
                sc.set_array(wealths)
                sc.set_clim(0, vmax)
            else:
                sc.set_offsets(np.empty((0, 2)))
                sc.set_array(np.array([]))
        title.set_text(f"t = {frame}")
        return ims + scatters + [title]

    anim = animation.FuncAnimation(fig, _update, frames=len(env_history),
                                   interval=50, blit=False)
    if save_path is not None:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        anim.save(save_path, writer=writer, fps=20)
    return anim


def live_view(env, agents, t):
    """Update (or create) a persistent figure showing the current simulation state.

    Call once per simulation step when ``live=True`` is passed to run().
    """
    if not hasattr(live_view, '_fig') or live_view._fig is None:
        live_view._fig, live_view._axes, live_view._ims, live_view._scatters = _make_fig(env)
        live_view._title = live_view._fig.suptitle(f"t = {t}")
        plt.ion()

    snap = env.landscape
    positions = np.array([a.position for a in agents]) if agents else np.empty((0, 2), dtype=int)
    wealths = np.array([a.bundle.sum() for a in agents]) if agents else np.array([])
    vmax = float(wealths.max()) if len(wealths) > 0 else 1.0

    for g, (im, sc) in enumerate(zip(live_view._ims, live_view._scatters)):
        im.set_data(snap[g])
        if len(positions) > 0:
            sc.set_offsets(positions[:, ::-1])
            sc.set_array(wealths)
            sc.set_clim(0, vmax)
        else:
            sc.set_offsets(np.empty((0, 2)))
            sc.set_array(np.array([]))

    live_view._title.set_text(f"t = {t}")
    live_view._fig.canvas.draw_idle()
    clear_output(wait=True)
    display(live_view._fig)
    plt.pause(0.01)


def run(time_steps, n_agents, env_size, resource_fns,
        max_resource=4, regrowth_rate=0.05,
        metabolic_rate_mean=1.0, preferences_mean=None,
        vision=6, bundle_mode="single_random", seed=0,
        reproduction_enabled=False, record=False, live=False):
    """Run the Sugarscape simulation and return a per-step summary DataFrame.

    Parameters
    ----------
    time_steps : int
        Number of simulation steps.
    n_agents : int
        Initial number of agents.
    env_size : tuple[int, int]
        (H, W) grid dimensions.
    resource_fns : list[callable]
        One landscape function per good.
    max_resource : float
        Peak resource value per cell.
    regrowth_rate : float
        Fraction of resource deficit recovered per step.
    metabolic_rate_mean : float
        Mean metabolic rate used when initialising agents.
    preferences_mean : list[float] or None
        Mean preference weights per good.
    vision : int
        Agent vision radius.
    bundle_mode : str
        Initial bundle mode ("single_random", "uniform", or "empty").
    seed : int or None
        RNG seed for reproducibility.
    reproduction_enabled : bool
        Whether to allow reproduction.

    Returns
    -------
    pd.DataFrame or tuple
        DataFrame (one row per time step) when ``record=False``; when
        ``record=True``, a tuple ``(df, agents_history, env_history)``.
    """
    rng = np.random.default_rng(seed)
    n_goods = len(resource_fns)

    agents, env = initialize(
        n_agents, env_size, resource_fns, max_resource,
        metabolic_rate_mean, preferences_mean, vision,
        n_goods, bundle_mode, rng
    )

    records = []
    env_history = []
    agents_history = []
    trade_records = []  # flat list of one dict per executed trade across the whole run

    # Snapshot endowments BEFORE the first step so the Edgeworth box can plot the
    # starting allocation against the contract curve.
    initial_bundles = np.array([a.bundle.copy() for a in agents])
    initial_preferences = np.array([a.preferences.copy() for a in agents])
    initial_ids = np.array([a.id for a in agents])

    for t in range(time_steps):
        deaths = update(agents, env, regrowth_rate, rng,
                        reproduction_enabled=reproduction_enabled)

        # Pull every trade executed THIS step off the agents' (just-populated)
        # trade_history dicts. Each trade is logged on both partners, so divide by 2
        # for total-trade counts when needed; here we just keep the raw rows so the
        # aggregator can group however it likes.
        n_trades_this_step = 0
        prices_this_step = []
        for a in agents:
            for k in range(len(a.trade_history['Price'])):
                trade_records.append({
                    't': t,
                    'agent_id': a.id,
                    'partner_id': a.trade_history['Partner ID'][k],
                    'price': a.trade_history['Price'][k],
                    'q_bought': a.trade_history['Quantity Bought'][k],
                    'q_sold': a.trade_history['Quantity Sold'][k],
                    'good_received': a.trade_history['Good Received'][k],
                    'good_given': a.trade_history['Good Given'][k],
                })
                # Each trade is logged twice (once per partner); count once.
                if a.id < a.trade_history['Partner ID'][k]:
                    n_trades_this_step += 1
                    prices_this_step.append(a.trade_history['Price'][k])

        if record:
            env_history.append(env.landscape.copy())
            positions = np.array([a.position for a in agents]) if agents else np.empty((0, 2), dtype=int)
            wealths = np.array([a.bundle.sum() for a in agents]) if agents else np.array([])
            agents_history.append((positions, wealths))

        if live:
            live_view(env, agents, t)

        if agents:
            utilities = np.array([a.utility_function(a.bundle) for a in agents])
            wealth = np.array([a.bundle.sum() for a in agents])
            mean_utility = float(np.mean(utilities))
            gini_wealth = gini(wealth)
            mean_bundle_per_good = [
                float(np.mean([a.bundle[g] for a in agents]))
                for g in range(n_goods)
            ]
        else:
            mean_utility = 0.0
            gini_wealth = 0.0
            mean_bundle_per_good = [0.0] * n_goods

        # Geometric mean of executed trade prices; the canonical Sugarscape price
        # statistic (Epstein & Axtell 1996, p. 105 — also the Mesa reference
        # implementation's `geometric_mean(flatten(...))`).
        if prices_this_step:
            log_p = np.log(prices_this_step)
            geo_mean_price = float(np.exp(log_p.mean()))
            log_p_sd = float(log_p.std())
        else:
            geo_mean_price = float('nan')
            log_p_sd = float('nan')

        records.append({
            't': t,
            'n_alive': len(agents),
            'deaths_this_step': deaths,
            'mean_utility': mean_utility,
            'gini_wealth': gini_wealth,
            'mean_bundle_per_good': mean_bundle_per_good,
            'n_trades': n_trades_this_step,
            'geo_mean_price': geo_mean_price,
            'log_price_sd': log_p_sd,
        })

    df = pd.DataFrame(records)
    trades_df = pd.DataFrame(trade_records)

    # Snapshot final state so the Edgeworth box helpers can compare initial vs final.
    final_bundles = np.array([a.bundle.copy() for a in agents]) if agents else np.empty((0, n_goods))
    final_ids = np.array([a.id for a in agents]) if agents else np.empty((0,), dtype=int)
    snapshots = {
        'initial_bundles': initial_bundles,
        'initial_preferences': initial_preferences,
        'initial_ids': initial_ids,
        'final_bundles': final_bundles,
        'final_ids': final_ids,
    }

    if record:
        return df, agents_history, env_history, trades_df, snapshots
    return df, trades_df, snapshots


def run_single(seed=1, 
               parameters={
                    'time_steps': 100,
                    'n_agents': 100,
                    'env_size': (50, 50),
                    'resource_fns': make_anti_correlated_2good(sigma=0.25),
                    'max_resource': 4,
                    'regrowth_rate': 0.05,
                    'metabolic_rate_mean': 1.0,
                    'preferences_mean': [0.5, 0.5],
                    'vision': 6,
                    'bundle_mode': "uniform",
                    'reproduction_enabled': False,
                    'record': True,
                    'live': False
               }
               ):

    time_steps = parameters['time_steps'] #Number of time steps (int)
    n_agents = parameters['n_agents'] #Number of agents (int)
    env_size = parameters['env_size'] #Tuple of (H, W) (int) grid dimensions
    resource_fns = parameters['resource_fns'] #Resource functions (landscape generators) (callables)
    max_resource = parameters['max_resource'] #Peak resource value per cell (float)
    regrowth_rate = parameters['regrowth_rate'] #Rate of resource regrowth (float)
    metabolic_rate_mean = parameters['metabolic_rate_mean'] #Mean metabolic rate (float)
    preferences_mean = parameters['preferences_mean'] #Mean preferences (list of floats)
    vision = parameters['vision'] #Vision range (int)
    bundle_mode = parameters['bundle_mode'] #Mode for bundle selection (string)
    reproduction_enabled = parameters['reproduction_enabled'] #Whether reproduction is enabled (bool) (not implemented)
    record = parameters['record'] #Whether to record history for animation (bool)
    live = parameters['live'] #Whether to display the simulation in real-time (bool)

    seed = seed


    df, agents_history, env_history, trades_df, snapshots = run(
        time_steps=time_steps,
        n_agents=n_agents, 
        env_size=env_size, 
        resource_fns=resource_fns,
        max_resource=max_resource, 
        regrowth_rate=regrowth_rate,
        metabolic_rate_mean=metabolic_rate_mean, 
        preferences_mean=preferences_mean,
        vision=vision, 
        bundle_mode=bundle_mode, 
        seed=seed,
        reproduction_enabled=reproduction_enabled, 
        record=record, 
        live=live
    )

    anim = animate(
        df,
        agents_history,
        env_history,
        save_path= f"output/model_run/run_seed{seed}.mp4" #f"output/model_run/run_seed{seed}.gif"   # or f"run_seed{seed}.mp4"
    )

    return df, agents_history, env_history, trades_df, snapshots, anim