from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray
Position = Tuple[int, int]


def _normalized_preferences(preferences: Sequence[float]) -> ArrayLike:
    prefs = np.asarray(preferences, dtype=float)
    prefs = np.clip(prefs, 1e-12, None)
    return prefs / prefs.sum()


def _stochastic_preferences(mean_preferences: Sequence[float], rng: np.random.Generator, sigma: float = 0.1) -> ArrayLike:
    raw = rng.normal(np.asarray(mean_preferences, dtype=float), sigma)
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum()


def gini(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0.0, None)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * x.sum()) - (n + 1) / n)


class Policy:
    def apply_harvest(self, agent: "Agent", harvested: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        return harvested, np.zeros_like(harvested)

    def redistribute(self, agents: Sequence["Agent"]) -> ArrayLike:
        if not agents:
            return np.zeros(0, dtype=float)
        return np.zeros_like(agents[0].bundle)


@dataclass
class HarvestTaxRedistributionPolicy(Policy):
    tax_rate: Sequence[float] | float
    tax_pool: Optional[ArrayLike] = None

    def _rate(self, n_goods: int) -> ArrayLike:
        if np.isscalar(self.tax_rate):
            return np.full(n_goods, float(self.tax_rate), dtype=float)
        r = np.asarray(self.tax_rate, dtype=float)
        if r.size != n_goods:
            raise ValueError("tax_rate must be scalar or have one value per good")
        return np.clip(r, 0.0, 1.0)

    def apply_harvest(self, agent: "Agent", harvested: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        rate = self._rate(harvested.size)
        tax = np.clip(harvested * rate, 0.0, harvested)
        net = harvested - tax
        if self.tax_pool is None:
            self.tax_pool = np.zeros_like(harvested)
        self.tax_pool += tax
        return net, tax

    def redistribute(self, agents: Sequence["Agent"]) -> ArrayLike:
        if self.tax_pool is None or len(agents) == 0:
            return np.zeros(0, dtype=float)
        living = [a for a in agents if a.alive]
        if not living:
            return np.zeros_like(self.tax_pool)
        transfer = self.tax_pool / len(living)
        for a in living:
            a.bundle += transfer
        paid = self.tax_pool.copy()
        self.tax_pool[:] = 0.0
        return paid


@dataclass
class Environment:
    height: int
    width: int
    resource_fns: Sequence[Callable[[ArrayLike, ArrayLike], ArrayLike]]
    max_resource: Sequence[float] | float = 10.0
    resources: ArrayLike = field(init=False)
    max_resources: ArrayLike = field(init=False)

    def __post_init__(self) -> None:
        n_goods = len(self.resource_fns)
        if n_goods < 2:
            raise ValueError(f"Number of goods must be at least 2 for Sugarscape trade, got {n_goods}")
        maxima = np.full(n_goods, float(self.max_resource), dtype=float) if np.isscalar(self.max_resource) else np.asarray(self.max_resource, dtype=float)
        if maxima.size != n_goods:
            raise ValueError("max_resource must be scalar or length equal to number of goods")

        y, x = np.meshgrid(np.linspace(0, 1, self.height), np.linspace(0, 1, self.width), indexing="ij")
        layers = []
        for fn, max_val in zip(self.resource_fns, maxima):
            z = np.asarray(fn(x, y), dtype=float)
            z = np.clip(z, 0.0, None)
            if np.isclose(z.max(), 0.0):
                layers.append(np.zeros_like(z))
            else:
                layers.append((z / z.max()) * max_val)
        self.max_resources = np.stack(layers, axis=0)
        self.resources = self.max_resources.copy()

    @property
    def n_goods(self) -> int:
        return self.resources.shape[0]

    def is_valid(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def cell_resources(self, pos: Position) -> ArrayLike:
        r, c = pos
        return self.resources[:, r, c].copy()

    def harvest(self, pos: Position) -> ArrayLike:
        r, c = pos
        h = self.resources[:, r, c].copy()
        self.resources[:, r, c] = 0.0
        return h

    def regrow(self, growth_rate: float) -> None:
        rate = float(np.clip(growth_rate, 0.0, 1.0))
        deficit = self.max_resources - self.resources
        self.resources += rate * deficit
        np.clip(self.resources, 0.0, self.max_resources, out=self.resources)

    def total_supply(self) -> ArrayLike:
        return self.resources.reshape(self.n_goods, -1).sum(axis=1)


@dataclass
class Agent:
    id: int
    preferences: ArrayLike
    bundle: ArrayLike
    vision: int
    position: Position
    metabolic_rate: float
    alive: bool = True

    def utility(self, bundle: Optional[ArrayLike] = None) -> float:
        b = self.bundle if bundle is None else bundle
        b = np.clip(np.asarray(b, dtype=float), 1e-12, None)
        p = _normalized_preferences(self.preferences)
        return float(np.prod(np.power(b, p)))

    def mrs(self, good_i: int, good_j: int) -> float:
        b = np.clip(self.bundle, 1e-12, None)
        p = _normalized_preferences(self.preferences)
        return float((p[good_i] / p[good_j]) * (b[good_j] / b[good_i]))

    def candidate_positions(self, env: Environment) -> List[Position]:
        out: List[Position] = []
        r0, c0 = self.position
        for dr in range(-self.vision, self.vision + 1):
            for dc in range(-self.vision, self.vision + 1):
                pos = (r0 + dr, c0 + dc)
                if env.is_valid(pos):
                    out.append(pos)
        return out

    def move_to_best_cell(self, env: Environment, rng: np.random.Generator) -> None:
        candidates = self.candidate_positions(env)
        scores = []
        for pos in candidates:
            score = self.utility(self.bundle + env.cell_resources(pos))
            scores.append(score)
        best = np.max(scores)
        best_positions = [p for p, s in zip(candidates, scores) if np.isclose(s, best)]
        self.position = best_positions[int(rng.integers(0, len(best_positions)))]

    def metabolize(self) -> None:
        p = _normalized_preferences(self.preferences)
        self.bundle -= self.metabolic_rate * p
        if np.any(self.bundle <= 0):
            self.alive = False


@dataclass
class StepMetrics:
    step: int
    alive_agents: int
    deaths: int
    mean_utility: float
    inequality_gini: float
    supply: ArrayLike
    demand: ArrayLike
    total_harvest: ArrayLike
    total_tax_collected: ArrayLike
    total_redistributed: ArrayLike
    trade_volume: Dict[Tuple[int, int], float]


@dataclass
class SugarscapeModel:
    """Proposal-aligned Sugarscape ABM.

    Trade controls:
    - trade_max_rounds: maximum bilateral negotiation rounds per good pair.
    - trade_initial_quantity: initial quantity attempted in each trade exchange.
    - trade_max_fraction: upper bound on fraction of a bundle tradable in one exchange.
    """

    num_agents: int
    environment_size: Tuple[int, int]
    resource_fns: Sequence[Callable[[ArrayLike, ArrayLike], ArrayLike]]
    resource_regrowth_rate: float
    vision: int
    metabolic_rate_mean: float
    preference_means: Sequence[float]
    initial_wealth_mean: float = 3.0
    preference_sigma: float = 0.1
    metabolic_sigma: float = 0.05
    trade_max_rounds: int = 15
    trade_initial_quantity: float = 1.0
    trade_max_fraction: float = 0.5
    max_resource: Sequence[float] | float = 10.0
    policy: Optional[Policy] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.env = Environment(
            height=self.environment_size[0],
            width=self.environment_size[1],
            resource_fns=self.resource_fns,
            max_resource=self.max_resource,
        )
        self.policy = self.policy or Policy()
        self.agents = self._initialize_agents()
        self.history: List[StepMetrics] = []

    @property
    def n_goods(self) -> int:
        return self.env.n_goods

    def _initialize_agents(self) -> List[Agent]:
        agents: List[Agent] = []
        pref_mean = _normalized_preferences(self.preference_means)
        for i in range(self.num_agents):
            pref = _stochastic_preferences(pref_mean, self.rng, sigma=self.preference_sigma)
            wealth = max(1e-6, float(self.rng.normal(self.initial_wealth_mean, self.initial_wealth_mean * 0.2)))
            bundle = np.maximum(1e-6, wealth * pref)
            mr = max(1e-6, float(self.rng.normal(self.metabolic_rate_mean, self.metabolic_sigma)))
            pos = (int(self.rng.integers(0, self.env.height)), int(self.rng.integers(0, self.env.width)))
            agents.append(
                Agent(
                    id=i,
                    preferences=pref,
                    bundle=bundle,
                    vision=self.vision,
                    position=pos,
                    metabolic_rate=mr,
                )
            )
        return agents

    def _alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def _neighbors(self, agent: Agent, alive: Sequence[Agent]) -> List[Agent]:
        out = []
        ar, ac = agent.position
        for other in alive:
            if other.id == agent.id:
                continue
            br, bc = other.position
            if max(abs(ar - br), abs(ac - bc)) <= agent.vision:
                out.append(other)
        return out

    def _utility_if_trade(self, agent: Agent, delta: ArrayLike) -> float:
        new_bundle = agent.bundle + delta
        if np.any(new_bundle <= 0):
            return -np.inf
        return agent.utility(new_bundle)

    def _trade_pair(self, a: Agent, b: Agent, trade_volume: Dict[Tuple[int, int], float]) -> None:
        if not (a.alive and b.alive):
            return

        for i, j in combinations(range(self.n_goods), 2):
            for _ in range(self.trade_max_rounds):
                mrs_a = a.mrs(i, j)
                mrs_b = b.mrs(i, j)
                if np.isclose(mrs_a, mrs_b):
                    break

                buyer, seller = (a, b) if mrs_a > mrs_b else (b, a)
                price = float(np.sqrt(mrs_a * mrs_b))
                if not np.isfinite(price) or price <= 0:
                    break

                dq_i = self.trade_initial_quantity
                dq_j = price * dq_i
                if seller.bundle[i] <= 1e-12 or buyer.bundle[j] <= 1e-12:
                    break

                dq_i = min(dq_i, seller.bundle[i] * self.trade_max_fraction)
                dq_j = min(dq_j, buyer.bundle[j] * self.trade_max_fraction)
                if dq_i <= 1e-12 or dq_j <= 1e-12:
                    break

                buyer_delta = np.zeros(self.n_goods)
                seller_delta = np.zeros(self.n_goods)
                buyer_delta[i] += dq_i
                buyer_delta[j] -= dq_j
                seller_delta[i] -= dq_i
                seller_delta[j] += dq_j

                if self._utility_if_trade(buyer, buyer_delta) <= buyer.utility():
                    break
                if self._utility_if_trade(seller, seller_delta) <= seller.utility():
                    break

                buyer.bundle += buyer_delta
                seller.bundle += seller_delta
                trade_volume[(i, j)] = trade_volume.get((i, j), 0.0) + dq_i

    def _trade_phase(self, alive: Sequence[Agent]) -> Dict[Tuple[int, int], float]:
        trade_volume: Dict[Tuple[int, int], float] = {}
        visited = set()
        for a in alive:
            for b in self._neighbors(a, alive):
                key = tuple(sorted((a.id, b.id)))
                if key in visited:
                    continue
                visited.add(key)
                self._trade_pair(a, b, trade_volume)
        return trade_volume

    def _demand_vector(self, alive: Sequence[Agent]) -> ArrayLike:
        if not alive:
            return np.zeros(self.n_goods, dtype=float)
        demand = np.zeros(self.n_goods, dtype=float)
        for a in alive:
            p = _normalized_preferences(a.preferences)
            demand += a.metabolic_rate * p
        return demand

    def step(self, t: int) -> StepMetrics:
        self.env.regrow(self.resource_regrowth_rate)

        alive = self._alive_agents()
        for a in alive:
            a.move_to_best_cell(self.env, self.rng)

        total_harvest = np.zeros(self.n_goods)
        total_tax_collected = np.zeros(self.n_goods)
        for a in alive:
            harvested = self.env.harvest(a.position)
            net, tax = self.policy.apply_harvest(a, harvested)
            a.bundle += net
            total_harvest += harvested
            total_tax_collected += tax

        trade_volume = self._trade_phase(alive)

        deaths = 0
        for a in alive:
            a.metabolize()
            if not a.alive:
                deaths += 1

        total_redistributed = self.policy.redistribute(self._alive_agents())
        if total_redistributed.size == 0:
            total_redistributed = np.zeros(self.n_goods)

        alive_now = self._alive_agents()
        wealth = [float(a.bundle.sum()) for a in alive_now]
        mean_utility = float(np.mean([a.utility() for a in alive_now])) if alive_now else 0.0

        metrics = StepMetrics(
            step=t,
            alive_agents=len(alive_now),
            deaths=deaths,
            mean_utility=mean_utility,
            inequality_gini=gini(wealth),
            supply=self.env.total_supply(),
            demand=self._demand_vector(alive_now),
            total_harvest=total_harvest,
            total_tax_collected=total_tax_collected,
            total_redistributed=total_redistributed,
            trade_volume=trade_volume,
        )
        self.history.append(metrics)
        return metrics

    def run(self, time_steps: int) -> List[StepMetrics]:
        for t in range(time_steps):
            if not self._alive_agents():
                break
            self.step(t)
        return self.history


def run_policy_experiment(
    time_steps: int,
    model_kwargs: Dict,
    intervention_policy: Policy,
) -> Dict[str, float]:
    """Run baseline vs intervention models and return aggregate comparison metrics.

    Args:
        time_steps: Number of simulation steps for each run.
        model_kwargs: Keyword arguments forwarded to SugarscapeModel.
        intervention_policy: Policy instance used for the intervention run.

    Returns:
        Dictionary with average utility, inequality, and death-rate comparisons.
    """
    baseline_model = SugarscapeModel(policy=Policy(), **model_kwargs)
    baseline_hist = baseline_model.run(time_steps)

    policy_model = SugarscapeModel(policy=intervention_policy, **model_kwargs)
    policy_hist = policy_model.run(time_steps)

    def _mean(hist: List[StepMetrics], attr: str) -> float:
        if not hist:
            return 0.0
        return float(np.mean([getattr(h, attr) for h in hist]))

    return {
        "baseline_mean_utility": _mean(baseline_hist, "mean_utility"),
        "policy_mean_utility": _mean(policy_hist, "mean_utility"),
        "utility_gain": _mean(policy_hist, "mean_utility") - _mean(baseline_hist, "mean_utility"),
        "baseline_mean_gini": _mean(baseline_hist, "inequality_gini"),
        "policy_mean_gini": _mean(policy_hist, "inequality_gini"),
        "gini_change": _mean(policy_hist, "inequality_gini") - _mean(baseline_hist, "inequality_gini"),
        "baseline_death_rate": _mean(baseline_hist, "deaths"),
        "policy_death_rate": _mean(policy_hist, "deaths"),
        "death_rate_change": _mean(policy_hist, "deaths") - _mean(baseline_hist, "deaths"),
    }


if __name__ == "__main__":
    def sugar_peak_fn(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.exp(-((x - 0.2) ** 2 + (y - 0.2) ** 2) / 0.01) + np.exp(-((x - 0.8) ** 2 + (y - 0.8) ** 2) / 0.02)

    def spice_peak_fn(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.exp(-((x - 0.8) ** 2 + (y - 0.2) ** 2) / 0.01) + np.exp(-((x - 0.2) ** 2 + (y - 0.8) ** 2) / 0.02)

    model = SugarscapeModel(
        num_agents=100,
        environment_size=(60, 60),
        resource_fns=[sugar_peak_fn, spice_peak_fn],
        max_resource=[12.0, 12.0],
        resource_regrowth_rate=0.05,
        vision=2,
        metabolic_rate_mean=0.2,
        preference_means=[0.5, 0.5],
        policy=HarvestTaxRedistributionPolicy(tax_rate=0.1),
        seed=7,
    )
    history = model.run(time_steps=100)
    if history:
        last = history[-1]
        print(
            {
                "steps_executed": len(history),
                "alive_agents": last.alive_agents,
                "mean_utility": round(last.mean_utility, 4),
                "gini": round(last.inequality_gini, 4),
                "supply": np.round(last.supply, 2).tolist(),
                "demand": np.round(last.demand, 2).tolist(),
            }
        )
