import numpy as np



class Agent:

    '''
    
    Create agents living in the Sugarscape

    Parameters:
    -- id: integer identifier
    -- preferences: list of length n_goods where each entry is the alpha in the cobb=douglas utility funciton
    -- bundle: array of length n_goods where each entry is the amount of that good and agent has
    -- vision: How far an agent can see (x,y)
    -- metabolism: array with length n_goods where each entry is the amount of that good the agent consumes each time step

    Dynamics:
    -- utility_fucntion: calculates utility based on current bundle, preferences, and cobb-douglas function
    -- move: agent moves to the location in vision that maximizes utility and consumes the reasources there
    -- metabolism: subtract metabolism from bundle, if they have less than 0 of any good, they die
    -- trade: implementation of the canonical trade algorithm used by axtell and mesa


    '''

    def __init__(self, id, preferences, bundle, vision, position, metabolic_rate):
        #Identifier
        self.id = id

        #Preferences and bundle arrays
        self.preferences = np.asarray(preferences, dtype=float)
        self.bundle = np.asarray(bundle, dtype=float)

        #Vision integer and position tuple (x,y)
        self.vision = vision
        self.position = position

        #Metabolic rate array
        self.metabolic_rate = np.asarray(metabolic_rate, dtype=float)

        #Trade history dictionary (keeps track of this time step trades)
        self.trade_history = {'Price': [], 'Quantity Bought': [], 'Quantity Sold': [], 'Good Given': [], 'Good Received': [], 'Partner ID': []}

    def utility_function(self, bundle):
        """

        Cobb-Douglas welfare: U = prod(bundle[g] ** (alpha[g] / sum(alpha)))
        -- for two goods, utility = good_0^alpha * good_1^(1-alpha)

        Parameters:
        -- Bundle: array length n_goods where index is good and entry is amount of good 

        Returns:
        -- utility: float

        """
        
        if np.any(bundle < 0):
            raise ValueError(f"Bundle contains negative amount of good: {bundle}")
        
        if np.any(bundle == 0):
            return 0.0 #Since Cobb-Douglas is multiplicative, any zero good is zero utility (maybe fix?)

        alpha = self.preferences / self.preferences.sum()

        utility = float(np.prod(np.power(bundle, alpha)))

        return utility
    
    def _calc_mrs(self, good_i, good_j):
            """Marginal rate of substitution: units of good j per unit of good i.

            Parameters:
            -- good_i: good taking mrs of = index of good
            -- good_j : units of mrs = index of good

            Returns:
            -- mrs: float, units of good j per unit of good i = how much of good j you would give up for good i
            """

            #Exponent for good i and j
            weight_good_i = self.preferences[good_i]/self.preferences.sum()
            weight_good_j = self.preferences[good_j]/self.preferences.sum()

            #Quantity of good i and j in bundle
            quantity_good_i = self.bundle[good_i]
            quantity_good_j = self.bundle[good_j]

            if quantity_good_i <= 0:
                return np.inf  #You would be willing to give up an infinite amount of good j for one unit of good i
            
            if quantity_good_j <= 0:
                return 0.0  #You wouldn't give any j for i
            
            #Return mrs        
            return (quantity_good_j / weight_good_j) / (quantity_good_i / weight_good_i)
    

    def trade(self, other_agent, good_i, good_j):
        """
        Attempt to trade with another agent
        -- Trades until MRS is equalized or no mutually beneficial trade is possible
        -- Might be interesting to implement and less rational version


        Parameters:
        -- other_agent: another Agent object to attempt to trade with
        -- good_i: index of good i (the good being bought by the buyer)
        -- good_j: index of good j (the good being sold by the seller)
        
        Returns:
        -- None, but updates bundles of self and other_agent if trade is successful
        
        """
        while True:
            mrs_self = self._calc_mrs(good_i, good_j)
            mrs_other = other_agent._calc_mrs(good_i, good_j)
            #Higher mrs means more of good i per unit of good j, so self wants to give up good i and get good j, while other wants the opposite.  If they're equal, no mutually beneficial trade is possible.

            if abs(mrs_self - mrs_other) < 1e-6:
                break

            price = np.sqrt(mrs_self * mrs_other)

            q_buy, q_sell = 0, 0

            if price >= 1:
                q_buy = 1
                q_sell = price
            else:
                q_buy = 1 / price
                q_sell = 1
            



            if mrs_self > mrs_other:
                buyer = self
                seller = other_agent
            else:
                buyer = other_agent
                seller = self

            if seller.bundle[good_i] < q_buy or buyer.bundle[good_j] < q_sell:
                return
            
            # Compute prospective bundles using indexed mutation (was using a 2-element
            # list which only works for n_goods=2; this generalises and makes the seller
            # lose good_i / gain good_j and the buyer the reverse).
            new_seller_bundle = seller.bundle.copy()
            new_seller_bundle[good_i] -= q_buy
            new_seller_bundle[good_j] += q_sell
            new_buyer_bundle  = buyer.bundle.copy()
            new_buyer_bundle[good_i]  += q_buy
            new_buyer_bundle[good_j]  -= q_sell

            if buyer.utility_function(new_buyer_bundle) <= buyer.utility_function(buyer.bundle) or seller.utility_function(new_seller_bundle) <= seller.utility_function(seller.bundle):
                return

            # MRS-non-crossing check: must be evaluated on PROSPECTIVE bundles, not
            # the current ones. Compute MRS directly from new_*_bundle to avoid
            # mutating state before the check passes.
            def _mrs(b, prefs, gi, gj):
                if b[gi] <= 0: return np.inf
                if b[gj] <= 0: return 0.0
                return (b[gj] / prefs[gj]) / (b[gi] / prefs[gi])
            new_mrs_buyer  = _mrs(new_buyer_bundle,  buyer.preferences,  good_i, good_j)
            new_mrs_seller = _mrs(new_seller_bundle, seller.preferences, good_i, good_j)
            if new_mrs_buyer <= new_mrs_seller:
                return

            seller.bundle = new_seller_bundle
            buyer.bundle = new_buyer_bundle

            # Log the trade on BOTH agents so per-agent history is symmetric and the
            # market-level aggregator can sweep one list of trades per step.
            for a, partner, q_in, q_out, gi_in, gj_out in (
                (buyer,  seller, q_buy,  q_sell, good_i, good_j),
                (seller, buyer,  q_sell, q_buy,  good_j, good_i),
            ):
                a.trade_history['Price'].append(price)
                a.trade_history['Quantity Bought'].append(q_in)
                a.trade_history['Quantity Sold'].append(q_out)
                a.trade_history['Good Given'].append(gj_out)
                a.trade_history['Good Received'].append(gi_in)
                a.trade_history['Partner ID'].append(partner.id)


        return
    
    #refine
    def move(self, env, agents, rng):
        """Move to the best reachable cell within vision and harvest its resources.

        Best is defined as highest prospective welfare = utility_function(self.bundle
        + resources_at_cell).  Occupied cells are skipped.  Ties broken first by
        Manhattan distance (closest wins), then randomly.

        Parameters
        ----------
        env : Environment
            The environment object.
        agents : list[Agent]
            All living agents (used to detect occupied cells).
        rng : np.random.Generator
            Random number generator for tie-breaking.
        """
        x, y = self.position
        v = self.vision

        # Collect positions of other agents for collision detection
        occupied = {a.position for a in agents if a is not self}

        # Build candidate cell grid (clipped to env bounds)
        x_min = max(0, x - v)
        x_max = min(env.H - 1, x + v)
        y_min = max(0, y - v)
        y_max = min(env.W - 1, y + v)

        best_utility = -np.inf
        best_distance = np.inf
        best_cells = []

        for dx in range(x_min, x_max + 1):
            for dy in range(y_min, y_max + 1):
                if (dx, dy) in occupied:
                    continue
                prospective = self.bundle + env.landscape[:, dx, dy]
                u = self.utility_function(prospective)
                dist = abs(dx - x) + abs(dy - y)
                if u > best_utility or (u == best_utility and dist < best_distance):
                    best_utility = u
                    best_distance = dist
                    best_cells = [(dx, dy)]
                elif u == best_utility and dist == best_distance:
                    best_cells.append((dx, dy))

        if best_cells:
            idx = rng.integers(len(best_cells))
            new_pos = best_cells[idx]
        else:
            new_pos = self.position

        self.position = new_pos

        # Harvest entire stack at new position
        nx, ny = new_pos
        self.bundle += env.landscape[:, nx, ny]
        env.landscape[:, nx, ny] = 0.0

    #Rework
    def metabolism(self):
        """Subtract per-good metabolic rate; clip at zero so bundles never go
        negative (negative bundles produce NaN under Cobb-Douglas). Death is
        triggered when any good would have gone non-positive THIS step.
        """
        will_die = bool(np.any(self.bundle <= self.metabolic_rate))
        self.bundle = np.maximum(self.bundle - self.metabolic_rate, 0.0)
        return will_die

    #Rework
    def get_neighbors_in_vision(self, agents):
        """Return list of other agents within Chebyshev distance <= self.vision.

        Parameters
        ----------
        agents : list[Agent]
            All living agents.

        Returns
        -------
        list[Agent]
            Agents (excluding self) within vision radius.
        """
        if not agents:
            return []
        positions = np.array([a.position for a in agents])
        self_pos = np.array(self.position)
        chebyshev = np.max(np.abs(positions - self_pos), axis=1)
        in_range = (chebyshev <= self.vision) & (chebyshev > 0)
        return [a for a, ok in zip(agents, in_range) if ok]   