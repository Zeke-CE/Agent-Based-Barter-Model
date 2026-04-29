import numpy as np


def make_landscape(H, W, resource_fns, max_resource):
    """
    Build a (n_goods, H, W) landscape array from a list of 2-D functions
    Handles multiple goods by stacking the resulting 2-D arrays along a new first dimension.

    Parameters:
    -- H : Height of grid.
    -- W : Width of grid.
    -- resource_fns : list[functions] = Each fn maps (x, y) arrays in [0,1]^2 to a non-negative scalar array.
    -- max_resource : float = Maximum resource value for scaling the landscape

    Returns:
    --landscape: np.ndarray with shape (n_goods, H, W).

    """

    #number of goods is number of functions
    n_goods = len(resource_fns)

    #Generate normalized grid of (x, y) coordinates between 0 and 1, with shape (H, W)
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')

    #initialize landscape array
    landscape = np.zeros((n_goods, H, W))

    # loop through list of functions for each good
    for g, fn in enumerate(resource_fns):

        #Get z grid (array) for good g
        z = np.clip(fn(x, y), 0, None)

        #scale z to max_resource => divide z by its max (normalize to [0, 1]) and then multiply by max_resource (scale to [0, max_resource])
        landscape[g] = (z / z.max() * max_resource) if z.max() > 0 else z

    return landscape

def make_anti_correlated_2good(H=50, W=50, sigma=0.25, max_resource=4):
    """
    Standard 2-good Sugarscape: sugar peaks UL+LR, spice peaks UR+LL (refactored from axtell/mesa implementation)

    Parameters:
    -- H : int
        Grid height.
    -- W : int
        Grid width.
    -- sigma : float
        Width of each Gaussian peak.
    -- max_resource : float
        Peak resource level (passed to make_landscape).

    Returns:
    -- list[sugar_fn, spice_fn] : list of 2 functions mapping (x, y) to resource levels for sugar and spice, respectively
    """

    # Gaussian function centered at (cx, cy) with width sigma for creating a good peak
    g = lambda x, y, cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    #Set location of sugar peaks at upper left (0.2, 0.2) and lower right (0.8, 0.8), and spice peaks at upper right (0.2, 0.8) and lower left (0.8, 0.2)
    sugar_fn = lambda x, y: g(x, y, 0.2, 0.2) + g(x, y, 0.8, 0.8)

    #Set location of spice peaks at upper right (0.2, 0.8) and lower left (0.8, 0.2)
    spice_fn = lambda x, y: g(x, y, 0.2, 0.8) + g(x, y, 0.8, 0.2)

    return [sugar_fn, spice_fn]

class Environment:

    def __init__(self, H, W, resource_fns, max_resource=4):
        '''
        Envirnoment class for grid-based resource landscape

        Parameters:
        -- H : int
            Grid height.
        -- W : int
            Grid width.
        -- resource_fns : list[functions]
            List of functions mapping (x, y) to resource levels for each good.
        -- max_resource : float
            Maximum resource value for scaling the landscape.

        '''

        #number of goods
        self.n_goods = len(resource_fns)

        #Height and width of grid
        self.H = H
        self.W = W

        #Maximum (before harvesting) shape (n_goods, H, W) landscape array
        self.max_landscape = make_landscape(H, W, resource_fns, max_resource)

        #Initial landscape starts at max (before harvesting)
        self.landscape = self.max_landscape.copy()

    def is_valid_position(self, pos):
        """
        Check if position is valid (within grid bounds)

        Parameters:
        -- pos : tuple[int, int]
            (row, col) to check

        Returns:
        --bool

        """

        #Position is a tuple of (x, y) coordinates
        x, y = pos

        #Check if x and y are within bounds of grid (0, 0) to (H-1, W-1)
        return 0 <= x < self.H and 0 <= y < self.W
    
    def get_resources_at(self, pos):
        """Return length-n_goods vector of current resources at pos.

        Parameters:
        -- pos : tuple[int, int]
            (row, col).

        Returns
        -------
        -- np.ndarray
            Shape (n_goods,).
        """
        x, y = pos

        return self.landscape[:, x, y].copy()
    
    def regrow(self, growth_rate):
        """Logistic-style regrowth toward max_landscape, vectorised over all goods.

        Parameters
        ----------
        growth_rate : float
            Fraction of deficit to recover each step.
        """

        deficit = self.max_landscape - self.landscape

        self.landscape = np.clip(self.landscape + growth_rate * deficit, 0, self.max_landscape)

        