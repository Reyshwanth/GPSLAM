import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib import colors

def plot_random_points(n_points=10, size=50, seed=None, save_path=None):
    """Generate n_points random points in a size x size square and plot them.

    Args:
        n_points (int): number of random points to generate.
        size (int or float): size of the square domain (0..size in x and y).
        seed (int|None): random seed for reproducibility.
        save_path (str|None): path to save the figure (if provided).
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, size, size=n_points)
    ys = rng.uniform(0, size, size=n_points)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, c='red', s=50, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        ax.text(x + size*0.01, y + size*0.01, str(i), fontsize=9)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_title(f"{n_points} random points in {size}x{size} space")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.4)

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_costmap(n_points=10, size=50, resolution=0.1, radius=5.0, radius_c=10, seed=None,
                 cmap='viridis', save_path=None, show_cell_grid=True,
                 highlight_value=2.0, highlight_cmap='RdYlBu_r', highlight_contour=True):
    """Generate a costmap for `n_points` random points inside a size x size square.

    Each point contributes a binary score: 1.0 if the cell is within `radius`
    meters of the point, otherwise 0.0. The map value at each cell is the sum
    of binary contributions from all points (i.e. the count of points within
    `radius` of that cell).

    Args:
        n_points (int): number of random points.
        size (float): domain size.
        resolution (float): grid resolution in meters.
        radius (float): radius at which contribution falls to zero.
        seed (int|None): RNG seed for reproducibility.
        cmap (str): matplotlib colormap name.
        save_path (str|None): optional path to save the figure.
    """
    rng = np.random.default_rng(seed)
    pts_x = rng.uniform(0, size, size=n_points)
    pts_y = rng.uniform(0, size, size=n_points)

    xs = np.arange(0, size + resolution/2, resolution)
    ys = np.arange(0, size + resolution/2, resolution)
    X, Y = np.meshgrid(xs, ys)

    # Initialize cost map (cumulative)
    cost = np.zeros_like(X, dtype=float)

    for px, py in zip(pts_x, pts_y):
        d = np.hypot(X - px, Y - py)
        # binary contribution: 1 inside radius, 0 outside
        contrib = (d <= float(radius)).astype(float)
        # accumulate contributions from all points (counts)
        cost += contrib


    cost_modified = np.copy(cost)
    # Second figure: show contours at level 3 (the "3 contour" the user added)


        # draw contour(s) at level 3 if present in the data range
    level3 = 3
    lower_bound=0
        #breakpoint()
    while(not(np.all(cost_modified >= 3))):


        if lower_bound!=0:
            # Create a binary mask for the level3 region (cost >= level3)
            mask = (cost_modified >= level3)
            # Compute dilation radius in pixels using the plotting resolution
            # resolution is meters per pixel in the current cost grid
            res = float(resolution)
            # avoid zero division
            if res <= 0:
                res = 0.1
            rad_px = max(1, int(np.ceil(float(radius_c) / res)))

            # Build a circular structuring element (disk) as footprint
            r = np.arange(-rad_px, rad_px + 1)
            xx, yy = np.meshgrid(r, r)
            footprint = (xx**2 + yy**2) <= (rad_px ** 2)
            #breakpoint()
            # Perform binary dilation (Minkowski sum) to pad the mask
            dilated = ndimage.binary_dilation(mask, structure=footprint)
            # Create a rectangular mask with padding of radius units
            rect_mask = np.ones_like(mask)
            ny, nx = mask.shape
            pad = rad_px  # use same radius in pixels as before
            if nx > 2*pad and ny > 2*pad:  # only if there's room for the rectangle
                rect_mask[pad:-pad, pad:-pad] = 0  # create hollow rectangle
                # Dilate the rectangular mask

            # Create a modified costmap by adding 1 to the existing cost where dilated region is True
            # This preserves existing counts and increments them by 1 inside the dilated area
            cost_modified = cost_modified + dilated.astype(float)+rect_mask.astype(float)
            #breakpoint()
            # Replace the image on the second figure to show the modified costmap

        fig, ax = plt.subplots(figsize=(7, 6))

        # Build a discrete colormap for categories 0..4 and mask cells >=5
        from matplotlib.colors import ListedColormap, BoundaryNorm
        # colors for 0..4: 0=lightgray,1=blue,2=green,3=orange,4=red
        discrete_colors = ['#ffddd2', '#ff8a75','#e61e0f', '#67000d']
        #discrete_colors = ['#d9d9d9', '#2b83ba', '#7fbf7b', '#fdae61', '#d7191c']
        cmap_discrete = ListedColormap(discrete_colors)

        # mask out >=5 for the discrete plot
        mask_ge5 = (cost_modified >= 4)
        disp_disc = np.ma.masked_where(mask_ge5, cost_modified)

        # boundaries for categories: [-0.5,0.5,1.5,2.5,3.5,4.5]
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm_disc = BoundaryNorm(bounds, cmap_discrete.N)

        im = ax.imshow(disp_disc, origin='lower', extent=[0, size, 0, size], cmap=cmap_discrete, norm=norm_disc, interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['0', '1', '2', '3'])
        cbar.set_label('cumulative count (binned)')

        # If there are values >=5, overlay a purple->black gradient for those cells
        if cost_modified.max() >= 4:
            # gradient colormap from purple to black
            gradient_cmap = colors.LinearSegmentedColormap.from_list('purple_black', ['#6a3d9a', '#000000'])
            disp_grad = np.ma.masked_where(~mask_ge5, cost_modified)
            im_grad = ax.imshow(disp_grad, origin='lower', extent=[0, size, 0, size], cmap=gradient_cmap,
                                vmin=4, vmax=float(cost_modified.max()), interpolation='nearest', zorder=2)
            # colorbar for gradient region: only integer ticks (5,6,...,max)
            max_int = int(np.ceil(float(cost_modified.max())))
            if max_int < 4:
                max_int = 4
            ticks = np.arange(4, max_int + 1)
            # increase pad so the gradient colorbar is spaced away from the discrete one
            cbar_grad = fig.colorbar(im_grad, ax=ax, fraction=0.03, pad=0.16, ticks=ticks)
            cbar_grad.ax.set_yticklabels([str(int(t)) for t in ticks])
            # move the label away from the ticks for clarity
            cbar_grad.set_label('>=4 localizers', labelpad=10)

        # draw a cell grid with cell size = radius/2 if requested
        if show_cell_grid:
            cell_size = float(radius) / 2.0
            # vertical lines
            xs_lines = np.arange(0, size + 1e-9, cell_size)
            for xv in xs_lines:
                ax.axvline(xv, color='k', linewidth=0.6, alpha=0.25, zorder=4)
            # horizontal lines
            ys_lines = np.arange(0, size + 1e-9, cell_size)
            for yv in ys_lines:
                ax.axhline(yv, color='k', linewidth=0.6, alpha=0.25, zorder=4)

        # Optionally draw a contour line at the highlight_value to emphasize it
        if highlight_contour and (highlight_value is not None):
            hv = float(highlight_value)
            # only draw contour if hv is within observed range
            if hv <= cost_modified.max() and hv > 0:
                try:
                    # use the same grid coordinates used for imshow
                    CS = ax.contour(xs, ys, cost_modified, levels=[hv], colors=['black'], linewidths=1.4, linestyles='-')
                    # draw a white thinner contour on top for contrast
                    ax.contour(xs, ys, cost_modified, levels=[hv], colors=['white'], linewidths=0.8, linestyles='--')
                    # Add filled contour in green with some transparency
                    #ax.contourf(xs, ys, cost_modified, levels=[hv, cost_modified.max()], colors=['green'], alpha=0.5)
                except Exception:
                    pass

        # Overlay points
        ax.scatter(pts_x, pts_y, c='white', edgecolor='black', s=40, zorder=3)
        for i, (x, y) in enumerate(zip(pts_x, pts_y), start=1):
            ax.text(x + size*0.005, y + size*0.005, str(i), color='white', fontsize=8, zorder=4)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_title(f"Range Map({n_points} points); rs={radius} m; rc={radius_c} m")
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(False)
        lower_bound+=1



        # Draw the dilated boundary for emphasis (contour at 0.5 between 0 and 1)

        #ax2.contour(xs, ys, dilated.astype(int), levels=[0.5], colors=['darkgreen'], linewidths=1.2)


        # Optionally draw the original level3 contour on top

        fig.savefig(f"plot_iteration_{lower_bound}.png", dpi=1000) # Save the plot with higher resolution
        plt.close() # Close the figure to free up memory


            
def compute_costmap(n_points=10, size=50, resolution=1.0, radius=5.0, seed=None):
    """Compute and return a binary cumulative costmap (counts) at given resolution.

    Returns a dict with keys: 'xs', 'ys', 'cost', 'pts_x', 'pts_y'
    """
    rng = np.random.default_rng(seed)
    pts_x = rng.uniform(0, size, size=n_points)
    pts_y = rng.uniform(0, size, size=n_points)

    xs = np.arange(0, size + resolution/2, resolution)
    ys = np.arange(0, size + resolution/2, resolution)
    X, Y = np.meshgrid(xs, ys)
    cost = np.zeros_like(X, dtype=float)

    for px, py in zip(pts_x, pts_y):
        d = np.hypot(X - px, Y - py)
        contrib = (d <= float(radius)).astype(float)
        cost += contrib

    return {'xs': xs, 'ys': ys, 'cost': cost, 'pts_x': pts_x, 'pts_y': pts_y}


def save_costmap_npz(file_path, costmap_dict):
    """Save costmap dict to a .npz file for later loading.

    file_path: str path to .npz file
    costmap_dict: dict returned by compute_costmap
    """
    np.savez(file_path, xs=costmap_dict['xs'], ys=costmap_dict['ys'], cost=costmap_dict['cost'],
             pts_x=costmap_dict['pts_x'], pts_y=costmap_dict['pts_y'])


if __name__ == '__main__':
    # Quick demo: 10 random points in a 50x50 space
    # no seed provided -> different points each run; set seed=int for reproducibility
    # Demo: plot a costmap for 10 random points in a 50x50 space
    # (set seed=int for reproducibility, or seed=None for different map each run)
    # also compute and save a coarser costmap (1m resolution) for use as a background
    cm = compute_costmap(n_points=100, size=50, resolution=0.1, radius=0.5, seed=None)
    save_costmap_npz('costmap.npz', cm)
    plot_costmap(n_points=100, size=50, resolution=0.1, radius=0.5, radius_c=5.0,seed=1, cmap='viridis')
    # If you prefer the scatter-only plot, call:
    # plot_random_points(n_points=10, size=50, seed=None)
