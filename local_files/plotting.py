import xarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.animation as animation
from typing import Optional, Dict, Tuple
import math
import datetime
from pathlib import Path

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

def save_animation(
    data: Dict[str, Tuple[xarray.Dataset, matplotlib.colors.Normalize, str]],
    fig_title: str,
    output_path: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    fps: int = 4
) -> None:
    """
    Creates and saves an animation of the data to a file.
    
    Args:
        data: Dictionary of data to plot
        fig_title: Title for the figure
        output_path: Path to save the animation (supports .mp4, .gif)
        plot_size: Size multiplier for the plot
        robust: Whether to use robust scaling
        cols: Number of columns in the grid
        fps: Frames per second for the animation
    """
    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)

    def update(frame):
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))
        return images

    ani = animation.FuncAnimation(
        fig=figure, func=update, frames=max_steps, interval=1000//fps)
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save animation based on file extension
    extension = Path(output_path).suffix.lower()
    if extension == '.mp4':
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer)
    elif extension == '.gif':
        ani.save(output_path, writer='pillow', fps=fps)
    else:
        raise ValueError(f"Unsupported file extension: {extension}. Use .mp4 or .gif")
    
    plt.close(figure)

def save_static_plot(
    data: Dict[str, Tuple[xarray.Dataset, matplotlib.colors.Normalize, str]],
    fig_title: str,
    output_path: str,
    time_index: int = 0,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    dpi: int = 100
) -> None:
    """
    Creates and saves a static plot of the data at a specific time index.
    
    Args:
        data: Dictionary of data to plot
        fig_title: Title for the figure
        output_path: Path to save the plot (supports any format matplotlib supports)
        time_index: Time index to plot
        plot_size: Size multiplier for the plot
        robust: Whether to use robust scaling
        cols: Number of columns in the grid
        dpi: DPI for the output image
    """
    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,
                                plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=time_index, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)