from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from plotly.graph_objects import Contour, Figure, Scatter
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots

axeskwargs = {
    "mirror": "allticks",
    "ticks": "inside",
    "showgrid": True,
    "showline": True,
    "showticklabels": True,
    "title_standoff": 2,
}


@dataclass
class DeterministicAnalisisFigure:

    z: NDArray[float64]

    def __post_init__(self) -> None:

        # fmt: off
        self.fig: Figure = make_subplots(
            rows=2,
            cols=4,
            specs=[
                [{"rowspan": 2}, {"rowspan": 2}, {"rowspan": 2}, {"colspan": 1}],
                [          None,           None,           None, {"colspan": 1}],
            ],
            horizontal_spacing=0.01,
            vertical_spacing=0.07,
        )
        # fmt: on

    def update_stress_axes(self, max_depth):
        for i in [1, 2, 3]:
            self.fig.update_xaxes(
                row=1,
                col=i,
                matches="x",
                title_text=f"Principal stresses and pore pressure [MPa]",
                **axeskwargs,
            )
            self.fig.update_yaxes(
                row=1,
                col=i,
                matches="y",
                range=[max_depth, 0],
                **(axeskwargs | {"showticklabels": False}),
            )
        self.fig.update_yaxes(
            row=1,
            col=1,
            matches="y",
            title_text="Depth [m]",
            showticklabels=True,
            title_standoff=2,
        )

    def update_fault_stress_axes(self, range_max: float):
        self.fig.update_xaxes(
            row=1,
            col=2,
            matches="x2",
            title_text="Stresses on fault [MPa]",
            showticklabels=True,
            range=[0, range_max],
        )

    def update_FS_axes(self, range_max: float):
        self.fig.update_xaxes(
            row=1,
            col=3,
            matches="x3",
            title_text="Security Factor (SF)",
            range=[0, range_max],
            **axeskwargs,
        )

    def update_mohr_coulomb_axes(self, Smax: float, Tmax: float):
        self.Smax = Smax
        self.Tmax = Tmax
        self.fig.update_xaxes(
            row=1,
            col=4,
            range=[0, Smax],
            title_text=f"σ<sub>n</sub> - Normal stress [MPa]",
            **axeskwargs,
        )
        self.fig.update_yaxes(
            row=1,
            col=4,
            range=[0, Tmax],
            title_text=f"𝜏<sub>n</sub> - Shear stress [MPa]",
            side="right",
            **axeskwargs,
        )

    def update_contour_axes(self, dPmin, dPmax, zmin, zmax):
        self.fig.update_xaxes(
            row=2,
            col=4,
            title_text=f"Pressure variation - ΔP [MPa]",
            range=[dPmin, dPmax],
            **axeskwargs,
        )
        self.fig.update_yaxes(
            row=2,
            col=4,
            title_text=f"Injection layer depth [m]",
            side="right",
            range=[zmin, zmax],
            **axeskwargs,
        )

    def plot(
        self,
        row: int,
        col: int,
        x: NDArray[float64] | list[float],
        y: NDArray[float64] | list[float],
        name: str,
        color: str = "black",
        width: int = 1,
        showlegend: bool = True,
        **kwargs,
    ):
        """Adds a Scatter plot to `fig`"""
        self.fig.add_trace(
            row=row,
            col=col,
            trace=Scatter(
                x=x,
                y=y,
                name=name,
                line=Line(color=color, width=width),
                showlegend=showlegend,
                **kwargs,
            ),
        )

    def add_stress_curve(self, curve: NDArray[float64], color: str, name: str):
        self.plot(1, 1, curve, self.z, name, color)

    def add_fault_stress_curve(self, curve: NDArray[float64], color: str, name: str):
        self.plot(1, 2, curve, self.z, name, color)

    def add_FS_curve(self, curve, color, name):
        self.plot(1, 3, curve, self.z, name, color, showlegend=False)

    def add_FS_hline(self, value, **kwargs):
        self.fig.add_vline(value, 1, 3, line_color="black", line_width=2, **kwargs)  # type: ignore

    def add_hlines_stress_curve(self, values: Any, **kwargs):
        for v in values:
            for i in [1, 2, 3]:
                self.fig.add_hline(v, 1, i, line_color="black", line_width=2, **kwargs)  # type: ignore

    def plot_mohr_envelope(self, coh: float):
        self.plot(1, 4, [0, self.Smax], [coh, self.Tmax], "Envelope", "red", 1, False)

    def plot_mohr_circle(
        self,
        sigma1: float,
        sigma3: float,
        angle: float,
        color: str = "green",
        label: str = "Mohr Circle",
    ):
        """Plot the upper half of Mohr's circle in `fig[1,4]` with shear plane point"""
        center = (sigma1 + sigma3) / 2
        radius = (sigma1 - sigma3) / 2

        theta = np.linspace(0, np.pi, 100)
        x = center + radius * np.cos(theta)
        y = radius * np.sin(theta)

        self.plot(1, 4, x, y, name=label, color=color, showlegend=False)

        theta_rad = np.radians(2 * angle)
        sigma_n = center - radius * np.cos(theta_rad)
        tau = radius * np.sin(theta_rad)
        self.plot(
            1,
            4,
            np.array([center, sigma_n]),
            np.array([0, tau]),
            name="Stress on fault",
            showlegend=True,
            line_dash="dash",
        )

    def add_contours(self, x, y, z):
        contour = Contour(
            x=x,
            y=y,
            z=z,
            colorscale="Turbo",
            ncontours=20,
            contours=dict(
                coloring="fill",
                showlines=False,
            ),
            colorbar=dict(
                x=1.05,
                y=0.23,
                len=0.5,
                title=dict(text="Safety Factor (SF)", side="right"),
            ),
            name="SF map",
            showlegend=False,
        )
        contour_line = Contour(
            x=x,
            y=y,
            z=z,
            contours=dict(
                start=1.0,
                end=1.0,
                size=0.001,
                coloring="none",
                showlabels=True,
                labelformat=".1f",
                labelfont=dict(size=16, color="black"),
            ),
            line=dict(color="black", width=2),
            showscale=False,
            coloraxis=None,
            colorscale=None,
            name="SF map",
            showlegend=False,
        )
        X, Y = np.meshgrid(x, y[0 : -1 : int(len(y) / 50)])
        mesh = Scatter(
            x=X.flatten(),
            y=Y.flatten(),
            mode="markers",
            marker=dict(size=5, color="rgba(0,0,0,0)"),
            visible=True,
            showlegend=False,
            name="ΔP vs z",
        )
        self.fig.add_trace(contour, row=2, col=4)
        self.fig.add_trace(contour_line, row=2, col=4)
        self.fig.add_trace(mesh, row=2, col=4)

    def update_layout(self):
        self.fig.update_layout(
            template="simple_white",
            plot_bgcolor="white",
            height=640,
            margin=dict(l=50, r=40, t=30, b=0),
            showlegend=True,
            dragmode="zoom",
        )

    def update_current_point(self, x, z):
        self.fig.add_hline(z, 1, 1, line_color="black", line_width=2, line_dash="dash")  # type: ignore
        self.fig.add_hline(z, 1, 2, line_color="black", line_width=2, line_dash="dash")  # type: ignore
        self.fig.add_hline(z, 1, 3, line_color="black", line_width=2, line_dash="dash")  # type: ignore
        self.fig.add_hline(z, 2, 4, line_color="black", line_width=2)  # type: ignore
        self.fig.add_vline(x, 2, 4, line_color="black", line_width=2)  # type: ignore
        self.plot(2, 4, [x], [z], name="Current", showlegend=False)
