from __future__ import annotations

import os
from typing import Any, Literal, cast, overload

import numpy as np
import streamlit as st
import streamlitrunner as sr
from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame, Series
from plotly.graph_objects import Contour, Figure, Histogram, Scatter
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore, truncnorm

from deterministic_analisis import DeterministicAnalisisFigure, axeskwargs
from session_state import ss
from tables_types import InjectionLayerTable, LayerTable

# os.system("cls")

VERTICAL_DIVISIONS: int = 1000

gamma: NDArray[float64] = np.zeros((VERTICAL_DIVISIONS, 1))


# %%          FUNCTIONS
############# FUNCTIONS ################################################################


def FS(
    inj_id: int,
    z: float,
    alpha: float,
    gammaW: float,
    dPr: float,
    dP: float,
    Ko: float,
    Ka: float,
    theta: float,
    phi: float,
    c: float,
    *gamma_thickness: tuple[float, float],
):
    gammas, thickness = zip(*gamma_thickness)
    gammas = np.array(gammas)
    thickness = np.array(thickness)
    SvEff0 = (
        (gammas[:inj_id] * thickness[:inj_id]).sum()
        + gammas[inj_id](z - thickness[:inj_id].sum())
        - alpha * (gammaW * z + dPr)
    )
    SvEff = SvEff0 - alpha * dP
    ShEff = Ko * SvEff0 - Ka * alpha * dP
    return (
        c + (ShEff * (np.cos(theta) ** 2) + SvEff * (np.sin(theta) ** 2)) * np.tan(phi)
    ) / ((SvEff - ShEff) * np.sin(theta) * np.cos(theta))


def dist(
    number_of_values: int,
    distribution_choice: Literal["Uniform", "Triangular", "Normal"],
    mean: float,
    std_dev: float,
    lower_value: float,
    upper_value: float,
):
    """Return a distribution based on statistics"""
    a = (lower_value - mean) / std_dev
    b = (upper_value - mean) / std_dev
    if distribution_choice == "Uniform":
        return np.random.uniform(lower_value, upper_value, number_of_values)
    elif distribution_choice == "Triangular":
        return np.random.triangular(lower_value, mean, upper_value, number_of_values)
    elif distribution_choice == "Normal":
        return truncnorm.rvs(a, b, mean, std_dev, number_of_values)


def line(
    row: int, col: int, value: float, figure: Figure, horizontal: bool = False, **kwargs
):
    """Adds a line to `fig`, horizontal or vertical"""
    if horizontal:
        figure.add_hline(value, row, col, line_color="black", line_width=2, **kwargs)  # type: ignore
    else:
        figure.add_vline(value, row, col, line_color="black", line_width=2, **kwargs)  # type: ignore


def plot(
    row: int,
    col: int,
    x: NDArray[float64] | list[float],
    y: NDArray[float64] | list[float],
    name: str,
    figure: Figure,
    color: str = "black",
    width: int = 1,
    showlegend: bool = True,
    **kwargs,
):
    """Adds a Scatter plot to `fig`"""
    figure.add_trace(
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


from dataframes import (
    column_config_injection_layer_df,
    column_config_layers_df,
    injection_layer_df,
    layers_df,
)

# %%          MAIN FUNCTION
############# MAIN FUNCTION ############################################################


def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
    <style>
        footer {visibility: hidden;}

        /* Remove top padding from main container */
        .block-container {
            padding-bottom: 0rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if ss.get("run_calcs") is None:
        ss.run_calcs = True

    if ss.get("layer_names") is None:
        ss.layer_names = layers_df["layer"].to_list()

    if ss.get("layer_slider_value") is None:
        ss.layer_slider_value = 460

    if ss.get("dP_slider_value") is None:
        ss.dP_slider_value = 1

    if ss.get("figure_tab2") is None:
        ss.figure_tab2 = None

    if ss.get("bins") is None:
        ss.bins = 50

    tab1, tab2, tab3 = st.tabs(
        [
            "Parameters",
            "Results: deterministic analisis",
            "Results: probabilistic analisis",
        ]
    )

    ####################################################################################
    # %           FIRST TAB
    ####################################################################################

    def set_run_calcs():
        ss.run_calcs = True

    cols = tab1.columns([13, 5, 4, 4, 4, 5, 4, 3])
    cols[0].header("Parameters of the injection layer")
    dPres = cols[1].number_input(value=0.0, label="Initial overpressure [MPa]")
    dPmax = cols[2].number_input(value=5.0, label="Maximum ΔP [MPa]")
    numDP = cols[3].number_input(value=50, label="Number of steps")
    gamaW = cols[4].number_input(value=10.0, label="Water gradient [kN/m³]")
    listcontainer = cols[5].container()

    Nrel = cols[6].number_input("Number of realizations", value=100000)
    cols[7].button("Run", on_click=set_run_calcs)

    ## DATAFRAMES

    final_injection_layer_df = InjectionLayerTable(
        tab1.data_editor(
            injection_layer_df,
            column_config=column_config_injection_layer_df,
            hide_index=True,
        )
    )

    tab1.header("Layer depths and Specific Weight - γ [kN/m³]")
    final_layers_df = LayerTable(
        tab1.data_editor(
            layers_df,
            column_config=column_config_layers_df,
            hide_index=True,
            num_rows="dynamic",
        )
    )

    if None in final_layers_df.iloc[-1, :].to_list():
        tab1.text("Finish input data")
        return

    ss.layer_names = final_layers_df["layer"].to_list()

    with listcontainer:
        inj_lay_name = st.selectbox("Injection layer", ss.layer_names, index=3)

    if "inj_top" not in ss:
        ss.inj_top = final_layers_df.depth[3]
    if "inj_bas" not in ss:
        ss.inj_bas = final_layers_df.depth[4]

    if None not in final_layers_df.depth:
        inj_layer_pos = (
            (final_layers_df.iloc[:, 0] == inj_lay_name).to_list().index(True)
        )
        ss.inj_top = (
            0.0 if inj_layer_pos == 0 else final_layers_df.depth[inj_layer_pos - 1]
        )
        ss.inj_bas = final_layers_df.depth[inj_layer_pos]

    inj_top = ss.inj_top
    inj_bas = ss.inj_bas

    ####################################################################################
    # %           GETTING PARAMETERS
    ####################################################################################

    depths = final_layers_df.depth.to_numpy()
    depths = np.insert(depths, 0, 0)
    max_depth = depths.max()
    z = np.linspace(0, max_depth, VERTICAL_DIVISIONS)
    inj_pos = (inj_top <= z) & (z <= inj_bas)
    dz = np.insert(np.diff(z), 0, 0)[:, None]

    for i in final_layers_df.index:
        gamma[(depths[i] <= z) & (z <= depths[i + 1])] = final_layers_df.average[i]

    Ko = final_injection_layer_df.get_mean_value_from_df("Ko")
    Ka = final_injection_layer_df.get_mean_value_from_df("Ka")
    biot = final_injection_layer_df.get_mean_value_from_df("biot")
    coh = final_injection_layer_df.get_mean_value_from_df("cohesion") * 1000
    phi = np.radians(final_injection_layer_df.get_mean_value_from_df("friction"))
    theta = np.radians(final_injection_layer_df.get_mean_value_from_df("inclination"))

    # MPa to kPa
    dPmax *= 1000
    dPres *= 1000
    numDP += 1

    dP = np.linspace(0, dPmax, numDP)

    ####################################################################################
    # %           CALCULATION
    ####################################################################################

    Pp = gamaW * z[:, None] * np.ones(numDP)
    FSres = np.zeros(VERTICAL_DIVISIONS)[:, None] * np.ones(numDP)

    Pp[inj_pos, :] = Pp[inj_pos, :] + dPres
    for i in range(numDP):
        Pp[inj_pos, i] = Pp[inj_pos, i] + dP[i]

    SvTotal = np.cumsum(gamma * dz)
    SvEff = SvTotal[:, None] - biot * Pp
    ShEff = Ko * SvEff
    for i in range(1, numDP):
        ShEff[inj_pos, i] = ShEff[inj_pos, 0] - Ka * biot * dP[i]
    ShTotal = ShEff + biot * Pp
    Tn = np.abs(((ShEff - SvEff) / 2) * np.sin(2 * theta))
    Sn = ShEff * (np.cos(theta)) ** 2 + SvEff * (np.sin(theta)) ** 2
    Ts = coh + Sn * np.tan(phi)
    FSres[inj_pos, :] = Ts[inj_pos, :] / Tn[inj_pos, :]
    FSdet = FSres[inj_pos, :]

    SvTotal /= 1000
    SvEff /= 1000
    ShEff /= 1000
    ShTotal /= 1000
    Pp /= 1000
    Sn /= 1000
    Tn /= 1000
    Ts /= 1000
    coh /= 1000
    dPmax /= 1000

    dP /= 1000

    layer = VERTICAL_DIVISIONS - ss.layer_slider_value
    dP_slider = ss.dP_slider_value

    # dP_slider *= 1000
    dPstep: int = np.where(dP >= dP_slider if dPmax > 0 else dP <= dP_slider)[0][0]

    ####################################################################################
    # %           BUILD FIGURE
    ####################################################################################
    f = DeterministicAnalisisFigure(z)
    f.update_stress_axes(max_depth)

    f.add_stress_curve(SvTotal, "red", "σ<sub>v</sub> Total")
    f.add_stress_curve(SvEff[:, dPstep], "brown", "σ'<sub>v</sub> Effective")
    f.add_stress_curve(ShTotal[:, dPstep], "green", "σ<sub>h</sub> Total")
    f.add_stress_curve(ShEff[:, dPstep], "cornflowerblue", "σ'<sub>h</sub> Effective")
    f.add_stress_curve(Pp[:, dPstep], "blue", "ΔP")

    f.update_fault_stress_axes(np.max(list(map(np.max, [Tn, Ts, Sn]))))
    f.add_fault_stress_curve(Sn[:, dPstep], "black", "σ<sub>n</sub> Normal")
    f.add_fault_stress_curve(Tn[:, dPstep], "magenta", "𝜏<sub>n</sub> Shear")
    f.add_fault_stress_curve(Ts[:, dPstep], "chocolate", "𝜏<sub>s</sub> Limit")

    f.update_FS_axes(np.max(FSres))
    f.add_FS_curve(FSres[:, dPstep], "black", "SF")
    f.add_FS_hline(1.0)
    Smax = np.max(Sn)
    Tmax = coh + Smax * np.tan(phi)

    f.update_mohr_coulomb_axes(Smax, Tmax)
    f.plot_mohr_envelope(coh)
    f.plot_mohr_circle(SvEff[layer, dPstep], ShEff[layer, dPstep], np.degrees(theta))

    f.add_hlines_stress_curve(depths)
    f.update_contour_axes(dP.min(), dP.max(), z[inj_pos][-1], z[inj_pos][0])
    f.add_contours(x=dP, y=z[inj_pos], z=FSdet)
    f.update_layout()

    # ss.figure_tab2 = fig

    ####################################################################################
    # %
    ####################################################################################

    figure = f.fig  # Figure(ss.figure_tab2)

    dP_slider_container = tab2.container()
    cols = tab2.columns([1, 35])
    layer_slider_container = cols[0].container()

    f.update_current_point(dP[dPstep], z[layer])

    event = cols[1].plotly_chart(
        figure, theme=None, on_select="rerun", selection_mode="points"
    )
    if event and event["selection"] and event["selection"]["points"]:
        ss.dP_slider_value = event["selection"]["points"][0]["x"]
        ss.layer_slider_value = VERTICAL_DIVISIONS - np.searchsorted(
            z, event["selection"]["points"][0]["y"]
        )
        st.rerun()

    with dP_slider_container:
        st.slider(
            "Pressure step - ΔP [MPa]",
            min_value=0.0 if dPmax > 0 else dPmax,
            max_value=dPmax if dPmax > 0 else 0.0,
            step=np.abs(dPmax / (numDP - 1)),
            # value=1.00,
            key="dP_slider_value",
        )

    with layer_slider_container:
        from streamlit_vertical_slider import vertical_slider

        slider_value = vertical_slider(
            label="Layer",
            min_value=1,
            max_value=VERTICAL_DIVISIONS,
            step=1,
            height=580,
            value_always_visible=False,
            default_value=int(ss.layer_slider_value),
            key=f"layer_slider_value{ss.layer_slider_value}",
        )

        if slider_value != ss.layer_slider_value:
            ss.layer_slider_value = slider_value
            st.rerun()

    ####################################################################################
    # %           GRAPHS ON THIRD TAB
    ####################################################################################

    cols = tab3.columns([6, 1, 4])
    cols[0].header("Histogram distribution of the injection layer parameters")
    nbins = cols[1].number_input("Number of bins", value=ss.bins)
    ss.bins = nbins
    cols[2].header("Analisis results")
    fig_prob = make_subplots(
        rows=2,
        cols=4,
        horizontal_spacing=0.05,
        vertical_spacing=0.075,
    )
    rows = [1, 1, 1, 2, 2, 2]
    cols = [1, 2, 3, 1, 2, 3]
    for i in final_injection_layer_df.index:

        data = final_injection_layer_df.iloc[i].to_list()
        hist = dist(Nrel, *data[1:])
        fig_prob.update_yaxes(
            row=rows[i],
            col=cols[i],
            title_text=f"Frequency",
            **axeskwargs,
        )

        fig_prob.update_xaxes(
            row=rows[i],
            col=cols[i],
            title_text=data[0],
            **axeskwargs,
        )
        fig_prob.add_trace(
            Histogram(
                x=hist,
                nbinsx=nbins,
                name=data[0],
                marker=dict(line=dict(width=0)),
            ),
            row=rows[i],
            col=cols[i],
        )
        fig_prob.add_annotation(
            text=f"Mean = {data[2]:.2f}<br>"
            f"Std = {data[3]:.2f}<br>"
            f"Min = {data[4]:.2f}<br>"
            f"Max = {data[5]:.2f}<br>"
            f"Samples = {Nrel}",
            xref="x2 domain",
            yref="y2 domain",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bordercolor="black",
            borderwidth=1,
            bgcolor="white",
            row=rows[i],
            col=cols[i],
        )

    fig_prob.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        height=650,
        margin=dict(l=50, r=40, t=30, b=0),
        showlegend=True,
    )
    tab3.plotly_chart(fig_prob, theme=None)

    ss.run_calcs = False


if __name__ == "__main__":
    # main()
    sr.run(main, screen=1, open_as_app=True, fill_page_content=False)
