from __future__ import annotations

import os

import numpy as np
import streamlitrunner as sr
from numpy import float64
from numpy.typing import NDArray
from tqdm import tqdm

from calc_functions import FS, dist
from dataframes import injection_layer_df, layers_df
from figure_deterministic import DeterministicAnalisisFigure
from figure_probabilistic import ProbabilisticAnalisisFigure
from tables_types import InjectionLayerTable, LayerTable

VERTICAL_DIVISIONS: int = 1000

COLUMNS: int = 100
try:
    COLUMNS: int = os.get_terminal_size().columns
except:
    pass

plotly_chart_options = {"theme": None, "on_select": "rerun", "selection_mode": "points"}

gamma: NDArray[float64] = np.zeros((VERTICAL_DIVISIONS, 1))

# %%          MAIN FUNCTION
############# MAIN FUNCTION ############################################################


def main():
    import streamlit as st

    from column_configs import col_conf_injection_layer, col_conf_layers
    from session_state import ss

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

    ss.init_key("run_calcs", True)
    ss.init_key("fig2", None)
    ss.init_key("fig3", None)
    ss.init_key("layer_slider_value", 460)
    ss.init_key("dP_slider_value", 0)
    ss.init_key("SFs", None)

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
    dPini = cols[1].number_input(value=0.0, label="Initial overpressure [MPa]")
    dPmax = cols[2].number_input(value=5.0, label="Maximum ΔP [MPa]")
    numDP = cols[3].number_input(value=50, label="Number of steps")
    gamaW = cols[4].number_input(value=10.0, label="Water gradient [kN/m³]")
    listcontainer = cols[5].container()

    Nrel = cols[6].number_input("Number of realizations", value=15_000)
    cols[7].button("Run", on_click=set_run_calcs)

    ## DATAFRAMES

    final_injection_layer_df = InjectionLayerTable(
        tab1.data_editor(
            injection_layer_df,
            column_config=col_conf_injection_layer,
            hide_index=True,
        )
    )

    tab1.header("Layer depths and Specific Weight - γ [kN/m³]")
    final_layers_df = LayerTable(
        tab1.data_editor(
            layers_df,
            column_config=col_conf_layers,
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

    depths: np.ndarray = final_layers_df.depth.to_numpy()
    thickness: list[float] = list(np.diff(np.insert(depths, 0, 0)))
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
    dPini *= 1000
    numDP += 1

    dP = np.linspace(0, dPmax, numDP)

    ####################################################################################
    # %           CALCULATION
    ####################################################################################

    Pp = gamaW * z[:, None] * np.ones(numDP)
    FSres = np.zeros(VERTICAL_DIVISIONS)[:, None] * np.ones(numDP)

    Pp[inj_pos, :] = Pp[inj_pos, :] + dPini
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
    Smax = np.max(Sn)
    Tmax = coh + Smax * np.tan(phi)
    ####################################################################################
    # %           BUILD FIGURE
    ####################################################################################
    import copy

    f = copy.deepcopy(ss.fig2)
    if f is None or ss.run_calcs:
        f = DeterministicAnalisisFigure(z)
        f.update_stress_axes(max_depth)
        f.add_stress_curve(SvTotal, "red", "σ<sub>v</sub> Total")
        f.update_fault_stress_axes(np.max(list(map(np.max, [Tn, Ts, Sn]))))
        f.update_FS_axes(np.max(FSres))
        f.update_mohr_coulomb_axes(Smax, Tmax)
        f.plot_mohr_envelope(coh)
        f.update_contour_axes(dP.min(), dP.max(), z[inj_pos][-1], z[inj_pos][0])
        f.add_contours(x=dP, y=z[inj_pos], z=FSdet)
        f.update_layout()
        ss.fig2 = copy.deepcopy(f)

    f.add_stress_curve(SvEff[:, dPstep], "brown", "σ'<sub>v</sub> Effective")
    f.add_stress_curve(ShTotal[:, dPstep], "green", "σ<sub>h</sub> Total")
    f.add_stress_curve(ShEff[:, dPstep], "cornflowerblue", "σ'<sub>h</sub> Effective")
    f.add_stress_curve(Pp[:, dPstep], "blue", "ΔP")

    f.add_fault_stress_curve(Sn[:, dPstep], "black", "σ<sub>n</sub> Normal")
    f.add_fault_stress_curve(Tn[:, dPstep], "magenta", "𝜏<sub>n</sub> Shear")
    f.add_fault_stress_curve(Ts[:, dPstep], "chocolate", "𝜏<sub>s</sub> Limit")

    f.add_FS_curve(FSres[:, dPstep], "black", "SF")

    f.plot_mohr_circle(SvEff[layer, dPstep], ShEff[layer, dPstep], np.degrees(theta))

    f.add_hlines_stress_curve(depths)
    f.add_FS_hline(1.0)

    ####################################################################################
    # %           SLIDERS LOGIC
    ####################################################################################

    dP_slider_container = tab2.container()
    cols = tab2.columns([1, 35])
    layer_slider_container = cols[0].container()

    f.update_current_point(dP[dPstep], z[layer])

    event2 = cols[1].plotly_chart(f.fig, **plotly_chart_options)
    if event2 and event2["selection"] and event2["selection"]["points"]:
        print(event2)
        ss.dP_slider_value = event2["selection"]["points"][0]["x"]
        ss.layer_slider_value = VERTICAL_DIVISIONS - np.searchsorted(
            z, event2["selection"]["points"][0]["y"]
        )
        st.rerun()

    ####################################################################################
    # %           GRAPHS ON THIRD TAB
    ####################################################################################

    cols = tab3.columns([6, 1, 4])
    cols[0].header("Histogram distribution of the injection layer parameters")
    nbins = cols[1].number_input("Number of bins", value=100, on_change=set_run_calcs)
    ss.bins = nbins
    cols[2].header("Analisis results")

    z_inj: NDArray[float64] = np.linspace(z[inj_pos].min(), z[inj_pos].max(), 51)

    f = copy.deepcopy(ss.fig3)
    if f is None or ss.run_calcs:

        f = ProbabilisticAnalisisFigure(Nrel, nbins)

        hist_data = []
        for i in final_injection_layer_df.index:
            data = final_injection_layer_df.iloc[i].to_list()
            hist_data.append(dist(Nrel, *data[1:]))
            f.add_var_hist(i, data, hist_data[-1])

        _Ko, _Ka, _alpha, _cohesion, _friction, _angtheta = hist_data
        print("---- Getting meshgrid for alpha")
        _dP, _z, _alpha = np.meshgrid(dP, z_inj, _alpha)
        print("---- Getting meshgrid for Ko")
        _, _, _Ko = np.meshgrid(dP, z_inj, _Ko)
        print("---- Getting meshgrid for Ka")
        _, _, _Ka = np.meshgrid(dP, z_inj, _Ka)
        print("---- Getting meshgrid for cohesion")
        _, _, _cohesion = np.meshgrid(dP, z_inj, _cohesion)
        print("---- Getting meshgrid for friction")
        _, _, _friction = np.meshgrid(dP, z_inj, _friction)
        print("---- Getting meshgrid for theta")
        _, _, _angtheta = np.meshgrid(dP, z_inj, _angtheta)

        gamma_data: list[NDArray[float64]] = []
        for i in final_layers_df.index:
            data = final_layers_df.iloc[i].to_list()
            gamma_data.append(dist(Nrel, *data[2:]))

        for i, g in enumerate(
            tqdm(gamma_data, ncols=COLUMNS, desc="Getting meshgrid for gamma in layers")
        ):
            _, _, gamma_data[i] = np.meshgrid(dP, z_inj, g)

        print("Getting FS map")
        SFs = FS(
            inj_layer_pos,
            dPini,
            gamaW,
            _dP * 1000,  # MPa -> kPa
            _z,
            _alpha,
            _Ko,
            _Ka,
            np.radians(_angtheta),
            np.radians(_friction),
            _cohesion * 1000,  # MPa -> kPa
            gamma_data,
            thickness,
        )

        fp: NDArray[float64] = ((SFs < 1).astype(int).sum(axis=2) / Nrel) * 100.0
        f.update_contour_axes(dP.min(), dP.max(), z_inj[-1], z_inj[0])
        f.add_contours(dP, z_inj, fp)
        ss.SFs = SFs
        ss.fig3 = copy.deepcopy(f)
        ss.run_calcs = False

    z_inj_layer_idx = layer if 0 <= layer < len(z_inj) else 0
    f.update_current_point(dP[dPstep], z_inj[z_inj_layer_idx])
    assert ss.SFs is not None
    f.add_SF_hist(ss.SFs[dPstep, z_inj_layer_idx, :])
    f.update_layout()

    event3 = tab3.plotly_chart(f.fig, **plotly_chart_options)

    if event3 and event3["selection"] and event3["selection"]["points"]:
        ss.dP_slider_value = event3["selection"]["points"][0]["x"]
        ss.layer_slider_value = VERTICAL_DIVISIONS - np.searchsorted(
            z, event3["selection"]["points"][0]["y"]
        )
        st.rerun()

    with dP_slider_container:
        st.slider(
            "Pressure step - ΔP [MPa]",
            min_value=0.0 if dPmax > 0 else dPmax,
            max_value=dPmax if dPmax > 0 else 0.0,
            step=np.abs(dPmax / (numDP - 1)),
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


if __name__ == "__main__":
    # main()
    sr.run(main, screen=1, open_as_app=True, fill_page_content=False)
