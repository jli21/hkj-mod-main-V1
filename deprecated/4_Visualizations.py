"""Visualizations page — distributions, scatter plots, rider rankings, trends."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui_helpers import (app_css, setup_sidebar, get_available_blup_cols)

app_css()
blup_filtered, _ = setup_sidebar()

st.header("Model Visualizations")

if blup_filtered is not None:
    available_blup = get_available_blup_cols(blup_filtered)

    if not available_blup:
        st.warning("No BLUP columns found in the data.")
        st.stop()

    # ---- Distribution plots ----
    st.subheader("BLUP Feature Distributions")
    dist_cols = st.columns(min(len(available_blup), 3))

    for i, col in enumerate(available_blup[:3]):
        with dist_cols[i % 3]:
            fig = px.histogram(
                blup_filtered, x=col, nbins=60,
                title=col.replace('_', ' ').title(),
                labels={col: col},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red",
                          annotation_text="Pop. Mean")
            fig.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

    # ---- PPM_uncertainty distribution ----
    if 'PPM_uncertainty' in blup_filtered.columns:
        st.subheader("PPM Uncertainty Distribution")
        st.markdown(
            "Uncertainty in the participant performance estimate. "
            "Used to **dampen Kelly bet fractions** for horses with less data."
        )
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            fig_unc = px.histogram(
                blup_filtered, x='PPM_uncertainty', nbins=60,
                title='PPM Uncertainty',
                color_discrete_sequence=['#e74c3c'],
            )
            med_u = blup_filtered['PPM_uncertainty'].median()
            fig_unc.add_vline(x=med_u, line_dash="dash", line_color="black",
                              annotation_text=f"Median: {med_u:.4f}")
            fig_unc.update_layout(height=350)
            st.plotly_chart(fig_unc, use_container_width=True)
        with col_u2:
            st.markdown("**Kelly Damping Formula:**")
            st.latex(r"\text{damping}_i = \exp\!\left(-\frac{u_i}{\tilde{u}}\right)")
            st.markdown(
                "Where $u_i$ is `PPM_uncertainty` and $\\tilde{u}$ is the median. "
                "Horses with uncertainty >> median get their Kelly fraction "
                "exponentially reduced."
            )

    # ---- Intercept vs Slope scatter ----
    if ('participant_re_intercept' in blup_filtered.columns
            and 'participant_re_slope' in blup_filtered.columns):
        st.subheader("Intercept vs Slope")
        n_sample = min(5000, len(blup_filtered))
        sample = blup_filtered.sample(n_sample, random_state=42)

        color_col = 'PPM_uncertainty' if 'PPM_uncertainty' in sample.columns else None

        fig_scatter = px.scatter(
            sample,
            x='participant_re_intercept',
            y='participant_re_slope',
            color=color_col,
            color_continuous_scale='YlOrRd' if color_col else None,
            opacity=0.4,
            title='Participant Random Effects (sampled)',
            labels={
                'participant_re_intercept': 'Intercept (baseline speed)',
                'participant_re_slope': 'Slope (distance response)',
            },
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "**Quadrants:** Top-right = fast baseline & prefers long distances. "
            "Bottom-left = slow & prefers short. Color intensity shows estimation uncertainty."
        )

    # ---- Top Riders ----
    rider_col_name = None
    for rc in ['Rider', 'Jockey']:
        if rc in blup_filtered.columns:
            rider_col_name = rc
            break

    if rider_col_name and 'rider_resid_mean_shrunk' in blup_filtered.columns:
        st.subheader("Top Riders by Shrunk Residual")
        min_rides = st.slider("Minimum rides", 10, 200, 30, key="min_rides_viz")

        rider_agg = (
            blup_filtered.groupby(rider_col_name)
            .agg(
                avg_effect=('rider_resid_mean_shrunk', 'mean'),
                rides=('rider_resid_mean_shrunk', 'count'),
            )
            .query(f"rides >= {min_rides}")
            .sort_values('avg_effect', ascending=False)
            .head(25)
            .reset_index()
        )

        fig_rider = px.bar(
            rider_agg,
            x='avg_effect', y=rider_col_name,
            orientation='h',
            title=f'Top 25 Riders (min {min_rides} rides)',
            color='avg_effect',
            color_continuous_scale='RdYlGn',
        )
        fig_rider.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_rider, use_container_width=True)

else:
    st.info("No BLUP data available.")
