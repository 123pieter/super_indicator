"""Interactive financial charts; deliberately independent of Streamlit."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def coin_chart(frame, threshold, *, show_bollinger=False):
    """Share time axes across price, volume, and score, with optional price overlays."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[.55, .16, .29], vertical_spacing=.035)
    fig.add_trace(go.Candlestick(x=frame.index, open=frame.open, high=frame.high,
                                low=frame.low, close=frame.close, name="Price",
                                increasing_line_color="#0f766e", decreasing_line_color="#dc5b5b"), row=1, col=1)
    if show_bollinger:
        middle = frame.bb_mid
        spread = 2 * frame.bb_std
        for label, values, dash in (
            ("Upper", middle + spread, "solid"),
            ("Middle", middle, "dash"),
            ("Lower", middle - spread, "solid"),
        ):
            fig.add_trace(go.Scatter(
                x=frame.index, y=values, mode="lines", name=f"Bollinger {label}",
                line=dict(color="#7c3aed", width=1.3, dash=dash),
                hovertemplate=f"Bollinger {label}: %{{y:.6g}}<extra></extra>",
            ), row=1, col=1)
    # Crosses are computed before date filtering so a boundary does not create a fake signal.
    signals = frame.loc[frame["cross"]]
    fig.add_trace(go.Scatter(x=signals.index, y=signals.low*.98, mode="markers",
                            marker=dict(symbol="triangle-up", size=11, color="#0f766e"),
                            name="Regime-qualified cross", hovertemplate="%{x}<br>Threshold crossed; ATR regime eligible<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=frame.index, y=frame.volume, name="Volume", marker_color="#a4b7c8"), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.super_score, name="Score", mode="lines",
                            line=dict(color="#2563eb", width=2), hovertemplate="%{x}<br>Score %{y:.3f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#d97706", row=3, col=1)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Score", range=[-.03, 1.03], row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=700, margin=dict(l=10,r=15,t=25,b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.07, x=0), dragmode="zoom")
    return fig


def equity_chart(results, selected, compare_all=False):
    """Plot raw, unstacked equity for the selected experiment; preserve result identity."""
    chosen = results[selected]
    visible = results if compare_all else {selected: chosen}
    fig = go.Figure()
    for key, result in visible.items():
        equity = result["equity"]
        name = f"{result['label']} / {result['direction']}"
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.to_numpy(), name=name, mode="lines",
            line=dict(width=3 if key == selected else 1.3),
            opacity=1 if key == selected else .55,
            customdata=((equity.to_numpy()-1)*100),
            hovertemplate=("%{fullData.name}<br>%{x}<br>Equity: %{y:.5f}"
                           "<br>Return from start: %{customdata:.2f}%<extra></extra>")))
    fig.add_hline(y=1., line_dash="dash", line_color="#94a3b8")
    fig.update_layout(title=f"{chosen['label']} / {chosen['direction']}" +
                      (" — compared with other experiments" if compare_all else ""),
                      yaxis_title="Equity (initial capital = 1)", xaxis_title="Candle start (UTC)",
                      height=480, hovermode="closest", margin=dict(l=15,r=15,t=55,b=15))
    return fig
