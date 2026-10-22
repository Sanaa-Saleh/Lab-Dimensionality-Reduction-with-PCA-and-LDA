"""Shared chart styling: consistent legends and protected 2D axes."""
def finish_chart(fig):
    if fig is None:
        return fig
    scene = any(str(k).startswith("scene") for k in fig.layout.to_plotly_json())
    if not scene:
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    # Legend only where traces use one; 3D PC4 colors remain untouched.
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.20, yanchor="top", title_text=""))
    margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    if not scene:
        margin["b"] = max(95, margin.get("b") or 0)
        fig.update_layout(margin=margin)
    return fig
