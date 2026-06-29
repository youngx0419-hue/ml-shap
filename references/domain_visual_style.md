# Domain Visual Style Guide

Use this file after domain detection and nearest-paper search. The goal is to make the figures look native to the research field while preserving publication clarity.

## General Rules

- Keep `axes.facecolor` white for quantitative charts.
- Use the domain background only for `figure.facecolor`, report accents, maps, or panel bands.
- Use sequential palettes for magnitude, diverging palettes for signed SHAP or residuals, and categorical palettes for groups.
- Do not use a one-color theme for all chart types.
- Always include units in axis labels when domain units are known.
- For maps and risk surfaces, include uncertainty or applicability warnings when available.

## Domain Style Cards

| Domain | Palette | Visual Emphasis | Preferred Plots |
|---|---|---|---|
| `carbon` | muted red-orange + charcoal + neutral gray | emissions pathway, decoupling, policy scenarios, regional comparison | scenario lines, stacked contribution bars, SHAP dependence with threshold bands, regional maps |
| `water` | blue-green + teal + amber warning | hydrological periods, pollutant gradients, basin/station structure | hydro-period beeswarm, parameter heatmaps, station maps, exceedance risk heatmaps |
| `risk` | colorblind-safe yellow-orange-red + dark gray | probability, uncertainty, thresholds, risk zoning | calibrated risk curves, risk maps, uncertainty bands, confusion matrix, quadrant charts |
| `spatial_temporal` | viridis/cividis + subdued categorical accents | spatial autocorrelation, temporal evolution, block validation | faceted maps, space-time heatmaps, Moran/Gi* maps, temporal SHAP ridgelines |
| `biochar_materials` | violet + green + graphite + warm highlight | process-property-performance links | process contour plots, materials radar/spider charts, SHAP beeswarm by feedstock, feature family heatmaps |
| `health_environment` | clinical blue + muted red + gray | risk calibration, subgroup fairness, exposure-response | calibration curves, subgroup forest plots, exposure-response ALE, decision-curve style plots |
| `agriculture_ecology` | green + earth + sky blue | seasonal dynamics, productivity, soil/vegetation gradients | seasonal panels, yield response ALE, map overlays, feature family bars |
| `energy_systems` | electric blue + amber + slate | load/renewable time profiles and scenario uncertainty | time-series forecast bands, SHAP by hour/season, scenario fan charts |
| `finance_business` | restrained blue + green/red deltas + gray | threshold decisions, lift, calibration, segments | lift/gain charts, calibrated probability plots, segment SHAP bars, waterfall for cases |
| `general` | neutral gray + blue + amber | clarity and robustness | beeswarm, bar, dependence/ALE, manual waterfall, method-decision summary |

## Plot Routing

| Domain Need | Chart Choice |
|---|---|
| Policy/scenario pathway | multi-scenario line chart with peak/threshold annotation |
| Pollution or exceedance risk | station/basin map plus time-period heatmap |
| Disaster or health risk | calibrated probability plot plus risk map and uncertainty map |
| Spatial transferability | block-CV performance map plus applicability/dissimilarity map |
| Materials optimization | 2D contour/ALE over process variables plus optimal region annotation |
| Correlated scientific predictors | grouped feature SHAP plus ALE, not only PDP |

## Style JSON Template

```json
{
  "palette_name": "water",
  "figure_facecolor": "#f2f7f7",
  "axes_facecolor": "white",
  "primary": "#287c8e",
  "secondary": "#6bb7a8",
  "accent": "#d9a441",
  "diverging_cmap": "RdBu_r",
  "sequential_cmap": "viridis",
  "map_cmap": "YlGnBu",
  "risk_cmap": "YlOrRd",
  "notes": ["Use hydrological period as hue when available."]
}
```
