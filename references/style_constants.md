---
name: style-constants-ML-SHAP
description: Times New Roman SCI字体 × 五套高级配色 × SVG路径嵌入 × Unicode化学下标 × 手动Waterfall。ML-SHAP: 弃用SimHei, 采用Times New Roman + sanitize_svg_u2212()。
---

# XGBoost-SHAP 绘图系统 ML-SHAP

> **ML-SHAP 核心变更**: (1) 字体从 SimHei 切换为 **Times New Roman** (SCI 标准, 完整 glyph 覆盖); (2) 所有 SVG 保存后执行 `sanitize_svg_u2212()` 防御性清除; (3) 特征名强制 Unicode 化学下标 (chem_sub); (4) Waterfall 手动 barh 替代 SHAP 内置; (5) fix_shap_text() 废弃。

---

## 零、全局铁律

### 0.0 字体配置 [ML-SHAP — Times New Roman, 弃用 SimHei]

```python
# ═══════════════════════════════════════════════════════════════
# ML-SHAP: Times New Roman — SCI 出版物标准, 完整 glyph 覆盖
# SimHei 缺陷: 无 U+2212 字形 / 数字渲染差 / 非 SCI 标准 → 已弃用
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'path'  # 路径嵌入字体, 跨平台一致

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Step 1: 查找 Times New Roman TTF
TNR_PATH = None
for f in fm.fontManager.ttflist:
    if 'times new roman' in f.name.lower() and f.fname.lower().endswith('.ttf'):
        TNR_PATH = f.fname
        fm.fontManager.addfont(f.fname)
        break
if TNR_PATH is None:
    for p in ['C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/Times.ttf']:
        if os.path.exists(p):
            TNR_PATH = p
            fm.fontManager.addfont(p)
            break

import seaborn as sns
sns.set_style('white')  # 必须先调用 (会重置 font.family)

# Step 2: 设置 Times New Roman (在 sns.set_style() 之后!)
if TNR_PATH:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # ASCII hyphen 作为负号
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

# Step 3: 验证字体
from matplotlib.font_manager import FontProperties
actual_font = fm.findfont(FontProperties())
assert 'times' in actual_font.lower() if TNR_PATH else True, \
    f'Font resolution error: {actual_font}'

# Step 4: rcParams
plt.rcParams.update({
    'axes.facecolor': 'white',
    'figure.facecolor': '#f2f5f3',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'grid.color': '#e0e0e0',
})

# Step 5: 现在安全 import shap
import shap

# ═══════════════════════════════════════════════════════════════
# 严禁 SimHei: 无 U+2212 字形, 数字衬线渲染差
# 严禁 SHAP 内置 waterfall_plot(): 内部使用 U+2212
# ═══════════════════════════════════════════════════════════════
```

### 0.0a warnings 精准过滤 [ML-SHAP 保留]

```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
# 保留 glyph-missing 警告
```

### 0.0b SVG U+2212 防御性清除 [ML-SHAP NEW — 替代 fix_shap_text]

```python
def sanitize_svg_u2212(svg_path):
    """Post-save: replace all U+2212 (MINUS SIGN) with U+002D (HYPHEN-MINUS).
    Defense-in-depth: catches any U+2212 that bypassed matplotlib text settings.
    """
    with open(svg_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    count = content.count('−')
    if count > 0:
        content = content.replace('−', '-')
        with open(svg_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
    return count
```

### 0.0c Unicode 化学下标 [??]

```python
SUB = str.maketrans({
    '0': '₀', '1': '₁', '2': '₂', '3': '₃',
    '4': '₄', '5': '₅', '6': '₆', '7': '₇',
    '8': '₈', '9': '₉', 'x': 'ₓ',
})

def chem_sub(feat):
    """PM2.5 → PM₂.₅, NO2 → NO₂, SO2 → SO₂, NH3 → NH₃, O3 → O₃, NOx → NOₓ"""
    result = []
    for i, ch in enumerate(feat):
        if ch.isdigit() or (ch == 'x' and i > 0 and feat[i-1].isalpha()):
            result.append(ch.translate(SUB))
        else:
            result.append(ch)
    return ''.join(result)

# 使用: DISPLAY_NAMES = [chem_sub(n) for n in RAW_NAMES]
# 禁止 print(DISPLAY_NAMES) — Windows GBK 会 crash
```

### 0.0d 热力图 colormap 规范 [ML-SHAP 保留]

```python
from matplotlib.colors import LinearSegmentedColormap

# 蓝橙双色交互矩阵: 中间色 #d5d8d3 (Azure 浅灰绿), 与 white axes 可区分
cmap_inter = LinearSegmentedColormap.from_list('inter', [
    '#2166ac', '#92c5de', '#d5d8d3', '#f4a582', '#d6604d'
])
```

### 0.1 网格线规则

| 图表类型 | 网格线 |
|---------|--------|
| 散点图/条形图 | 允许 `ax.grid(axis='x', alpha=0.3, ls='--'); ax.set_axisbelow(True)` |
| imshow 热力图 | **禁止** `ax.grid(False)` |
| 瀑布图 | 允许 `ax.grid(axis='x', alpha=0.3, ls='--')` |

### 0.2 图表尺寸标准

| 图类型 | figsize |
|--------|---------|
| 单面板 (Beeswarm, Importance, Bootstrap CI) | (7, 5) |
| Waterfall 手动 barh | (7, n×0.35+1.5) |
| 依赖图 ×1 | (7, 5) |
| Main vs Interaction | (10, 5.5) |
| Interaction Matrix | (10, 8) |
| SHAP Heatmap | (13, 8) |
| Prediction Diagnostics (1×3) | (13, 4.2) |
| SHAP Clustering (2×2) | (10, 8) |
| Correlation Heatmap | (10, 8) |
| City AQI / Seasonal | (14, 5-6) |

### 0.3 保存函数 [ML-SHAP 强化]

```python
def save_chart(fig, name):
    """保存 SVG + PNG, 自动 U+2212 清除"""
    svg_path = f'{CHARTS_DIR}/{name}.svg'
    png_path = f'{CHARTS_DIR}/{name}.png'
    fig.savefig(svg_path, facecolor=AZURE_BG, edgecolor='none',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_path, dpi=300, facecolor=AZURE_BG, edgecolor='none',
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    sanitize_svg_u2212(svg_path)  # ML-SHAP: 防御性清除
```

---

## 一、五套配色体系

### 1.1 碧山栀子 Bishan Gardenia
```python
BISHAN = {
    'bg': '#f5f0e0', 'primary': '#5a783c', 'secondary': '#d2b43c',
    'accent1': '#e8c8a0', 'accent2': '#c9a05a', 'dark': '#3d5226',
}
```

### 1.2 雨过天青 Rain Azure
```python
RAIN_AZURE = {
    'bg': '#f2f5f3', 'primary': '#5b8c85', 'secondary': '#6b9ec4',
    'accent1': '#a3c4bc', 'accent2': '#d6644a', 'dark': '#1e3630',
}
```

### 1.3 霜降柿红 Frost Persimmon
```python
FROST_PERSIMMON = {
    'bg': '#f9f3ea', 'primary': '#c1724a', 'secondary': '#8b5e3c',
    'accent1': '#d4946e', 'accent2': '#e8c9a0', 'dark': '#5c3d2e',
}
```

### 1.4 松烟墨韵 Pine Ink
```python
PINE_INK = {
    'bg': '#f8f7f4', 'primary': '#3d3d4d', 'secondary': '#a89060',
    'accent1': '#6d6d7a', 'accent2': '#9a9aa5', 'dark': '#2a2a33',
}
```

### 1.5 紫藤花语 Wisteria
```python
WISTERIA = {
    'bg': '#f6f4f8', 'primary': '#7b6b8e', 'secondary': '#6a9a7a',
    'accent1': '#9b8db5', 'accent2': '#8aad8e', 'dark': '#3a3345',
}
```

### 1.6 配色注册表
```python
PALETTE_REGISTRY = {'bishan': BISHAN, 'azure': RAIN_AZURE, 'frost': FROST_PERSIMMON, 'ink': PINE_INK, 'wisteria': WISTERIA}
DOMAIN_PALETTE_MAP = {'air_quality':'azure', 'water':'azure', 'spatial':'azure', 'carbon':'frost', 'risk':'frost', 'biochar':'wisteria', 'general':'bishan'}
```

---

## 二、ML-SHAP 质量门禁 (每次分析完成后确认)

- [ ] Times New Roman + svg.fonttype='path' 已配置
- [ ] chem_sub() Unicode 下标已应用到所有图表
- [ ] Waterfall 使用手动 barh (非 SHAP 内置)
- [ ] save_chart() 内部调用 sanitize_svg_u2212()
- [ ] 交互效应使用逐样本分解算法
- [ ] 所有 print() ASCII-safe (无 R²/₂/₃/→)
- [ ] 报告表格逐列显式赋值
- [ ] 报告 §0 因果声明存在
