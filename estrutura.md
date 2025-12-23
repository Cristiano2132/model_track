# Estrutura do pacote model_track

Documento gerado automaticamente — mapeamento técnico das classes, funções e exemplos.

## Sumário
- Estrutura de arquivos
- Descrições por módulo
- Exemplos de uso rápidos
- Diagramas UML simples

## Estrutura de arquivos (resumo)
- Diretórios: `binning`, `stability`, `stats`, `woe`
- Arquivos principais:
  - [src/model_track/binning/bins_applier.py](src/model_track/binning/bins_applier.py#L1-L200)
  - [src/model_track/binning/quantile_binner.py](src/model_track/binning/quantile_binner.py#L1-L200)
  - [src/model_track/binning/tree_binner.py](src/model_track/binning/tree_binner.py#L1-L200)
  - [src/model_track/woe/woe_calculator.py](src/model_track/woe/woe_calculator.py#L1-L400)
  - [src/model_track/woe/woe_by_period.py](src/model_track/woe/woe_by_period.py#L1-L200)
  - [src/model_track/stability/woe.py](src/model_track/stability/woe.py#L1-L400)
  - [src/model_track/stats/summary.py](src/model_track/stats/summary.py#L1-L200)
  - [src/model_track/stats/categorical_correlation.py](src/model_track/stats/categorical_correlation.py#L1-L20) *(vazio)*
  - [src/model_track/woe/__init__.py](src/model_track/woe/__init__.py#L1-L50)

## Descrições por módulo

**binning/bins_applier.py**
- `BinApplier`
  - `__init__(self, df: pd.DataFrame)`
  - `_validate_bins(self, bins: List[float])`
  - `_generate_labels(self, bins: List[float]) -> List[str]`
  - `apply(self, column: str, bins: List[float]) -> pd.Series`
- Finalidade: aplicar cortes (bins) predefinidos a uma coluna (usa `pd.cut`), retorna labels e converte NaNs para `"N/A"`.

**binning/quantile_binner.py**
- `QuantileBinner`
  - `__init__(self, n_bins=3, min_unique=5)`
  - `_prepare_data(self, df, feature)`
  - `fit(self, df, feature, target=None)`
- Finalidade: gerar pontos de corte por quantis; popula `self.bins_`. `target` é ignorado (compatibilidade API).

**binning/tree_binner.py**
- `TreeBinner`
  - `__init__(self, max_depth=2, min_samples_leaf=50)`
  - `_prepare_data(self, df, feature, target)`
  - `fit(self, df, feature, target)`
- Finalidade: treinar `DecisionTreeClassifier` raso para extrair thresholds (dependência: `sklearn`).

**woe/woe_calculator.py**
- `WoeCalculator` (métodos estáticos)
  - `compute_table(df, target_col, feature_col, event_value=1, epsilon=1e-8, add_totals=True, round=4) -> pd.DataFrame`
  - `compute_mapping(df, target_col, feature_col, event_value=1, epsilon=1e-8) -> Dict`
- Finalidade: calcular WOE e IV por categoria, com validações (target binário, proteção contra divisão por zero).

**woe/woe_by_period.py**
- `WoeByPeriod`
  - `compute(df, target_col, feature_col, date_col, event_value=1) -> pd.DataFrame`
- Finalidade: delega `WoeCalculator.compute_table` por período (`date_col`) e concatena resultados em formato long.

**stability/woe.py**
- `WoeStability`
  - `__init__(self, df: pd.DataFrame, date_col: str, event_value: int = 1)`
  - `global_table(self, feature_col: str, target_col: str) -> pd.DataFrame`
  - `generate_view(self, feature_col: str, target_col: str, ax: plt.Axes=None)`
- Finalidade: calcular tabela global de WOE e plotar evolução temporal (dependência: `matplotlib`).

**stats/summary.py**
- `get_summary(df: pd.DataFrame) -> pd.DataFrame`
- Finalidade: resumo por coluna (dtype, n_na, pct_na, top class, distincts, min/max para num/datetime).

**stats/categorical_correlation.py**
- Arquivo presente, sem implementação.

## Exemplos de uso rápidos

- Aplicar bins já calculados:
```python
from model_track.binning.bins_applier import BinApplier
df = ...  # pandas DataFrame
applier = BinApplier(df)
series_binned = applier.apply("age", [18, 30, 50])