# model-track-cr

O **model-track-cr** é uma biblioteca Python voltada para **binning, WOE, estabilidade e monitoramento de variáveis** em modelos de Machine Learning, com foco em **modelos de crédito e risco**.

O projeto foi construído seguindo rigorosamente **Test-Driven Development (TDD)**, garantindo:
- qualidade de código
- segurança para refatorações
- documentação viva através dos testes

---

## 📦 Estrutura do Projeto
```bash
.
├── Makefile
├── README.md
├── exemplo_uso.ipynb
├── poetry.lock
├── pyproject.toml
├── pytest.ini
├── src
│   └── model_track
│       ├── binning
│       ├── encoding
│       ├── stability
│       ├── stats
│       └── woe
├── tests
│   ├── conftest.py
│   ├── test_bin_applier.py
│   ├── test_tree_binning.py
│   ├── test_quantile_binning.py
│   ├── test_summary.py
│   └── test_woe.py
└── uv.lock
````

---

## 🧰 Ferramentas Utilizadas

- **Poetry** — gestão de dependências e versionamento
- **pytest** — testes automatizados
- **pytest-cov / coverage** — cobertura de código
- **Makefile** — automação de rotinas
- **GitHub Actions** — CI/CD
- **Git Flow** — fluxo de desenvolvimento e release

---
## 🚀 Instalação


Clone o repositório:

```bash
git clone https://github.com/SEU_USUARIO/model-track-cr.git
cd model-track-cr
```
Instale as dependências:

```bash
pip install -e.
poetry install
```
Ou via Makefile:
```bash
make install
```


🧪 Testes e Qualidade

Rodar testes:
```bash
make test
```
Rodar testes com cobertura:
```bash
make cov
```
O relatório HTML ficará disponível em:


`htmlcov/index.html`




🛠 Desenvolvimento (TDD)

1️⃣ Ativar ambiente virtual
```bash
poetry shell
````

2️⃣ Fluxo TDD recomendado

1. Criar ou atualizar um teste em tests/
2. Rodar:
```bash
make test
```

3.	Implementar o código mínimo para passar
4.	Refatorar com segurança
5.	Validar cobertura:
```bash
make cov
```




🧩 Fixtures Globais

Fixtures compartilhadas devem ficar em:

`tests/conftest.py`

O `pytest` carrega esse arquivo automaticamente.

---
## 🤝 Como Contribuir (Git Flow)

🔹 Regras Importantes
*	❌ Não é permitido push direto na main
*	✅ Toda mudança passa por Pull Request
*	✅ CI deve estar verde
*	✅ Testes obrigatórios
*	✅ TDD é mandatório



📊 Comportamento da CI

| Evento                | Branch   | Testes | Publish |
|-----------------------|----------|--------|---------|
| Pull Request          | develop  | Sim    | Não     |
| Pull Request          | main     | Sim    | Não     |
| Push                  | develop  | Sim    | Não     |
| Push                  | main     | Sim    | Não     |
| Push de tag `vX.Y.Z`  | main     | Sim    | Sim     |



🌳 Git Flow — Estrutura de Branches

| Branch      | Como criar (git) | Quando usar | Merge com |
|-------------|------------------|-------------|-----------|
| main        | —                | Produção / release | release/* |
| develop     | —                | Base do desenvolvimento | feature/*, fix/* |
| feature/*   | git checkout develop<br>git pull<br>git checkout -b feature/nome | Nova funcionalidade | develop |
| fix/*       | git checkout develop<br>git pull<br>git checkout -b fix/nome | Correção pontual | develop |
| release/*   | git checkout develop<br>git pull<br>git checkout -b release/x.y.z | Preparar release | main |




1️⃣ Criar branch a partir da develop

```bash
git checkout develop
git pull origin develop
git checkout -b feature/nome-da-feature
```
Ou para correções:
```bash
git checkout -b fix/nome-do-fix
```

2️⃣ Desenvolver seguindo TDD

```bash
make test
make cov
```


3️⃣ Commitar mudanças

```bash
git add .
git commit -m "feat: descrição clara da mudança"
```



4️⃣ Push da branch
```bash
git push origin feature/nome-da-feature
```

5️⃣ Abrir Pull Request

O PR deve conter:
*	descrição clara
*	motivação
*	exemplos de uso (se aplicável)

O PR só será aceito se:
*	CI passar
*	cobertura mínima for respeitada
*	arquitetura estiver consistente



**Criar tag e publicar**

```bash
git checkout main
git pull origin main
poetry version patch 
# vai exibir a nova versao vx.x.xxx
git tag vx.x.xxx
git push origin vx.x.xxx
```
👉 A GitHub Action de publish será disparada automaticamente
👉 O pacote será publicado no PyPI


## Exemplo de uso

### Imports
```python
from model_track.binning import (
    BinApplier,
    TreeBinner,
    QuantileBinner
)

from model_track.woe import (
    WoeCalculator,
    WoeByPeriod
)
from model_track.stats import (
    get_summary
)
from model_track.stability.woe import (
    WoeStability
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Gerar dados para exemplo
```python
np.random.seed(42)

n_per_period = 50
periods = ["2024-01", "2024-02", "2024-03", "2024-04"]

rows = []

for period in periods:
    idade = np.random.normal(loc=40, scale=12, size=n_per_period).clip(18, 75)
    renda = np.random.lognormal(mean=8.2, sigma=0.5, size=n_per_period)

    # ----- introduzindo missing -----
    idade[np.random.rand(n_per_period) < 0.05] = np.nan
    renda[np.random.rand(n_per_period) < 0.05] = np.nan

    # ----- probabilidade do evento (default, fraude, etc.) -----
    prob_event = (
        0.15
        + 0.002 * (idade < 25)
        + 0.003 * (idade > 60)
        + 0.004 * (renda < 2500)
    )

    prob_event = np.clip(prob_event, 0.02, 0.7)

    vr = np.random.binomial(1, prob_event)

    rows.append(
        pd.DataFrame(
            {
                "idade": idade,
                "renda": renda,
                "vr": vr,
                "period": period,
            }
        )
    )

df = pd.concat(rows, ignore_index=True)
df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
# ----- Summary inicial -----
df_summary = get_summary(df=df)
df_summary
```

### Categorização
```python
target = "vr"

binner = TreeBinner(
    max_depth=2,
    min_samples_leaf=1
)
binner.fit(df, feature='renda', target=target)
bins = binner.bins_
bins = [round(b, 2) for b in bins]
applier = BinApplier(df)
df[f"{'renda'}_cat"] = applier.apply('renda', bins)

binner = QuantileBinner(n_bins=3)
binner.fit(df, feature='idade')
bins = binner.bins_
bins = [round(b, 2) for b in bins]
applier = BinApplier(df)
df[f"{'idade'}_cat"] = applier.apply('idade', bins)

# ----- Tratamento explícito de missing pós-binning -----
for feature in features:
    df[f"{feature}_cat"] = (
        df[f"{feature}_cat"]
        .astype("object")
        .fillna("N/A")
    )

# ----- Summary após binning -----
get_summary(df=df)
````
### Calcular woe e iv

```python

woe_tables = {}

for feature in features:
    woe_table = WoeCalculator.compute_table(
        df=df,
        target_col=target,
        feature_col=f"{feature}_cat",
        event_value=1,
        add_totals=True,
    )



    print(f"\nWOE / IV — {feature.upper()}")
    display(woe_table)


# %%
# ----- Exemplo de mapeamento WOE -----
woe_mapping_renda = WoeCalculator.compute_mapping(
    df=df,
    target_col=target,
    feature_col="renda_cat",
)

woe_mapping_renda

# %%
df_result = WoeByPeriod.compute(
        df=df,
        target_col='vr',
        feature_col="renda_cat",
        date_col="period",
    )

df_result
```
### Verificar estabilidade do woe

```python
ws = WoeStability(df=df, date_col="period")

global_woe = ws.global_table(
    feature_col="renda_cat",
    target_col="vr",
)


# tabela global
global_woe = ws.global_table(
    feature_col="renda_cat",
    target_col="vr",
)
display(global_woe)
# gráfico em subplot existente
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ws.generate_view(
    feature_col="idade_cat",
    target_col="vr",
    ax=ax,
)

```

📚 Roadmap (em evolução)
*	Estabilidade de WOE por safra
*	PSI automático
*	Seleção de variáveis por estabilidade
*	CLI para análises rápidas
*	Integração com pipelines de crédito
*	Relatórios automáticos


📝 Licença

MIT
