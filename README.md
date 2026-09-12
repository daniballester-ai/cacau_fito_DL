# CacauFito

Prova de conceito de visão computacional que classifica a condição fitossanitária de uma folha de cacau — **sadia**, **CSSVD** (vírus do inchaço do broto) ou **antracnose** — a partir de uma foto, com uma API de inferência e um frontend de upload por trás.

Repositório de entrega da disciplina **PPGTI3003 — Aprendizagem Profunda** (PPgTI/UFRN).

## Entrega da disciplina

A atividade pede quatro itens. Aqui estão os quatro, direto:

| # | Item exigido | Onde está |
|---|---|---|
| 1 | ML/Data project canvas | [`docs/ml_canvas.html`](docs/ml_canvas.html)  |
| 2 | Notebook com passo a passo (markdown + código) | [`notebooks/train-kaggle-v2.ipynb`](notebooks/train-kaggle-v2.ipynb), rodado no Kaggle: [kaggle.com/code/danielleballester/train-kaggle-optuna](https://www.kaggle.com/code/danielleballester/train-kaggle-optuna/) |
| 3 | Link do repositório | [github.com/daniballester-ai/cacau_fito_DL](https://github.com/daniballester-ai/cacau_fito_DL) (este repositório) |
| 4 | Vídeo de apresentação (pitch, até 10 min) | [`video/cacaufito-pitch.mp4`](video/cacaufito-pitch.mp4), narrado, cobrindo requisitos, casos de teste e arquitetura (rede + pipeline) [![Assistir no YouTube](https://img.youtube.com/vi/5YuKhI_38p4/maxresdefault.jpg)](https://www.youtube.com/watch?v=5YuKhI_38p4) |


## Problema e tarefa de ML

Cacauicultores não têm hoje uma forma acessível de identificar pragas e doenças foliares do cacaueiro a partir de uma foto. Duas doenças graves, **CSSVD** e **antracnose**, reduzem produtividade e se espalham se não identificadas cedo. A tarefa é **classificação de imagem supervisionada**, com 3 classes mutuamente exclusivas (`healthy`, `cssvd`, `anthracnose`), a partir de uma foto de folha de cacau tirada em campo. Detalhes completos (fontes de dados, target, produtos de dados, direcionadores de pitch) estão no [ML Canvas](docs/ml_canvas.html).

## Arquitetura (rede + pipeline)

```text
┌──────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────┐    ┌─────┐    ┌─────┐
│  Dados   │ -> │ Treino (+ busca  │ -> │ Avaliação  │ -> │ Artefato │ -> │ API │ -> │ App │
│ (Kaggle) │    │ Optuna)          │    │ (teste)    │    │ (.pt)    │    │     │    │     │
└──────────┘    └──────────────────┘    └────────────┘    └──────────┘    └─────┘    └─────┘
```

- **Dados**: [Amini Cocoa Contamination Dataset](https://www.kaggle.com/datasets/ohagwucollinspatrick/amini-cocoa-contamination-dataset) (Kaggle, CC BY 4.0), 5.529 imagens rotuladas, split 70/15/15 estratificado (treino/validação/teste), sem sobreposição entre conjuntos.
- **Rede**: **EfficientNet-B0** pré-treinada (ImageNet), com o backbone convolucional **congelado**; só a camada final (`Linear(1280 → 3)`) é treinada, transfer learning apropriado para um dataset de porte médio e uma GPU única.
- **Otimização de hiperparâmetros**: busca sistemática com **Optuna** (TPE + `MedianPruner`) sobre `learning rate` e `weight_decay`, em vez de valores fixos escolhidos manualmente. Ver "Resultados" abaixo para o ganho medido.
- **Avaliação**: métricas por classe (precisão/recall/F1) no conjunto de teste, com sinalização explícita quando uma classe tem poucas amostras de teste (piso de confiabilidade documentado).
- **Serviço de inferência**: FastAPI (`src/inference_service/`) carrega o artefato do modelo e expõe `POST /predict`, retornando o rótulo previsto, a confiança, e uma **flag de incerteza** (`is_uncertain`) quando a confiança é baixa ou há disputa acirrada entre as duas classes mais prováveis.
- **Frontend**: página de upload sem login (`frontend/`), que mostra o resultado e destaca visualmente quando ele é incerto.

## Resultados

| Métrica | Sem Optuna (`train_kaggle_v1.ipynb`) | Com Optuna (`train-kaggle-v2.ipynb`, versão final) | Δ |
|---|---|---|---|
| Acurácia no teste | 0,778 | **0,781** | +0,3 p.p. |
| Melhor acurácia de validação | 0,799 | 0,802 | +0,3 p.p. |
| F1 healthy | 0,765 | 0,768 | +0,3 p.p. |
| F1 cssvd | 0,780 | 0,779 | −0,1 p.p. |
| F1 anthracnose | 0,794 | 0,799 | +0,5 p.p. |
| Recall cssvd | 0,747 | 0,765 | +1,8 p.p. |
| Precisão healthy | 0,701 | 0,718 | +1,7 p.p. |

A busca do Optuna (10 trials, 5 completos + 5 interrompidos pelo pruner) encontrou `lr = 0,00142` e `weight_decay = 0,000126` como melhor combinação, próxima do valor manual original (`lr = 0,001`), o que valida a escolha inicial com evidência em vez de sorte. Custo: ~1h40 adicionais de GPU no Kaggle.

**Limitação reconhecida**: dataset pequeno e informal, coletado por terceiros (não é do Sul da Bahia); é um PoC, não um produto validado em campo.

## Requisitos e casos de teste cobertos

A regra de incerteza da API (`is_uncertain` / `uncertainty_reason`) e as demais funcionalidades novas (histórico de predições, estatísticas agregadas) são cobertas por testes automatizados em `tests/`:

- `test_uncertainty.py`: confiança alta não é sinalizada; confiança baixa é (`low_confidence`); disputa acirrada entre as duas classes mais prováveis é (`close_call`); os dois critérios podem coexistir.
- `test_history.py` / `test_history_api.py` / `test_predict_history_integration.py`: gravação e leitura do histórico, retenção com expurgo do mais antigo, paginação, ordenação mais-recente-primeiro, histórico vazio não gera erro, falha ao gravar histórico não derruba `/predict`.
- `test_stats.py` / `test_stats_api.py`: contagem por classe, distribuição desbalanceada, histórico vazio retorna zeros, e resposta 503 quando o armazenamento falha.

Os mesmos três casos de uso ponta a ponta aparecem demonstrados com a aplicação real rodando no vídeo do pitch ("Uma foto, um diagnóstico confiante", "E quando o modelo está em dúvida?" e "Arquivo errado? Mensagem clara"): resultado confiante, resultado incerto sinalizado visualmente, e erro de entrada tratado sem quebrar a aplicação.

## Como rodar

```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart torch torchvision pillow
python -m uvicorn src.inference_service.main:app --reload
```

Abra `http://127.0.0.1:8000` no navegador. Imagens de exemplo (uma por classe, mais um caso incerto) em [`samples/`](samples/).

## Testes

```bash
python -m pip install pytest
python -m pytest tests/ -v
```

## Estrutura

```text
notebooks/                 notebooks de treino (Kaggle): v1 (baseline) e v2 (com Optuna, final)
src/inference_service/     API (predict, history, stats)
frontend/                  upload + histórico
models/                    modelo treinado + mapeamento de classes + relatório de avaliação
samples/                   imagens de exemplo (uma por classe + caso incerto)
video/                     vídeo do pitch (entrega item 4)
docs/                      ML/Data project canvas (entrega item 1)
tests/                     suíte pytest
```

## Autoria

Danielle Magalhães Ballester ([danielleballester@gmail.com](mailto:danielleballester@gmail.com)), trabalho de PPGTI/UFRN.
