# Briefing: CacauFito — pitch de projeto (Aprendizagem Profunda)

**Fonte:** referências locais do deck (texto) — `references/mira_pitch_script.md`, `references/ml_canvas.md`
**Data da extração:** 2026-09-11

## Essência em uma frase

Um pipeline completo de visão computacional (dados públicos → transfer learning com EfficientNet-B0 → otimização de hiperparâmetros com Optuna → API de inferência → frontend) que classifica a condição fitossanitária de uma folha de cacau (sadia, CSSVD ou antracnose) a partir de uma foto, dando ao pequeno produtor uma triagem visual acessível direto do celular.

## Conceitos-chave (candidatos a slide)

1. **Contextualização do problema** — cacauicultores não têm forma acessível de identificar CSSVD/antracnose numa folha; oportunidade de triagem visual pelo celular — sugestão visual: **d3-fluxo** (foto → scan → rótulo).
2. **Dataset (Amini Cocoa Contamination Dataset)** — 5.529 imagens, 3 classes (cssvd 2.237, healthy 1.726, anthracnose 1.566), split 70/15/15 estratificado e disjunto — sugestão visual: **comparacao** (pilhas de fotos se organizando em 3 caixas, depois 3 splits).
3. **Arquitetura da solução (pipeline ponta a ponta)** — dados → treino (com busca de hiperparâmetros) → avaliação → exportação de artefato → API de inferência → frontend — sugestão visual: **d3-fluxo** (esteira com blocos acendendo em sequência).
4. **Arquitetura da rede** — EfficientNet-B0 pré-treinada, backbone congelado + cabeça linear nova (1280→3), entrada 224×224, augmentation (flip, rotação, jitter de cor) — sugestão visual: **d3-fluxo** (diagrama da rede, camadas congeladas vs. treinável).
5. **Regra de incerteza** — predição marcada como incerta por baixa confiança ou "close call" entre top-2 classes — sugestão visual: **comparacao** (confiante vs. incerto).
6. **Transfer learning: por que e como** — dataset pequeno demais para treinar do zero; features do ImageNet reaproveitadas; só a última camada é treinada — sugestão visual: **d3-fluxo** (cadeado fechando sobre as camadas congeladas).
7. **Otimização de hiperparâmetros (Optuna)** — busca de `lr` e `weight_decay`, TPE + MedianPruner, 10 trials (5 completos, 5 pruned), melhor trial `lr=0,00142`/`weight_decay=0,000126` com acc. de validação 0,793 em 4 épocas — sugestão visual: **d3-fluxo** (funil de trials, um brilhando como vencedor).
8. **Benchmark sem × com Optuna** — comparação direta de métricas antes/depois da busca — sugestão visual: **card_metricas** / **comparacao** (barras lado a lado).
9. **Resultados finais (com Optuna)** — acurácia de teste 0,781, F1 por classe, matriz de confusão — sugestão visual: **card_metricas** (matriz de confusão "viva").
10. **Aplicação final (demo)** — upload de foto → resultado (confiante / incerto / erro tratado), sem login — sugestão visual: **screenshots reais** já capturados em `assets/frontend-*.png` (home, confiante healthy, confiante anthracnose, incerto, erro).
11. **Fechamento** — resumo do pipeline completo, limitações reconhecidas, próximos passos — sugestão visual: **d3-fluxo** (blocos do pipeline se reconectando) + QR code para `https://github.com/daniballester-ai/cacau_fito_DL`.

## Dados e números

**Dataset:**
- 5.529 imagens totais; classes: cssvd 2.237, healthy 1.726, anthracnose 1.566
- Split: treino 3.869 / val 830 / teste 830 (70/15/15, estratificado, sem sobreposição)
- 6 imagens com anotação de classe mista, resolvidas por classe majoritária

**Arquitetura:**
- Backbone: EfficientNet-B0 (ImageNet1K_V1), congelado
- Cabeça treinável: `Linear(1280 → 3)`
- Entrada: RGB 224×224, normalização ImageNet
- Augmentation (treino): flip horizontal, rotação ±15°, jitter de cor

**Otimização com Optuna:**
- Espaço de busca: `lr` (log-uniforme, 1e-4–1e-2), `weight_decay` (log-uniforme, 1e-6–1e-3)
- 10 trials configurados, TPE + MedianPruner; 5 completos, 5 pruned
- Melhor trial: `lr = 0,00142`, `weight_decay = 0,000126`, acc. de validação (busca, 4 épocas) = 0,793
- Custo da busca: ~1h40 no Kaggle (bem acima da estimativa inicial de ~5 min, porque cada trial treina sobre o `train_loader` completo)

**Benchmark sem × com Optuna (treino completo, 15 épocas):**

| Métrica | Sem Optuna | Com Optuna | Δ |
|---|---|---|---|
| Acurácia teste | 0,778 | 0,781 | +0,3 p.p. |
| Melhor acc. validação | 0,799 | 0,802 | +0,3 p.p. |
| F1 healthy | 0,765 | 0,768 | +0,3 p.p. |
| F1 cssvd | 0,780 | 0,779 | −0,1 p.p. |
| F1 anthracnose | 0,794 | 0,799 | +0,5 p.p. |
| Recall cssvd | 0,747 | 0,765 | +1,8 p.p. |
| Precisão healthy | 0,701 | 0,718 | +1,7 p.p. |
| Tempo/época (treino final) | ~216s | ~230s | +~6% |

**Resultados finais (com Optuna, versão de referência para o pitch):**
- Acurácia teste: 0,781
- Por classe (precisão/recall/F1): healthy 0,718/0,826/0,768 · cssvd 0,793/0,765/0,779 · anthracnose 0,851/0,753/0,799
- Confusão principal: cssvd × healthy (58 casos)
- Todas as classes acima do piso mínimo de 50 amostras de teste (sem aviso de baixa confiabilidade)

## Trechos de código emblemáticos

Não há código-fonte diretamente nas referências deste deck (o roteiro descreve o notebook em prosa); se quiser um slide com código, os trechos-chave estão em `notebooks/train-kaggle-v2.ipynb` (célula do `objective(trial)` do Optuna e da montagem do modelo EfficientNet-B0) — não copiados aqui para manter o briefing enxuto.

## Narrativa sugerida

Arco de 8 blocos, já com timing sugerido pelo roteiro-fonte (~10 min total):
1. Contextualização (1:30) → 2. Dados (1:30) → 3. Arquitetura rede+pipeline (2:30) → 4. Transfer learning (1:00) → 5. Otimização/Optuna (1:15) → 6. Resultados + benchmark sem×com Optuna (1:45) → 7. Demo da aplicação (1:00) → 8. Fechamento (0:30).

Esse arco já é problema → solução → como funciona → validação/resultados → prova ao vivo → CTA, adequado ao pipeline padrão do Mira.

## Lacunas (resolvidas)

- **Screenshots reais da aplicação**: capturados rodando o app localmente (`python -m uvicorn src.inference_service.main:app`) com Playwright, cobrindo os 3 casos exigidos pelo enunciado da disciplina (confiante, incerto, erro tratado). Arquivos em `assets/`:
  - `frontend-home.png` — tela inicial (upload vazio)
  - `frontend-confiante-healthy.png` — sample "Sadia", 87,1% de confiança
  - `frontend-confiante-anthracnose.png` — sample "Antracnose", 95,0% de confiança
  - `frontend-incerto.png` — `samples/uncertain_cssvd_1.jpg`, badge "Resultado incerto" (CSSVD 42,1% vs. Antracnose 41,2%, close call)
  - `frontend-erro.png` — upload de arquivo `.txt`, mensagem "Unsupported file type 'text/plain'. Supported: JPEG, PNG, WEBP."
- **Link do repositório**: `https://github.com/daniballester-ai/cacau_fito_DL` — usar no `/mira-qrcode` do slide de fechamento.
