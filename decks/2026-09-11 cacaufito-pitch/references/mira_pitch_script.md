# Roteiro do Pitch — CacauFito (para Mira Animator)

> Fonte de conteúdo para o pipeline do Mira (`/mira-new` → `/mira-extract` → `/mira-planner` → `/mira-copywriter` → `/mira-builder` → `/mira-animator` → `/mira-validator`, ou `/mira-fast` para gerar tudo de uma vez).
> Formato alvo: vídeo/apresentação estilo pitch, **máximo 10 minutos**, cobrindo requisitos, casos de teste e arquitetura (rede + pipeline), conforme exigido na entrega da disciplina PPGTI3003.
> Tema sugerido: paleta verde/marrom (cacau/folha), objetos animados em loop, analogias concretas do cotidiano para conceitos técnicos.

---

## Timing sugerido (10 min)

| Bloco | Tempo | Slides (aprox.) |
|---|---|---|
| 1. Contextualização | 1:30 | 2-3 |
| 2. Dados | 1:30 | 2 |
| 3. Arquitetura (rede + pipeline) | 2:30 | 3-4 |
| 4. Transfer learning | 1:00 | 1-2 |
| 5. Otimização (Optuna) | 1:15 | 2 |
| 6. Resultados/métricas + benchmark sem×com Optuna | 1:45 | 3-4 |
| 7. Aplicação final (demo) | 1:00 | 1-2 |
| 8. Fechamento | 0:30 | 1 |

---

## Bloco 1 — Contextualização do problema

**Mensagem central:** cacauicultores não têm hoje uma forma acessível de identificar doenças foliares do cacaueiro a partir de uma foto.

- **Problema de negócio:** duas doenças/pragas foliares graves do cacaueiro — **CSSVD** (vírus do inchaço do broto do cacau) e **antracnose** — reduzem produtividade e, se não identificadas cedo, se espalham pela plantação.
- **Quem sofre com isso:** o pequeno/médio produtor de cacau, que não tem acesso fácil a um agrônomo/fitopatologista para cada folha suspeita.
- **Oportunidade (ML canvas, direcionador PITCH):** triagem visual acessível, direto do celular do produtor — tirar uma foto da folha e saber, na hora, se ela está sadia, com CSSVD ou com antracnose.
- **O que estamos validando:** se transfer learning com dados públicos (sem coleta própria em campo) já é suficiente para distinguir essas 3 classes.

**Sugestão de objeto animado:** ilustração de uma folha de cacau sendo fotografada por um celular, com um "raio-x"/scan animado revelando o rótulo (sadia / CSSVD / antracnose) — metáfora concreta de "diagnóstico instantâneo".

---

## Bloco 2 — Dados (dataset)

- **Fonte:** Amini Cocoa Contamination Dataset (Kaggle), licença **CC BY 4.0** — imagens de folhas de cacau em campo.
- **Volume:** 5.529 imagens rotuladas (`Train.csv` com anotações por bounding box, resolvidas para rótulo único por imagem pela classe majoritária — apenas 6 imagens tinham classes mistas).
- **Distribuição de classes:** cssvd (2.237), healthy (1.726), anthracnose (1.566) — moderadamente desbalanceado.
- **Split:** 70% treino / 15% validação / 15% teste, estratificado por classe, com verificação de disjunção entre os conjuntos (nenhuma imagem repetida entre splits).
- **Por que não há gabarito de teste oficial:** o `Test.csv` original é de uma competição Zindi/Amini sem rótulos públicos — por isso criamos nosso próprio split de teste a partir do `Train.csv`.

**Sugestão de objeto animado:** três "pilhas" de fotos de folhas se organizando em caixas rotuladas (sadia/cssvd/antracnose), depois se dividindo em três grupos menores (treino/val/teste) — metáfora de organização/triagem.

---

## Bloco 3 — Arquitetura da solução (rede + pipeline)

**Mensagem central:** a solução é um pipeline desacoplado — dados → treino → inferência → frontend — não só um modelo isolado.

### 3.1 Pipeline de ponta a ponta
1. **Preparação dos dados** — download, manifesto, split (Kaggle notebook, seções 1-4).
2. **Treino do modelo** — transfer learning + otimização de hiperparâmetros (seções 5-7 do notebook).
3. **Avaliação** — métricas no conjunto de teste, com aviso de confiabilidade por classe (seção 8).
4. **Exportação do artefato** — pesos (`.pt`), mapeamento de classes (`.json`) e relatório de avaliação (`.json`) (seção 9).
5. **Serviço de inferência** — API que carrega o artefato e expõe um endpoint de predição (recebe imagem → devolve label + confiança + flag de incerteza).
6. **Frontend** — página simples de upload, sem login, que mostra o resultado e destaca visualmente quando a predição é incerta.

### 3.2 Arquitetura da rede
- **Backbone:** EfficientNet-B0 pré-treinada (ImageNet).
- **Estratégia:** backbone **congelado** (features fixas) + nova cabeça linear (`Linear(1280 → 3)`) treinada do zero para as 3 classes do problema.
- **Entrada:** imagem RGB 224×224, normalizada com média/desvio do ImageNet.
- **Data augmentation no treino:** flip horizontal, rotação (±15°), jitter de cor (brilho/contraste/saturação) — reduz overfitting num dataset de porte médio.

**Sugestão de objeto animado:** diagrama 3D/interativo (via `/mira-3d`) do pipeline como uma esteira: foto entra → bloco "dados" → bloco "rede congelada + cabeça treinável" → bloco "avaliação" → bloco "API" → bloco "app". Cada bloco pode "acender" em sequência (`/mira-sequence-director`) conforme a narração passa por ele.

### 3.3 Regra de incerteza (explicabilidade mínima)
- Predição é marcada como **incerta** quando: (a) a confiança do top-1 está abaixo de um limiar documentado, ou (b) a diferença entre a 1ª e a 2ª classe mais prováveis é pequena ("close call").
- Isso evita apresentar ao usuário uma resposta confiante quando o modelo está, na prática, em dúvida entre duas classes.

---

## Bloco 4 — Transfer learning: por que e como

- **Por que transfer learning:** o dataset (5.529 imagens) é pequeno para treinar uma CNN do zero; usar uma rede pré-treinada no ImageNet aproveita features visuais genéricas (bordas, texturas, formas) já aprendidas em milhões de imagens.
- **O que foi congelado vs. treinado:** todas as camadas convolucionais (`model.features`) ficam congeladas (`requires_grad = False`); apenas a camada final do classificador é treinada — reduz drasticamente o número de parâmetros treináveis e o tempo de treino, adequado ao volume de dados disponível.
- **Trade-off:** menos flexível que fine-tuning completo, mas menos propenso a overfitting e muito mais rápido de treinar numa GPU T4 do Kaggle.

**Sugestão de objeto animado:** ícone de "cérebro" pré-treinado (ImageNet) sendo "transplantado" para o problema do cacau, com um cadeado fechando sobre as camadas convolucionais (congeladas) e só a última camada piscando/"acesa" (treinável).

---

## Bloco 5 — Otimização de hiperparâmetros (Optuna)

- **Motivação:** em vez de escolher `learning rate` e `weight decay` manualmente ("chute"), usamos uma busca sistemática com **Optuna**.
- **Como funciona a busca:** cada *trial* treina uma cabeça de classificação nova (mesmo backbone congelado) por poucas épocas (4) e mede a acurácia de validação — uma versão "rápida e barata" do treino completo, usada só para ranquear combinações de hiperparâmetros.
- **Espaço de busca:** `lr` (escala log, 1e-4 a 1e-2) e `weight_decay` (escala log, 1e-6 a 1e-3).
- **Amostrador/estratégia:** TPE (padrão do Optuna) com `MedianPruner`, que interrompe trials claramente ruins antes de terminar — economiza tempo de GPU.
- **Quantidade de trials:** 10 configurados — 5 rodaram até o fim e 5 foram interrompidos (*pruned*) pelo `MedianPruner` por estarem abaixo da mediana das trials anteriores.
- **Resultado da busca (execução real no Kaggle):**
  - Melhor combinação: **`lr = 0,00142`**, **`weight_decay = 0,000126`** (trial 3 de 10).
  - Acurácia de validação da busca (só 4 épocas): **0,793**.
  - Trials completos variaram de 0,743 a 0,793 de acurácia — mostrando que a escolha de `lr` sozinha já muda o resultado em ~5 pontos percentuais, mesmo num treino curto.
- **Resultado usado no treino final:** o `lr`/`weight_decay` do melhor trial alimentaram diretamente o otimizador do treino completo de 15 épocas, chegando a **0,802** de acurácia de validação (melhor época) — ver benchmark completo no Bloco 6.
- **Honestidade sobre custo:** a busca em si levou **~1h40 no Kaggle** (bem mais que os ~5 min inicialmente estimados), porque cada trial treina sobre o `train_loader` completo — um trade-off consciente de tempo de GPU por uma escolha de hiperparâmetros mais confiável que "chutar".

**Sugestão de objeto animado:** várias "bolinhas" (trials) caindo em um gráfico/funil e sendo filtradas, com uma delas brilhando ao final como "vencedora" — metáfora de busca/seleção.

---

## Bloco 6 — Resultados e métricas (com Optuna, versão final)

- **Acurácia geral (teste):** **0,781** (78,1%) — `notebooks/train-kaggle-v2.ipynb`.
- **Por classe (precisão / recall / F1):**
  - healthy: 0,718 / 0,826 / 0,768
  - cssvd: 0,793 / 0,765 / 0,779
  - anthracnose: 0,851 / 0,753 / 0,799
- **Matriz de confusão:** maior confusão ainda entre *cssvd* e *healthy* (58 casos), esperado dado que sintomas iniciais de CSSVD podem ser sutis; a rede acerta bem *anthracnose* (177/235).
- **Confiabilidade:** todas as 3 classes têm mais de 50 amostras no conjunto de teste (piso mínimo documentado) → nenhum aviso de baixa confiabilidade disparado.
- **Como interpretar para a banca:** o objetivo do PoC não é produção, é validar a viabilidade da abordagem (transfer learning + dataset público) — os números confirmam que o sinal visual das doenças é aprendível pela rede, mesmo sem dados próprios de campo.

**Sugestão de objeto animado:** matriz de confusão "viva" onde os números aparecem crescendo (contadores animados) e a diagonal principal pisca em verde (acertos) enquanto o resto pisca em âmbar (erros).

---

## Bloco 6.1 — Benchmark: sem Optuna × com Optuna

**Mensagem central:** a otimização sistemática de hiperparâmetros trouxe uma melhora pequena, porém consistente, em quase todas as métricas — validando que vale a pena buscar em vez de "chutar" `lr`/`weight_decay`.

| Métrica | Sem Optuna (`train_kaggle_v1.ipynb`, `lr=1e-3` fixo) | Com Optuna (`train-kaggle-v2.ipynb`, `lr` e `weight_decay` buscados) | Δ |
|---|---|---|---|
| Acurácia no teste | 0,778 | 0,781 | +0,3 p.p. |
| Melhor acurácia de validação (treino completo) | 0,799 | 0,802 | +0,3 p.p. |
| F1 healthy | 0,765 | 0,768 | +0,3 p.p. |
| F1 cssvd | 0,780 | 0,779 | −0,1 p.p. |
| F1 anthracnose | 0,794 | 0,799 | +0,5 p.p. |
| Recall cssvd | 0,747 | 0,765 | +1,8 p.p. |
| Precisão healthy | 0,701 | 0,718 | +1,7 p.p. |
| Tempo por época (treino final) | ~216 s | ~230 s | +~6% (leve overhead do `weight_decay`) |
| Custo extra da busca | — | ~10 trials × 4 épocas ≈ 1h40 no Kaggle | + tempo de GPU |

**Leituras principais para o pitch:**
- A busca **não** encontrou um `lr` radicalmente diferente do valor manual original (`0,00142` vs `0,001` chutado) — o que já é uma validação interessante: a escolha manual inicial estava próxima do ótimo, e o Optuna confirma isso com evidência em vez de sorte.
- O ganho maior aparece no **recall de cssvd** (+1,8 p.p.) e na **precisão de healthy** (+1,7 p.p.) — a classe mais crítica de negócio (detectar CSSVD cedo) melhorou, mesmo que a acurácia geral tenha subido pouco.
- **Trade-off honesto:** a busca custou ~1h40 de GPU adicional para um ganho de +0,3 p.p. de acurácia — vale a pena mostrar essa análise de custo-benefício na apresentação como exemplo de pensamento crítico sobre otimização, não só "rodei o Optuna e melhorou".

**Sugestão de objeto animado:** duas barras lado a lado (sem Optuna / com Optuna) crescendo até seus valores finais, com um selo/etiqueta "+0,3 p.p." aparecendo entre elas; opcionalmente, um pequeno "relógio" animado ao lado mostrando o custo de tempo extra da busca.

---

## Bloco 7 — Aplicação final (demo)

- **Frontend:** página simples (`frontend/index.html`), sem necessidade de login — upload de uma foto da folha.
- **Fluxo da demo ao vivo:**
  1. Abrir a página do CacauFito.
  2. Fazer upload de uma foto de folha (idealmente uma de cada classe, para mostrar os 3 rótulos possíveis).
  3. Mostrar o resultado: label previsto + score de confiança.
  4. Mostrar um caso **incerto** (foto ambígua ou de baixa confiança) e destacar como a interface sinaliza isso visualmente ("resultado incerto"), em vez de fingir certeza.
  5. Mostrar um caso de **erro tratado** (ex.: arquivo que não é imagem) e a mensagem clara exibida ao usuário, sem quebrar a aplicação.
- **Por trás da tela:** a página chama a API de inferência, que carrega o artefato do modelo treinado (pesos + mapeamento de classes) gerado pelo notebook — fechando o ciclo notebook → modelo → produto.

**Sugestão de objeto animado:** um "screencast" real da aplicação (gravado depois que o Kaggle terminar) embutido no slide, ou, se preferir animação pura, um mockup de celular com a tela de upload → resultado, com transição animada entre os 3 estados (confiante / incerto / erro).

---

## Bloco 8 — Fechamento

- **Resumo em uma frase:** um pipeline completo — dados públicos, transfer learning, busca de hiperparâmetros com Optuna, API de inferência e frontend — mostrando que é viável oferecer triagem visual acessível de doenças do cacaueiro a partir de uma simples foto.
- **Limitações reconhecidas (transparência):** dataset público (não é do Sul da Bahia), PoC sem validação em campo, backbone congelado (não fine-tuned).
- **Próximos passos (se houver tempo):** fine-tuning parcial do backbone, coleta de dados regionais, monitoramento pós-deploy.

**Sugestão de objeto animado:** os blocos do pipeline (Bloco 3) reaparecem todos juntos e "conectados" com uma linha animada, fechando o ciclo visualmente.

---

## Notas de produção para o Mira

- Usar `/mira-chart` para os gráficos de métricas (matriz de confusão, precisão/recall por classe) em vez de imagens estáticas, se o pipeline suportar dados tabulares simples.
- Usar `/mira-3d` para o diagrama do pipeline (Bloco 3) e `/mira-sequence-director` para coreografar a sequência de blocos acendendo.
- Usar `/mira-qrcode` no slide de fechamento apontando para o repositório GitHub/GitLab do projeto.
- Formato: 16:9 para vídeo de pitch.
- Depois de gerado, gravar a narração em português e exportar com `/mira-slide-to-video` (ou gravar tela + narração externamente), respeitando o limite de 10 minutos.
