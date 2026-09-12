# ML/Data Project Canvas — CacauFito

Baseado no [Machine Learning Canvas](https://www.ownml.co/machine-learning-canvas).

## 1. Qual o problema a ser resolvido?

Cacauicultores não têm uma forma acessível de identificar pragas e doenças foliares do cacaueiro a partir de uma foto. A solução proposta usando visão computacional é um modelo que classifica a condição fitossanitária de uma folha de cacau — sadia, CSSVD ou antracnose.

## 2. Fontes dos Dados

- **Amini Cocoa Contamination Dataset** — imagens de folha de cacau em campo, licença CC BY 4.0, hospedado no Kaggle.
- **iNaturalist.org** — fonte complementar de fotos públicas de *Theobroma cacao*, para ampliar diversidade se necessário.

## 3. Target

`class` — condição da folha, 3 categorias mutuamente exclusivas por imagem: **sadia**, **CSSVD** (vírus do inchaço do broto) e **antracnose**. O modelo aprende a mapear a imagem da folha à classe mais provável.

## 4. Principais Atributos (variáveis)

Diferente de um problema tabular, o atributo de entrada é a **imagem inteira** da folha — a rede convolucional extrai automaticamente os padrões visuais relevantes (textura, cor, lesões).

**Mais concreto / fácil de obter:** foto da folha (câmera de celular, em campo), data e local (metadados da captura).
**Mais concreto / difícil de obter:** dataset já rotulado (depende de terceiros disponibilizarem).
**Mais subjetivo / fácil de obter:** variedade do cacaueiro (nem sempre identificável na foto).
**Mais subjetivo / difícil de obter:** condição correta da folha (requer conhecimento fitopatológico), estágio/severidade da doença.

## 5. Produtos de Dados

Modelo de classificação de imagem servido por uma API de inferência, que recebe a foto de uma folha e retorna a condição prevista com grau de confiança. Um frontend simples permite o envio da foto e a visualização do resultado, sem necessidade de login.

## 6. Direcionadores PITCH

- **Oportunidade:** triagem visual acessível de condição fitossanitária, direto do celular do produtor.
- **O que está buscando:** validar se transfer learning com dados públicos já é suficiente para distinguir as classes.
- **Vantagem do produto de dados:** pipeline desacoplado (dados → treino → inferência → frontend), pronto para trocar por dados do Sul da Bahia no futuro.
