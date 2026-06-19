# Attention-Aware Toy

Protótipo experimental de percepção de atenção para futuras aplicações de
interação humano-robô (Human-Robot Interaction, HRI). O projeto combina visão
computacional e modelos multimodais locais para explorar como um robô social
poderia perceber sinais visuais de engajamento e decidir quando iniciar uma
interação.

Este repositório é um *toy project*: ele serve como ambiente de experimentação,
aprendizado e validação de ideias. Não é um sistema de produção, não realiza
reconhecimento de identidade e não deve ser usado para inferir intenção, emoção
ou estados mentais.

## Objetivo

O fluxo atual usa a webcam para detectar um rosto, estimar pose da cabeça e
direção aproximada das íris, compor um índice de atenção e acompanhar por quanto
tempo a pessoa permanece voltada para a câmera. Quando a atenção se sustenta, o
sistema:

1. captura o quadro atual;
2. descreve a cena com um modelo visual local;
3. gera uma resposta textual curta e contextual;
4. exibe no vídeo os sinais utilizados na estimativa.

A intenção de longo prazo é usar esses experimentos como base para algum projeto
futuro envolvendo HRI, computer vision e comportamentos de robôs sensíveis ao
contexto social.

## Arquitetura experimental

- **Detecção facial e landmarks:** MediaPipe e OpenCV.
- **Estimativa de atenção:** pose da cabeça, direção das íris, posição do nariz,
  abertura dos olhos e posição do rosto no quadro.
- **Descrição visual:** Qwen3-VL executado localmente pelo Ollama.
- **Resposta contextual:** Qwen2.5 executado localmente pelo Ollama.
- **Controle temporal:** duração do olhar, sessões de atenção e cooldown entre
  interações.

O maior rosto detectado é usado como referência. O `attention_score` pondera pose
da cabeça (40%), direção das íris (30%), posição do nariz (15%), simetria e
abertura dos olhos (10%) e posição do rosto (5%). Atenção sustentada é confirmada
quando o índice permanece acima de `0.7` por pelo menos um segundo.

## Requisitos

- Python 3.11 ou superior;
- webcam acessível pelo OpenCV;
- Ollama, opcional, para descrição visual e geração de resposta;
- modelos locais `qwen3-vl:2b-instruct` e `qwen2.5:3b`.

Sem Ollama ou sem os modelos, o protótipo continua executando com mensagens de
fallback, permitindo testar o restante do pipeline.

## Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows, ative o ambiente com `venv\Scripts\activate`.

Instale o Ollama conforme as instruções da sua plataforma e baixe os modelos:

```bash
ollama pull qwen3-vl:2b-instruct
ollama pull qwen2.5:3b
```

O servidor pode ser iniciado manualmente quando necessário:

```bash
ollama serve
```

## Execução

Para validar somente a geração textual:

```bash
python -m src.text_app
```

Para executar o pipeline com webcam:

```bash
python -m src.app
```

Pressione `q` ou `Esc` na janela de vídeo para encerrar. `Ctrl+C` também executa
o fluxo normal de limpeza. Ao sair, o projeto solicita ao Ollama que descarregue
imediatamente os modelos utilizados; o serviço Ollama permanece ativo, mas os
modelos deixam de ocupar RAM em standby.

## Configuração do Ollama

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endereço da API local |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo de resposta textual |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:2b-instruct` | Modelo de descrição visual |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Timeout da resposta textual |
| `OLLAMA_VISION_TIMEOUT_SECONDS` | `60` | Timeout da descrição visual |

Exemplo:

```bash
OLLAMA_MODEL=qwen2.5:7b OLLAMA_TIMEOUT_SECONDS=45 python -m src.text_app
```

## Testes

Com o ambiente virtual ativo:

```bash
python -m unittest discover -s tests -v
```

## Limitações e uso responsável

- O índice de atenção é uma heurística, não uma medida objetiva de atenção.
- Iluminação, câmera, oclusões, óculos e características individuais afetam as
  estimativas.
- Olhar para a câmera não implica interesse, consentimento ou intenção de
  interagir.
- As respostas dos modelos locais podem conter erros.
- O protótipo ainda não possui calibração por usuário nem avaliação científica.

Qualquer evolução para experimentos com pessoas deve incluir consentimento,
privacidade, revisão ética, métricas claras e avaliação dos vieses do sistema.

## Estado do projeto

O repositório está em desenvolvimento exploratório. Interfaces, limiares,
modelos e decisões de arquitetura podem mudar conforme os experimentos avancem.
