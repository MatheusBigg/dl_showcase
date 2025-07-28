# Deep Learning Showcase

Repositório contendo diversos modelos de deep learning para diferentes aplicações. Cada modelo é implementado como um módulo independente com seu próprio conjunto de dados, treinamento e interface de inferência.

## Modelos Disponíveis

### 1. Detecção de Objetos Domésticos
- Detecção em tempo real de objetos comuns em ambientes residenciais
- Técnica: YOLOv9 com transfer learning
- **Métricas**: mAP50-95: 60.7%, Precisão: 77.2%, Recall: 72.2%
- [Detalhes do modelo](dl_models/dl_house_objects/README.md)

### 2. [Próximo Modelo - EM BREVE]
- [Descrição breve]
- [Link para detalhes]

## Estrutura do Projeto

```
dl_showcase/
├── dl_models/ # Pasta de modelos
│ ├── dl_house_objects/ # Detecção de objetos domésticos
│ └── ... # Outros modelos
├── app/ # API e backend
├── shared/ # Recursos compartilhados
│ ├── mlflow/ # Experiment tracking
│ └── runs/ # Resultados de treinos
└── tests/ # Testes unitários
```


## Como Executar
1. Clone o repositório:
```bash
git clone https://github.com/MatheusBigg/dl_showcase.git
cd dl_showcase
```

2. Configure o ambiente:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

3. Execute o modelo de detecção de objetos domésticos:
```bash
python dl_models/dl_house_objects/main.py
```