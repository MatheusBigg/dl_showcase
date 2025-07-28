
# Detecção de Objetos Domésticos

Modelo YOLOv9 para detecção em tempo real de objetos domésticos comuns (copos, garrafas, pratos, talheres).

## Métricas de Desempenho
| Métrica               | Valor   | Explicação                     |
|-----------------------|---------|--------------------------------|
| **mAP50-95**          | 0.607   | Precisão média em IoU 50%-95%  |
| **mAP50**             | 0.791   | Precisão média em IoU ≥ 50%    |
| **Precisão**          | 0.772   | Falsos positivos baixos        |
| **Recall**            | 0.722   | Boa cobertura de objetos       |
| **Box Loss (val)**    | 1.027   | Erro de localização aceitável  |

## Análise Comparativa (Evolução do Modelo)
| Versão          | mAP50-95 | Precisão | Observações                     |
|-----------------|----------|----------|---------------------------------|
| YOLOv8 (50ep)   | 0.540    | 0.738    | Baseline inicial                |
| YOLOv9 (100ep)  | 0.581    | 0.726    | +4.1% mAP vs baseline          |
| **YOLOv9 Final**| **0.607**| **0.772**| +6.7% mAP com augmentations     |

**Melhorias-chave:**
- Aumento de 4.5% no mAP50-95 com ajuste de hiperparâmetros
- Redução de 2.5% na loss de detecção (dfl_loss)
- Trade-off otimizado entre precisão e recall

## Como Usar

### Treinamento
```bash
python dl_training/train.py
```