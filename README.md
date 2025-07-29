# Sistema de Visão Computacional para Ambiente Industrial

## Visão Geral da Solução

Este projeto implementa uma solução completa de visão computacional para detectar objetos industriais (paletes, caixas, empilhadeiras) e extrair informações de QR codes em imagens de armazéns. A arquitetura utiliza processamento assíncrono com YOLOv8 customizado e pipeline otimizado para decodificação de QR codes.

## Stack Tecnológico

- **FastAPI** para API REST com documentação automática
- **YOLOv8** (Ultralytics) para detecção de objetos
- **Celery** para processamento assíncrono
- **RabbitMQ** como message broker 
- **Redis** para cache de resultados
- **PostgreSQL** para persistência de dados
- **Docker** para containerização

## Fluxo de Processamento

### 1. Recebimento da Imagem

O processo inicia quando uma imagem é enviada via POST para `/api/v1/images/upload`:

1. **Upload**: Imagem recebida pelo FastAPI e salva temporariamente
2. **Validação**: Verificação de formato e tamanho
3. **Task Creation**: Criação de task Celery com ID único
4. **Queue**: Envio da task para fila RabbitMQ
5. **Response**: Retorno imediato do task_id para consulta posterior

### 2. Processamento Assíncrono

O worker Celery consome a task da fila RabbitMQ e executa o pipeline:

#### 2.1 Detecção YOLO
```
Imagem Original → YOLOv8 → Detecções (bbox + confiança)
```
- Execução do modelo YOLOv8 treinado para 4 classes: `box`, `qr_code`, `pallet`, `forklift`
- Threshold de confiança configurável (padrão: 0.85)
- Retorna bounding boxes normalizadas com scores

#### 2.2 Extração e Decodificação de QR Codes
```
Detecções QR → Crop das Regiões → Múltiplas Estratégias → Decodificação
```

**Processo de Crop:**
- Para cada detecção com classe `qr_code`, extrai a região usando as coordenadas do bounding box
- Aplica margem de segurança (padding) para garantir captura completa
- Converte coordenadas normalizadas para pixels absolutos

**Pipeline de Decodificação:**
O sistema implementa 7 estratégias sequenciais para maximizar taxa de sucesso:

1. **Original**: Decodificação direta do crop
2. **Adaptive Threshold**: Binarização adaptativa com filtro gaussiano
3. **Noise Reduction**: Filtro mediano + threshold Otsu
4. **Sharpening**: Filtro de nitidez + binarização
5. **Multi-Scale**: Redimensionamento (1.5x, 2.0x) + threshold Otsu
6. **Otsu Variants**: Blur gaussiano + Otsu normal/invertido
7. **Rotations**: Rotações (90°, 180°, 270°) para QR codes inclinados

**Critério de Parada**: Assim que uma estratégia consegue decodificar, o processo para e retorna o conteúdo.

### 3. Fluxo de Dados: RabbitMQ → Redis → PostgreSQL

#### 3.1 RabbitMQ (Message Broker)
- **Entrada**: Tasks de processamento ficam em fila
- **Consumo**: Workers Celery processam tasks de forma distribuída
- **Durabilidade**: Fila persistente para não perder tasks em caso de restart

#### 3.2 Redis (Cache + Result Backend)
- **Cache de Resultados**: Armazena resultados temporariamente com TTL
- **Task Status**: Mantém estado das tasks (PENDING → PROCESSING → SUCCESS/FAILURE)
- **Performance**: Acesso rápido para consultas de resultado via task_id

#### 3.3 PostgreSQL (Persistência)
- **Metadados**: Informações das tasks (timestamps, status, paths)
- **Histórico**: Registro permanente para auditoria e analytics
- **Relacionamentos**: Estrutura normalizada para consultas complexas

### 4. Consulta de Resultados

```
GET /api/v1/results/{task_id} → Redis (se disponível) → PostgreSQL (fallback)
```

O sistema implementa cache inteligente:
- **Hit Redis**: Retorno imediato se resultado está em cache
- **Miss Redis**: Busca no PostgreSQL e popula cache para próximas consultas

## Dataset e Modelo

O modelo YOLOv8 foi treinado em dataset customizado contendo:
- **Classes**: box, qr_code, pallet, forklift
- **Distribuição**: 70% train / 20% validation / 10% test
- **Performance**: mAP@0.5 de 0.847

## Configuração

```bash
docker-compose up --build
# API disponível em http://localhost:8080
```

### Variáveis Principais
```env
CONFIDENCE_THRESHOLD=0.85
REDIS_URL=redis://redis:6379/0
RABBITMQ_URL=amqp://admin:admin123@rabbitmq:5672//
POSTGRES_URL=postgresql://fpf_user:fpf_pass@postgres:5432/fpf_db
```



## Exemplo de Resposta

```json
{
  "task_id": "abc123-def456",
  "status": "completed",
  "detected_objects": [
    {
      "object_id": "OBJ_001",
      "class": "pallet",
      "confidence": 0.92,
      "bounding_box": {
        "x": 150, "y": 200,
        "width": 300, "height": 180
      }
    }
  ],
  "qr_codes": [
    {
      "qr_id": "QR_001",
      "data": "PALLET-ABC-123",
      "confidence": 0.98,
      "position": {"x": 200, "y": 150}
    }
  ],
  "processing_time": 0.847
}
```

---

A solução foi projetada para ser robusta e escalável, adequada para ambientes industriais que requerem processamento confiável de grandes volumes de imagens com detecção precisa de objetos e decodificação eficiente de QR codes.
