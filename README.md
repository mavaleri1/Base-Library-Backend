# Base Library - AI-Powered Educational Material Generation Platform

A comprehensive microservices-based platform for generating educational materials using LangGraph workflows, featuring handwritten note recognition, material synthesis, and interactive question generation.

## 🏗️ Architecture Overview

This project implements a sophisticated AI-powered educational platform built on a microservices architecture with the following core components:

### Core Services

1. **Core AI Service** - Main LangGraph workflow orchestrator
2. **Article Service** - File storage and export management
3. **Prompt Studio Service** - Dynamic prompt generation and user customization
4. **PostgreSQL** - Primary data storage

## 🔄 LangGraph Workflow Architecture

The platform is built around a sophisticated LangGraph workflow that processes educational materials through multiple AI-powered nodes:

### Workflow Flow

```mermaid
graph TD
    A[START] --> B[Input Processing]
    B --> C[Content Generation]
    C --> D{Images Present?}
    D -->|Yes| E[Handwritten Recognition]
    D -->|No| F[Question Generation]
    E --> G[Material Synthesis]
    G --> H[Material Editing]
    H --> F[Question Generation]
    F --> I[Answer Generation]
    I --> J[END]
```

### Node Descriptions

#### 1. **Input Processing Node** (`input_processing`)
- **Purpose**: Analyzes user input and determines workflow path
- **Functionality**: 
  - Processes text input and image uploads
  - Validates content using security guards
  - Determines next workflow step based on input type

#### 2. **Content Generation Node** (`generating_content`)
- **Purpose**: Generates educational material based on exam questions
- **Functionality**:
  - Creates comprehensive study materials
  - Uses personalized prompts from Prompt Configuration Service
  - Determines workflow path (recognition vs. direct question generation)

#### 3. **Recognition Node** (`recognition_handwritten`)
- **Purpose**: Processes handwritten notes from uploaded images
- **Functionality**:
  - OCR processing of handwritten content
  - Text extraction and formatting
  - Integration with image processing services

#### 4. **Synthesis Node** (`synthesis_material`)
- **Purpose**: Combines generated content with recognized notes
- **Functionality**:
  - Merges AI-generated material with handwritten notes
  - Creates comprehensive study materials
  - Ensures content coherence and completeness

#### 5. **Edit Material Node** (`edit_material`)
- **Purpose**: Interactive material editing with human-in-the-loop (HITL)
- **Functionality**:
  - Iterative material refinement
  - User feedback integration
  - Content improvement based on user input

#### 6. **Question Generation Node** (`generating_questions`)
- **Purpose**: Creates assessment questions with HITL feedback
- **Functionality**:
  - Generates relevant exam questions
  - Implements feedback loop for question refinement
  - Ensures question quality and relevance

#### 7. **Answer Generation Node** (`answer_question`)
- **Purpose**: Generates comprehensive answers to questions
- **Functionality**:
  - Creates detailed answer explanations
  - Provides educational value
  - Finalizes the learning material package

## 🛠️ Technical Architecture

### State Management

The workflow uses a comprehensive state model (`GeneralState`) that tracks:

```python
class GeneralState(BaseModel):
    # Input data
    input_content: str
    display_name: Optional[str]
    image_paths: List[str]
    
    # Processing results
    recognized_notes: str
    generated_material: str
    synthesized_material: str
    
    # Questions and answers
    questions: List[str]
    questions_and_answers: List[str]
    
    # HITL interaction
    feedback_messages: List[Any]
    edit_count: int
    needs_user_input: bool
    agent_message: Optional[str]
```

### Security & Validation

- **Security Guard**: Content validation and sanitization
- **Input Validation**: Multi-layer content checking
- **Fuzzy Matching**: Advanced threat detection
- **Graceful Degradation**: System continues operation even with security failures

### Model Configuration

The platform supports multiple LLM providers:

- **OpenAI** (GPT-4, GPT-3.5)
- **OpenRouter** (Various models)
- **Fireworks AI**
- **DeepSeek**

Each node can be configured with different models based on requirements.

## 🚀 Services Architecture

### Core AI Service (Port 8000)
- **Framework**: FastAPI + LangGraph
- **Functionality**: 
  - Workflow orchestration
  - State management
  - HITL interaction handling
  - Artifact management

### Artifacts Service (Port 8001)
- **Framework**: FastAPI
- **Functionality**:
  - File storage and management
  - Export capabilities (PDF, Markdown, ZIP)
  - Thread-based organization
  - Web3 authentication support

### Prompt Configuration Service (Port 8002)
- **Framework**: FastAPI
- **Functionality**:
  - Dynamic prompt generation
  - User profile management
  - Template customization
  - Prompt versioning

### Database (Port 5431)
- **Type**: PostgreSQL 16
- **Functionality**:
  - Workflow state persistence
  - User data storage
  - Artifact metadata
  - Authentication data

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
OPENAI_API_KEY=your_openai_key
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/core

# Optional LLM Providers
OPENROUTER_API_KEY=your_openrouter_key
FIREWORKS_API_KEY=your_fireworks_key
DEEPSEEK_API_KEY=your_deepseek_key

# Security
SECURITY_ENABLED=true
SECURITY_FUZZY_THRESHOLD=0.85
SECURITY_MIN_CONTENT_LENGTH=10

# Artifacts
ARTIFACTS_BASE_PATH=/app/data/artifacts
ARTIFACTS_MAX_FILE_SIZE=10485760
```

### Configuration Files

- `configs/graph.yaml` - LangGraph node configurations
- `configs/prompts.yaml` - Prompt templates
- `configs/providers.yaml` - LLM provider settings

## 🐳 Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start with LangFuse monitoring
docker-compose --profile langfuse up -d
```

### Service Health Checks

- **Core**: `http://localhost:8000/health`
- **Artifacts**: `http://localhost:8001/health`
- **Prompt Config**: `http://localhost:8002/health`

## 📊 Monitoring & Observability

### LangFuse Integration (Optional)

The platform supports LangFuse for AI observability:

- **LangFuse Web UI**: `http://localhost:3000`
- **MinIO Console**: `http://localhost:9091`
- **ClickHouse**: Analytics database
- **Redis**: Caching layer

### Logging

- Centralized logging in `/logs` directory
- Structured logging with different levels
- Service-specific log files

## 🔄 Workflow Execution

### Human-in-the-Loop (HITL) Pattern

The platform implements sophisticated HITL patterns:

1. **Generation**: AI creates initial content
2. **Feedback**: User provides input/feedback
3. **Editing**: AI refines content based on feedback
4. **Completion**: Process continues until satisfaction

### State Persistence

- PostgreSQL checkpoints for workflow state
- Resume capability for interrupted workflows
- Thread-based session management

## 🎯 Key Features

### Educational Material Generation
- AI-powered content creation
- Handwritten note recognition
- Material synthesis and editing
- Interactive question generation

### Export Capabilities
- PDF export with custom templates
- Markdown export for documentation
- ZIP archives for complete packages
- Custom styling and formatting

### Web3 Integration
- Blockchain authentication
- NFT content verification
- Decentralized ownership tracking
- Content hash validation

### Security Features
- Content sanitization
- Injection attack prevention
- Fuzzy threat detection
- Graceful security degradation

## 🚀 Getting Started

1. **Clone the repository**
2. **Set up environment variables**
3. **Run database migrations**
4. **Start services with Docker Compose**
5. **Access the API endpoints**

## 📝 API Documentation

### Core Service Endpoints
- `POST /process` - Start new workflow
- `GET /status/{thread_id}` - Check workflow status
- `POST /continue/{thread_id}` - Continue HITL interaction

### Artifacts Service Endpoints
- `GET /threads` - List all threads
- `POST /export/pdf/{thread_id}/{session_id}` - Export to PDF
- `POST /export/markdown/{thread_id}/{session_id}` - Export to Markdown

### Prompt Configuration Endpoints
- `GET /prompts/{user_id}/{node_name}` - Get personalized prompts
- `POST /profiles` - Create user profiles
- `PUT /profiles/{profile_id}` - Update user profiles

## 🔮 Future Enhancements

- Advanced AI model integration
- Enhanced Web3 features
- Real-time collaboration
- Advanced analytics and insights
- Multi-language support
- Mobile application

---

This platform represents a cutting-edge approach to AI-powered educational content generation, combining the power of LangGraph workflows with modern microservices architecture and Web3 technologies.



