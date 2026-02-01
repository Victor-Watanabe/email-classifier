📧 Email Classifier

Uma plataforma de classificação de emails usando Inteligência Artificial e NLP, capaz de identificar se um email é produtivo ou improdutivo e sugerir respostas automáticas.

O backend foi desenvolvido em Python com FastAPI, integrando modelos de Machine Learning e LLM (Google Gemini) para análise de textos e PDFs.

🚀 Funcionalidades

🧠 Classificação de emails via Machine Learning

📄 Processamento de PDFs enviados pelo usuário

📡 API REST para envio de texto ou arquivos e recebimento de resultados

🤖 Sugestão de respostas automáticas usando LLM (Gemini)

🛠️ Stack do Backend e Motivação
Tecnologia	Motivação
FastAPI	Framework moderno para APIs, rápido e com documentação automática.

Uvicorn	Servidor ASGI de alta performance para rodar FastAPI de forma assíncrona.

python-dotenv	Carrega variáveis de ambiente de .env para manter chaves e configs seguras.

Pydantic	Valida e garante consistência dos dados recebidos pela API.

python-multipart	Permite uploads de arquivos PDF via FormData.

PyPDF2	Extrai texto de PDFs enviados pelo usuário.

scikit-learn	Treinamento e inferência de modelos de classificação de texto.

joblib	Serializa e carrega modelos treinados rapidamente.

spaCy	Pré-processamento de texto e NLP avançado para melhorar a classificação.

google-generativeai (Gemini)	Integração com LLM para gerar respostas automáticas e insights de texto.

📥 Pré-requisitos

Python 3.8+

Pip ou ambiente virtual (venv / conda)

Chave do Google Gemini: para utilizar a LLM do projeto, você precisa de uma chave de API válida.

⚠️ Importante:
Crie um arquivo .env na raiz do backend e adicione sua chave do Gemini:

GEMINI_API_KEY=Sua_Chave_Aqui


Sem essa chave, a funcionalidade de geração de respostas automáticas não funcionará.

🧠 Como Funciona

Pré-processamento: limpeza de texto, tokenização, vetorização.

Treinamento: scikit-learn treina modelo de classificação.

Inferência: modelo classifica emails novos.

Sugestão de Resposta: LLM (Gemini) gera respostas automáticas, quando habilitado.

📈 Avaliação

Métricas de desempenho: acurácia, F1-score, precisão, recall

🔧 Instalação e execução

Clone o repositório:

git clone https://github.com/Victor-Watanabe/email-classifier.git
cd email-classifier


Crie e ative um ambiente virtual:

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Instale as dependências:

pip install -r backend/requirements.txt


Executar com modelos pré-treinados (recomendado para teste rápido):

python backend/app.py


Os modelos já estão treinados e podem ser usados diretamente.

Matriz de confusão para análise detalhada de classificação

