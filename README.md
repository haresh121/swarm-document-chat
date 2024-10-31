# Swarm Document Chat System

### Getting Started

1. Install Dependancies
```bash
pip install -r requirements.txt
```
And to install SWARM
```bash
pip install git+https://github.com/openai/swarm.git
```

2. Process the Documents and insert the embeddings to Qdrant Vector Database
```bash
python prep_data.py
```

3. Run the Chat Interface by using
```bash
python main.py
```