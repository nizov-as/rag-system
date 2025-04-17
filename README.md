# RAG System (Retrieval-Augmented Generation)

## Структура проекта
```
rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py
├── config/
│   ├── config_10.ini
│   └── config_lc_bge.ini
│   └── config_lc_bert.ini
├── data/
│   └── documents/
├── modules/
│   ├── split_util.py
│   └── rag_bm25_faiss.py
│   └── rag_lc_bge.py
│   └── rag_lc_bert.py
├── tests/
│   └── rag_test.ipynb
│   └── laws_accuracy_test.ipynb
```

**Ноутбук с запуском модулей и тестированием системы находится в tests/rag_test.ipynb**

## Быстрый запуск
```bash
pip install -r requirements.txt
python main.py
```
