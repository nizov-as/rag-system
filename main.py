from modules.rag_bm25_faiss import SearchBase
from modules.rag_bm25_faiss import Doc_Base
from modules.llm import LLM
from modules.config_parse import Config


def main():
    doc_base = Doc_Base(name='doc_base_test', path='./data')
    doc_base.fill(doc_path='./input')
    doc_base.save()
    doc_base.load()

    search_base = SearchBase(config=Config())
    search_base.fill(doc_base)
    search_base.load()

    my_llm = LLM()

    query = input("Введите вопрос: ")

    llm_res, qq3, res3, exp_docs, prompt = search_base.get_response(my_llm, query, doc_base)

    result = {
        "Запрос": query,
        "Промпт": prompt,
        "Ответ RAG": llm_res
    }

    print("\nРезультаты генерации RAG: \n")
    print(result)


if __name__ == "__main__":
    main()
