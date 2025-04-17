from modules.core import SearchBase

if __name__ == "__main__":
    rag = SearchBase(config_path="config/config_10.ini")

    user_query = input("Введите вопрос: ")
    result = rag.search_doc(user_query)

    print("\nОтвет: ")
    print(result)
