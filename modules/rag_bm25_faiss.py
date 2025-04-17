import re
import logging
from split_util import RecursiveCharacterTextSplitter
from split_util import Doc
import pandas as pd
import numpy as np
import os
import re
import pickle as pkl
from datetime import datetime
import inspect
import configparser
from tqdm.notebook import tqdm
from dataclasses import dataclass
import random
#import bm25s
from rank_bm25 import BM25Okapi # type: ignore
import faiss
from collections import OrderedDict
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import download
import nltk
import string
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def clean(text):
    """
        очистка текста
        Parameters:
            text : входной текст
        Returns:
            очищенный текст
    """
    pat1 = "\n{2,}"
    pat2 = "[^а-яА-ЯёЁa-zA-Z0-9№\.?!, - :;\n\"«»]"
    pat3 = " {2,}"
    text1 = re.sub(pat1, "\n", text)
    text2 = re.sub(pat2, " ", text1)

    return re.sub(pat3, " ", text2).strip()


def rrf(meta_ls):
    """
        расчет reciprocal rank fusion
        Parameters:
            meta_ls : List[List[item]] список списков items от разных поисковиков
        Returns:
            RRF рейтинг вместе со score List[(score, item)]
        """
    full_meta_ls = []
    for ls in meta_ls:
        full_meta_ls = full_meta_ls + [(key, x) for key, x in enumerate(ls)]
    rating_dict = {}
    for key, x in full_meta_ls:
        rt = rating_dict.get(x, 0)
        rt_new = 1.0 / (60 + key)
        rating_dict[x] = rt + rt_new
    new_rating = [(rat, x) for x, rat in rating_dict.items()]
    new_rating.sort(key=lambda x: x[0], reverse=True)
    return new_rating


def doc_str(ls_doc):
    s = ''
    for doc in ls_doc:
        s = s + str(doc) + "\n ==========  ========== \n"
    return s


def meta_str(ls_doc):
    s = ''
    for doc in ls_doc:
        s = s + str(doc.metadata) + "\n ==========  ========== \n"
    return s


def idx_match(groups):
    """
    вспомогательная функция
    """
    LN = len(groups)
    idx = None
    for k in range(LN):
        if groups[k] != '' and groups[k] is not None:
            idx = k
            break
    return idx


def level1_split(doc):
    """
     Parameters:
            doc: Документ для разбиения по первому уровню (пункты и подпункты)
        Returns:
            ls_doc: list с документами, полученные после разбивки по первому уровню
    """
    txt = doc.page_content
    source = doc.metadata['source']

    pat_02 = "(\n\d+\. )|(\n\d+\.\d+\. )|(\n\d+\.\d+\.\d+\. )|(\n\d+\.\d+\.\d+\.\d+\. )"
    m_iterator = re.finditer(pat_02, txt)
    l_match = list(m_iterator)
    metadata0 = {'source': source, 'level1_name': '', 'level1_id': ''}
    if len(l_match) > 0:
        ls_tec = []
        for m in l_match:
            idx = idx_match(m.groups())
            # print(m.groups())
            # print(idx)
            group = m.group().replace("\n", "")
            span = m.span()
            metadata = {'source': source, 'level1_name': group, 'level1_id': idx}

            ls_tec.append((span[0], metadata))

        ls_tec1 = [(0, metadata0)] + ls_tec[:-1]
        ls2 = zip(ls_tec1, ls_tec)
 
        ls_doc = [Doc(page_content=txt[x1[0]:x2[0]], metadata=x1[1]) for x1, x2 in ls2]
    else:
        ls_doc = [Doc(page_content=txt[:], metadata=metadata0)]
    return ls_doc


def level2_split(doc, text_splitter):
    """
     Parameters:
            doc: Документ для разбиения по второму уровню (разбиение на части документов с предыдущего уровня)
            text_splitter : сплиттер документов
     Returns:
            ls_doc: list с документами, полученные после разбивки по второму уровню
    """
    txt = doc.page_content
    source = doc.metadata['source']
    #level0_name = doc.metadata['level0_name']
    #level0_id = doc.metadata['level0_id']
    level1_name = doc.metadata['level1_name']
    level1_id = doc.metadata['level1_id']
    
    ls_txt = text_splitter.split_text(txt)
    ls_doc = [Doc(page_content=x, metadata={'source': source, 'level1_name': level1_name, 'level1_id': level1_id, "order": k}) for k, x in enumerate(ls_txt)]
    return ls_doc


def get_concat(s1, s2):
    """
     склейка текстов s1, s2 с удалением повторов
     Parameters:
            s1 : string
            s2 : string
     Returns:
            string
    """
    l1 = len(s1)
    l2 = len(s2)
    # print(l1, l2)
    flag = False
    res = None
    for i in range(l1 - 1, -1, -1):
        for k in range(l2):
            if s1[i:] == s2[:k]:
                res = k
                flag = True
        if flag:
            break
    if res is None:
        return s1[:] + "\n" + s2[:]
    else:
        return s1[:] + "\n" + s2[res:]


def all_concat(*args):
    """
     склейка всех текстов
     Parameters:
            args : list string
     Returns:
            string
    """
    tmp = ''
    for el in args:
        tmp = get_concat(tmp, el)
    return tmp


# metadata = {'source':source, 'level0_name':level0_name, 'level0_id':level0_id, 'level1_name':level1_name, 'level1_id':level1_id, "order":k, 'id' : m}
def meta_concat(meta1, meta2):
    """
     склейка метданных для документов
     Parameters:
            meta1 : dict
            meta2 : dict
     Returns:
            dict
    """
    meta_conc = dict()
    for key, value in meta1.items():
        if key == 'id':
            meta_conc[key] = meta2.get(key, -1)
        elif key == 'pref_ln':
            meta_conc[key] = meta1.get(key, 0)
        else:
            if meta1.get(key, '') == meta2.get(key, ''):
                meta_conc[key] = meta2.get(key, '')
            else:
                meta_conc[key] = str(meta1.get(key, '')) + ":+:" + str(meta2.get(key, ''))
                meta_conc[key] = str(meta_conc.get(key, '')).replace("   ", " ").replace("  ", " ")
            ls = str(meta_conc.get(key, '')).split(":+:")
            ls1 = []
            tmp = None
            for x in ls:
                if x != tmp:
                    ls1.append(x)
                tmp = x

            meta_conc[key] = ":+:".join(ls1)
            meta_conc['order'] = 'составной'
    return meta_conc


def meta_mask(meta0, mask=[]):
    """
         маскировка метданных для документов
         Parameters:
                meta0 : dict
                mask  : список ключей для показа, остальные маскируются
         Returns:
                dict
    """
    meta_conc = dict()
    if len(mask) < 0:
        return meta0
    for key, value in meta0.items():
        if key in mask:
            meta_conc[key] = meta0.get(key, '')
    return meta_conc


def doc_concat(doc1, doc2):
    """
         склейка документов
         Parameters:
                doc1 : документ
                doc2 : документ
         Returns:
                документ
    """
    txt1 = doc1.page_content
    txt2 = doc2.page_content
    meta1 = doc1.metadata
    meta2 = doc2.metadata
    pref_ln2 = meta2.get('pref_len', 0)
    meta_full = meta_concat(meta1, meta2)
    txt = get_concat(txt1, txt2[pref_ln2:])
    return Doc(page_content=txt, metadata=meta_full)


def all_doc_concat(*args):
    """
     склейка всех документов
     Parameters:
            args : list документов
     Returns:
            документ
    """
    if args == ():
        return Doc.empty()
    else:
        doc = args[0]
        for el in args[1:]:
            doc = doc_concat(doc, el)
        return doc


class Doc_Base():
    """
        класс база разбитых документов
    """

    def __init__(self, name='doc_base', path="./data"):

        self.name = name
        self.path = path
        self.fl_ls = None
        self.txt_files = None
        self.doc_path = None

        self.txt_files_names = None
        self.txt_files_dict = None

        self.full_docs = None
        self.base = None

    def fill(self, doc_path = "./input"):
        """
        заполняем базу документами из папки doc_path, разбиваем документы
        """
        self.doc_path = doc_path  # Исправлено с self.path на doc_path
        self.fl_ls = [x for x in os.listdir(doc_path) if os.path.splitext(x)[1] == ".txt"] 
        self.txt_files_names = [f for f in os.listdir(doc_path) if os.path.isfile(os.path.join(doc_path, f)) and f.endswith('.txt')]
        self.make_txt_files_dict()
        self.make_docs_clear()
        self.split_docs()

    def make_txt_files_dict(self):
        mapping = {
            "law_1.txt": "Федеральный закон от 10.12.2003 № 173-ФЗ «О валютном регулировании и валютном контроле».",
            "law_2.txt": "Положение о Федеральной налоговой службе, утвержденное постановлением Правительства Российской Федерации от 30.09.2004 № 506.",
            "law_3.txt": "Постановление Правительства Российской Федерации от 06.03.2022 № 295 «Об утверждении Правил выдачи Правительственной комиссией по контролю за осуществлением иностранных инвестиций в Российской Федерации разрешений на осуществление (исполнение) резидентами сделок (операций) с иностранными лицами в целях реализации дополнительных временных мер экономического характера по обеспечению финансовой стабильности Российской Федерации и внесении изменения в Положение о Правительственной комиссии по контролю за осуществлением иностранных инвестиций в Российской Федерации».",
            "law_4.txt": "Указание Банка России от 23.03.2022 № 6096-У «О внесении изменений в Положение Банка России от 27.02.2017 № 579-П «О Плане счетов бухгалтерского учета для кредитных организаций и порядке его применения».",
            "law_5.txt": "Указ Президента Российской Федерации от 28.02.2022 № 79 «О применении специальных экономических мер в связи с недружественными действиями Соединенных Штатов Америки и примкнувших к ним иностранных государств и международных организаций».",
            "law_6.txt": "Указ Президента Российской Федерации от 18.03.2022 № 126 «О дополнительных временных мерах экономического характера по обеспечению финансовой стабильности Российской Федерации в сфере валютного регулирования».",
            "law_7.txt": "Указ Президента Российской Федерации от 05.03.2022 № 95 «О временном порядке исполнения обязательств перед некоторыми иностранными кредиторами».",
            "law_8.txt": "Указ Президента Российской Федерации от 5 марта 2022 года № 95 «О временном порядке исполнения обязательств перед некоторыми иностранными кредиторами.",
            "law_9.txt": "Решение Совета директоров Банка России от 24.06.2022 «Об установлении режима счетов типа С для проведения расчетов и осуществления (исполнения) сделок (операций), на которые распространяется порядок исполнения обязательств, предусмотренный Указом Президента Российской Федерации от 5 марта 2022 года № 95 «О временном порядке исполнения обязательств перед некоторыми иностранными кредиторами»",
            "law_10.txt": "Решение Совета директоров Банка России от 22.07.2022 «О внесении изменений в решение Совета директоров Банка России от 24 июня 2022 года».",
            "law_11.txt": "Разъяснения Банка России от 18.03.2022 № 2-ОР, размещенного на официальном сайте Банка России в информационно-телекоммуникационной сети «Интернет», а также в правовой информационно-справочной системе «Консультант плюс» в целях применения Указа № 79 при определении лиц, являющихся резидентами, необходимо руководствоваться понятием «резидент», определенным в статье 1 Федерального закона от 10 декабря 2003 года № 173-ФЗ «О валютном регулировании и валютном контроле».",
            "law_12.txt": "Официальное разъяснение Банка России от 16.04.2022 № 4-ОР «О применении отдельных положений Указа Президента Российской Федерации от 28 февраля 2022 года № 79 «О применении специальных экономических мер в связи с недружественными действиями Соединенных Штатов Америки и примкнувших к ним иностранных государств и международных организаций».",
            "law_13.txt": "Информация Банка России от 09.03.2022 «Банк России вводит временный порядок операций с наличной валютой»",
            "law_14.txt": "Информация от 01.08.2022 «Банк России продлил ограничения на снятие наличной иностранной валюты еще на 6 месяцев, до 9 марта 2023 года»",
            "law_15.txt": "Информация от 06.03.2023 «Банк России продлил еще на полгода, до 09.09.2023 ограничения на снятие наличной иностранной валюты»",
            "law_16.txt": "Информация от 07.09.2023 «Банк России продлил еще на полгода, до 09.03.2024 ограничения на снятие наличной иностранной валюты»",
            "law_17.txt": "Информация от 07.03.2024 «Банк России продлил еще на 6 месяцев, до 09.09.2024 ограничения на снятие наличной иностранной валюты»",
            "law_18.txt": "Информация от 06.09.2024 ««Банк России продлил еще на полгода, до 09.03.2025 ограничения на снятие наличной иностранной валюты»",
            "law_19_1.txt": "Гражданский кодекс Российской Федерации (часть первая)",
            "law_19_2.txt": "Гражданский кодекс Российской Федерации (часть вторая)",
            "law_19_3.txt": "Гражданский кодекс Российской Федерации (часть третья)",
            "law_19_4.txt": "Гражданский кодекс Российской Федерации (часть четвертая)",
            "law_20.txt": "Федеральный закон от 10.07.2002 № 86-ФЗ «О Центральном банке Российской Федерации (Банке России)»",
            "law_21.txt": "Федеральный закон от 02.12.1990 № 395-1 «О банках и банковской деятельности»",
            "law_22.txt": "Федеральный закон от 21.12.2013 № 353-ФЗ «О потребительском кредите (займе)»",
            "law_23.txt": "Федеральный закон от 03.07.2016 № 230-ФЗ «О защите прав и законных интересов физических лиц при осуществлении деятельности по возврату просроченной задолженности и о внесении изменений в Федеральный закон «О микрофинансовой деятельности и микрофинансовых организациях»",
            "law_24.txt": "Федеральный закон от 07.08.2001 № 115-ФЗ «О противодействии легализации (отмыванию) доходов, полученных преступным путем, и финансированию терроризма»",
            "law_25.txt": "Федеральный закон от 23.12.2003 № 177-ФЗ «О страховании вкладов в банках Российской Федерации»",
            "law_26.txt": "Федеральный закон от 27.06.2011 № 161-ФЗ «О национальной платежной системе»",
            "law_27.txt": "Федеральный закон от 07.10.2022 № 377-ФЗ «Об особенностях исполнения обязательств по кредитным договорам (договорам займа) лицами, призванными на военную службу по мобилизации в Вооруженные Силы Российской Федерации, лицами, принимающими участие в специальной военной операции, а также членами их семей и о внесении изменений в отдельные законодательные акты Российской Федерации»",
            "law_28.txt": "Федеральный закон от 26.10.2002 № 127-ФЗ «О несостоятельности (банкротстве)»",
            "law_29.txt": "Федеральный закон от 16.07.1998 № 102-ФЗ «Об ипотеке (залоге недвижимости)»",
            "law_30.txt": "Федеральный закон от 02.10.2007 № 229-ФЗ «Об исполнительном производстве»",
            "law_31.txt": "Закон РФ от 07.02.1992 № 2300-1 «О защите прав потребителей»",
            "law_32.txt": "Федеральный закон от 30.12.2004 № 218-ФЗ «О кредитных историях»",
            "law_33.txt": "Федеральный закон от 04.06.2018 № 123-ФЗ «Об уполномоченном по правам потребителей финансовых услуг»",
            "law_34.txt": "Федеральный закон от 03.04.2020 № 106-ФЗ «О внесении изменений в Федеральный закон «О Центральном банке Российской Федерации (Банке России)» и отдельные законодательные акты Российской Федерации в части особенностей изменения условий кредитного договора, договора займа»",
            "law_35.txt": "Постановление Правительства РФ от 07.09.2019 № 1170 «Об утверждении Правил предоставления субсидий акционерному обществу «ДОМ.РФ» на возмещение недополученных доходов и затрат в связи с реализацией мер государственной поддержки семей, имеющих детей, в целях создания условий для погашения обязательств по ипотечным жилищным кредитам (займам) и Положения о реализации мер государственной поддержки семей, имеющих детей, в целях создания условий для погашения обязательств по ипотечным жилищным кредитам (займам)»",
            "law_36.txt": "Постановление Правительства РФ от 30.12.2017 № 1711 «Об утверждении Правил предоставления субсидий из федерального бюджета акционерному обществу «ДОМ.РФ» в виде вкладов в имущество акционерного общества «ДОМ.РФ», не увеличивающих его уставный капитал, для возмещения российским кредитным организациям и акционерному обществу «ДОМ.РФ» недополученных доходов по выданным (приобретенным) жилищным (ипотечным) кредитам (займам), предоставленным гражданам Российской Федерации, имеющим детей, и Правил возмещения российским кредитным организациям и акционерному обществу «ДОМ.РФ» недополученных доходов по выданным (приобретенным) жилищным (ипотечным) кредитам (займам), предоставленным гражданам Российской Федерации, имеющим детей»",
            "law_37.txt": "Постановление Правительства РФ от 30.04.2022 № 805 «Об утверждении Правил предоставления субсидий из федерального бюджета акционерному обществу «ДОМ.РФ» в виде вклада в имущество акционерного общества «ДОМ.РФ», не увеличивающего его уставный капитал, на цели возмещения кредитным и иным организациям недополученных доходов по жилищным (ипотечным) кредитам (займам), выданным работникам аккредитованных организаций, осуществляющих деятельность в области информационных технологий, и Правил возмещения кредитным и иным организациям недополученных доходов по жилищным (ипотечным) кредитам (займам), выданным работникам аккредитованных организаций, осуществляющих деятельность в области информационных технологий»",
            "law_38.txt": "Постановление Правительства РФ от 23.04.2020 № 566 «Об утверждении Правил возмещения кредитным и иным организациям недополученных доходов по жилищным (ипотечным) кредитам (займам), выданным гражданам Российской Федерации в 2020 - 2024 годах»",
            "law_39.txt": "Постановление Правительства РФ от 30.11.2019 № 1567 «Об утверждении Правил предоставления субсидий из федерального бюджета российским кредитным организациям и акционерному обществу «ДОМ.РФ» на возмещение недополученных доходов по выданным (приобретенным) жилищным (ипотечным) кредитам (займам), предоставленным гражданам Российской Федерации на строительство (приобретение) жилого помещения (жилого дома) на сельских территориях (сельских агломерациях)»",
            "law_40.txt": "Постановление Правительства РФ от 07.12.2019 № 1609 «Об утверждении условий программы «Дальневосточная ипотека», Правил предоставления субсидий из федерального бюджета акционерному обществу «ДОМ.РФ» в виде вкладов в имущество акционерного общества «ДОМ.РФ», не увеличивающих его уставный капитал, для возмещения российским кредитным организациям и иным организациям недополученных доходов по жилищным (ипотечным) кредитам, предоставленным гражданам Российской Федерации на приобретение или строительство жилых помещений на территориях субъектов Российской Федерации, входящих в состав Дальневосточного федерального округа, и внесении изменений в распоряжение Правительства Российской Федерации от 2 сентября 2015 г. N 1713-р»",
            "law_41.txt": "Указание Банка России от 17.05.2022 № 6139-У «О минимальных (стандартных) требованиях к условиям и порядку осуществления добровольного страхования жизни и здоровья заемщика по договору потребительского кредита (займа), к объему и содержанию предоставляемой информации о договоре добровольного страхования жизни и здоровья заемщика по договору потребительского кредита (займа), а также о форме, способах и порядке предоставления указанной информации»",
            "law_42.txt": "Положение Банка России от 29.06.2021 № 762-П «О правилах осуществления перевода денежных средств»",
            "law_43.txt": "Инструкция Банка России от 30.06.2021 № 204-И «Об открытии, ведении и закрытии банковских счетов и счетов по вкладам (депозитам)»"
        }
        self.txt_files_dict = mapping

    def save(self):
        with open(f'{self.path}/{self.name}.pkl', 'wb') as fl:
            pkl.dump(self.base, fl)
        with open(f'{self.path}/{self.name}_full.pkl', 'wb') as fl:
            pkl.dump(self.full_docs, fl)

    def load(self):
        if os.path.isfile(f'{self.path}/{self.name}.pkl'):
            with open(f'{self.path}/{self.name}.pkl', 'rb') as fl:
                self.base = pkl.load(fl)

        if os.path.isfile(f'{self.path}/{self.name}_full.pkl'):
            with open(f'{self.path}/{self.name}_full.pkl', 'rb') as fl:
                self.full_docs = pkl.load(fl)

    def get_by_id(self, idx):
        res = self.base.get(idx, Doc.empty())
        return res

    def make_docs_clear(self):
        pat = '\s{4,}'
        punct = re.escape(string.punctuation)
        pattern_caps = rf'(?m)^[A-ZА-ЯЁ0-9\s{punct}]+$(?:\r?\n)?' # удаляет строки с русским капслоком
        pattern_date = r'(?s)^.*?Дата\s+сохранения:\s*\d{2}[./-]\d{2}[./-]\d{4}[^\n]*\n?' #удаляет лишнюю метаинфу скачивания с консультанта (до "дата сохранения dd.mm.yyyy включительно")
        pattern_consultant = r'(?m)^Документ предоставлен КонсультантПлюс\r?\n?'
        pattern_dashes = r'-{3,}' #удаляет 3+ тире подряд
        patterns = [pattern_date, pattern_dashes, pattern_caps, pattern_consultant]
        full_docs = []
        if self.txt_files_names is not None:
            for src in self.txt_files_names:
                txt = self.load_txt(os.path.join(self.doc_path, src))

                txt = re.sub(pat, '\n\n', txt)
                for pat in patterns: 
                    txt = re.sub(pat, '', txt)
                doc = Doc(page_content=txt, metadata={"source": self.txt_files_dict[src]})
                full_docs.append(doc)
        self.full_docs = full_docs[:]
        return None

    # Doc(page_content = x, metadata = {'source':source, 'level0_name':level0_name, 'level0_id':level0_id, 'level1_name':level1_name, 'level1_id':level1_id, "order":k, 'id' = k} )
    def split_docs(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        ls_doc_02 = []
        for doc in self.full_docs:
            rr = level1_split(doc)
            ls_doc_02 = ls_doc_02 + rr

        ls_doc_03 = []
        for doc in ls_doc_02:
            rr = level2_split(doc, text_splitter)
            ls_doc_03 = ls_doc_03 + rr
    
        self.base = {}
        for k, d in enumerate(ls_doc_03):
            metadata = d.metadata
            metadata['id'] = k
            pref = "## Документ: " + metadata.get('source', 'неизвестен') + "\n" + "## Пункт или подпункт: " + metadata.get('level1_name', 'неизвестен') + " <#####>\n"
            pref_len = len(pref)
            metadata['pref_len'] = pref_len 
            self.base[k] = Doc(page_content=(pref + d.page_content), metadata=metadata)

        print(f'splited: {len(self.base.items())}')
        return None

    def load_txt(self, txt_path):
        text = ''
        with open(txt_path, 'r', encoding='cp1251') as fl:
            text = fl.read()
        return text


class Reranker():
    """
        класс реранкер
    """
    def __init__(self, device, model_path):
        self.model_path = model_path
        # 'DiTy/cross-encoder-russian-msmarco'
        if device != 'cuda':
            self.reranker_model = CrossEncoder(self.model_path, max_length=512, device='cpu')
        else:
            self.reranker_model = CrossEncoder(self.model_path, max_length=512, device='cuda')
        self.device = device

        # Загрузка базового токенизатора для реранкера
        base_tokenizer = AutoTokenizer.from_pretrained("DiTy/cross-encoder-russian-msmarco")
        # Сохранение токенизатора в указанную директорию
        base_tokenizer.save_pretrained("./reranker_finetuned")

    def rank(self, qst, documents, k=5):
        rank_result = self.reranker_model.rank(qst, [x.page_content for x in documents])
        res_ls = [(x['score'], documents[x['corpus_id']]) for x in rank_result[:k]]
        scores, res_docs = zip(*res_ls)
        return res_docs

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None

    def index(self, corpus_tokens):
        self.corpus = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)

    def retrieve(self, query_tokens_list, corpus, k=10):
        query_tokens = query_tokens_list[0]
        scores = self.bm25.get_scores(query_tokens)
        indices = np.argsort(scores)[::-1][:k]
        doc_results_bm = np.array([corpus[i] for i in indices]).reshape(1, -1)
        doc_scores_bm = np.array([scores[i] for i in indices]).reshape(1, -1)
        return doc_results_bm, doc_scores_bm

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pkl.dump((self.k1, self.b, self.corpus), f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            k1, b, corpus = pkl.load(f)
        instance = cls(k1=k1, b=b)
        instance.index(corpus)
        return instance


class SearchBase():
    """
    Класс для поиска в базе документов.
    Готовим на основе базы документов.
    """
    def __init__(self, config):
        self.config = config
        self.doc_base = None
        self.doc_corpus_token = None

        # Настройки для модели эмбеддинга
        self.emb_model_device = self.config.emb_model_device
        self.doc_emb_model_name = self.config.doc_emb_model_name
        
        config = AutoConfig.from_pretrained("intfloat/multilingual-e5-large")
        # Если параметр norm_bias присутствует, заменим его на элемент elementwise_affine
        if hasattr(config, "norm_bias"):
            # Сохраним значение, затем удалим параметр, чтобы он не передавался в LayerNorm
            norm_bias = config.norm_bias
            del config.norm_bias
            # Если нужно, можно добавить новый атрибут, который используется внутри модели
            config.elementwise_affine = norm_bias

        # Инициализируем токенизатор и модель из Transformers
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        self.model = AutoModel.from_pretrained(self.doc_emb_model_name, config=config)
        self.model.to(self.emb_model_device)
        self.model.eval()

        # Получаем размер эмбеддинга через тестовый пример
        with torch.no_grad():
            inputs = self.tokenizer("Тестовое предложение. Проверка размера.",
                                    padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.emb_model_device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            vv = self.average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vv = F.normalize(vv, p=2, dim=1)
            vv = vv.cpu().numpy()
        # Для батча из одного примера shape = (1, embedding_dim)
        self.doc_vec_size = vv.shape[1]

        # Инициализация остальных компонентов
        self.reranker_device = self.config.reranker_device
        self.doc_reranker = Reranker(self.reranker_device, self.config.doc_reranker_name) 
        self.doc_bm25_db = BM25(k1=1.5, b=0.75)
        #self.doc_bm25_db = bm25s.BM25(k1=1.5, b=0.75)
        self.doc_faiss_db = faiss.IndexFlatIP(self.doc_vec_size)
        self.stemmer = SnowballStemmer("russian")

    def average_pool(self, last_hidden_states, attention_mask):
        """
        Выполняет усреднение (mean pooling) с учётом маски.
        last_hidden_states: (batch_size, sequence_length, hidden_size)
        attention_mask: (batch_size, sequence_length)
        """
        # Обнуляем позиции, которые не входят в attention_mask
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # Усредняем по последовательности
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode_text(self, text, normalize_embeddings=True, device=None):
        """
        Кодирует текст, используя токенизатор и модель Transformers.
        Возвращает нормализованный эмбеддинг в виде numpy массива.
        """
        if device is None:
            device = self.emb_model_device
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        emb = self.average_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=1)
        # Для батча из одного примера возвращаем вектор размерности (embedding_dim,)
        return emb.cpu().numpy()[0]

    def fill(self, doc_base):
        """
        Заполнение поисковой базы и создание индексов.
        Parameters:
            doc_base : база документов по чанкам
        """
        self.doc_base = doc_base
        nltk.download('punkt_tab')  # Загружаем необходимые данные для токенизации

        doc_keys = list(self.doc_base.base.keys())
        doc_keys.sort()
        doc_corpus_tokens = [
            [self.stemmer.stem(word) for word in word_tokenize(self.doc_base.base[key].page_content.lower(), language='russian')]
            for key in doc_keys
        ]
        doc_corpus_docs = {key: self.doc_base.base[key].page_content for key in doc_keys}
        self.doc_corpus_token = doc_corpus_tokens

        # Индексирование для BM25
        self.doc_bm25_db.index(doc_corpus_tokens)
        
        vec = []
        for k in tqdm(doc_keys):
            # Получаем эмбеддинг для каждого документа через наш метод encode_text
            vec.append(self.encode_text(doc_corpus_docs[k], normalize_embeddings=True, device="cpu"))
        self.doc_faiss_db.add(np.array(vec))
        self.doc_bm25_db.save(f'{self.config.base_path}/doc_bm25_db.pkl')
        #self.doc_bm25_db.save(f'{self.config.base_path}/doc_bm25_db')
        faiss.write_index(self.doc_faiss_db, f"{self.config.base_path}/doc_faiss_db.bin")

    def load(self):
        """
        Загрузка базы поиска на основании сохраненных данных и индексов.
        """
        self.doc_base = Doc_Base(path=self.config.base_path)
        self.doc_base.load()
        self.doc_faiss_db = faiss.read_index(f"{self.config.base_path}/doc_faiss_db.bin")
        #self.doc_bm25_db = bm25s.BM25.load(f'{self.config.base_path}/doc_bm25_db')
        self.doc_bm25_db = BM25.load(f'{self.config.base_path}/doc_bm25_db.pkl')

    def rerank_doc(self, qst, docs, K):
        return self.doc_reranker.rank(qst, docs, K)

    def search_doc(self, qst, kt, doc_base):
        """
        Поиск документов по запросу:
         - Сначала ищем по BM25
         - Затем дополняем результаты поиском по эмбеддингам через FAISS
        """
        self.doc_base = doc_base
        self.config.reload()
        doc_tokens = [self.stemmer.stem(y) for y in word_tokenize(qst.lower(), language='russian')]
        doc_keys = list(self.doc_base.base.keys())
        doc_keys.sort()

        # Поиск по BM25
        doc_results_bm, doc_scores_bm = self.doc_bm25_db.retrieve([doc_tokens], corpus=doc_keys, k=kt)
        ls_id = set()
        for i in range(doc_results_bm.shape[1]):
            ids, score = doc_results_bm[0, i], doc_scores_bm[0, i]
            ls_id.add(ids)

        # Получаем эмбеддинг запроса с помощью метода encode_text
        vec = self.encode_text(qst, normalize_embeddings=True, device="cpu")
        doc_scores_fs, doc_results_fs = self.doc_faiss_db.search(vec.reshape((1, self.doc_vec_size)), k=kt)
        for l in doc_results_fs[0]:
            ls_id.add(l)

        ls_doc = [self.doc_base.base[k] for k in ls_id]

        return ls_doc

    # Doc(page_content = x, metadata = {'source':source, 'level0_name':level0_name, 'level0_id':level0_id, 'level1_name':level1_name, 'level1_id':level1_id, "order":k, 'id' = k} )
    def add_link(self, doc):
        meta = doc.metadata
        page_content = doc.page_content
        # pref = "Информация об источнике данных \nДокумент: " + meta.get('source', 'неизвестен') + "\n" + "Раздел: " +  meta.get('level0_name', 'неизвестен') + "\n" + "Пункт: " +  meta.get('level1_name', 'неизвестен') + "\n"
        return Doc(page_content=page_content, metadata=meta)

    def llm_chat(self, llm, doc_ls, qst):
        self.config.reload()
        llm_config = self.config
        #ls_result_new = self.expand(doc_ls, window=2)
        #ls_result_new = self.expand_lev(doc_ls)
        prompt2 = llm_config.prompt_template_response

        new_ls = [self.add_link(x).page_content for x in doc_ls]
        # print(new_ls)
        #new_ls = [self.x.page_content for x in doc_ls]
        prompt2 = prompt2.replace("{context}", '\n'.join(new_ls)).replace("{qst}", qst)
        #print(prompt2)
        #llm.change_temp(llm_config.temperature_response_gen)
        res = llm.invoke(prompt2)
        return res, doc_ls, prompt2
    
    def get_response(self, llm, qst, doc_base, doc_K1=30, doc_K2=5):
        """
            поиск ответа на вопрос
            сначала в базе вопросов и ответов с реранжированием
            потом в базе чанков документов с реранжированием
            и генерация ответа LLM
            Parameters:
                qst : вопрос
                K   : количество топ в поиске
            Returns:
                lm_res  : ответ LLM,
                qq1_0   : найденные вопросы до реранкера,
                res1    : найденные вопросы после реранкера,
                res2    : найденные ответы по вопросам,
                qq3     : найденные чанки документов,
                res3    : найденные чанки документов после реранкера
                exp_docs: найденные чанки документов после дополнения контекста
                prompt  : промпт для LLM
        """
        self.config.reload()

        qq3 = self.search_doc(qst, doc_K1, doc_base)
        res3 = self.rerank_doc(qst, qq3, doc_K2)

        # llm_res, exp_docs, prompt = search_base.llm_chat(llm, res3, qst)
        llm_res, exp_docs, prompt = self.llm_chat(llm, res3, qst)
        return llm_res, qq3, res3, exp_docs, prompt

    def get_doc(self, ids):
        return self.doc_base.get_by_id(ids)

    def expand(self, res_ls, window=1):
        """
            расширение контекста для найденных чанков
        """
        tmp_ls = []
        base_len = len(self.doc_base.base)
        for res in res_ls:
            doc_id = res.metadata.get('id', -1)
            src = res.metadata.get('source', '')
            ls_ids = [k for k in range(doc_id - window, doc_id + window + 1) if k >= 0 and k < base_len]
            ls_docs = [self.get_doc(k) for k in ls_ids]
            ls_docs = [doc for doc in ls_docs if src == doc.metadata.get('source', '')]
            # print(ls_docs)
            tmp_ls.append(all_doc_concat(*ls_docs))
        # print(tmp_ls)
        return tmp_ls

    def get_all_children(self, doc, N=10):
        """
            self.doc_base.base = base
            получение всех подпунктов документа
        """
        idx = doc.metadata.get('id', -1)
        src = doc.metadata.get('source', '')
        #level0_name = doc.metadata.get('level0_name', '')
        level1_name = doc.metadata.get('level1_name', '').replace(" ", '')
        ln = len(level1_name)
        ls_idx = range(idx, idx + N + 1)
        tmp_docs = [self.doc_base.base.get(k) for k in ls_idx]
        docs = [x for x in tmp_docs if x.metadata.get('source', '') == src and level1_name == x.metadata.get('level1_name', '')[:ln]]
        return docs

    # Doc(page_content = x, metadata = {'source':source, 'level0_name':level0_name, 'level0_id':level0_id, 'level1_name':level1_name, 'level1_id':level1_id, "order":k, 'id' = k} )
    def expand_lev(self, res_ls):
        """
            self.doc_base.base = base
            расширение контекста для найденных чанков до пункта
        """
        tmp_ls = []
        for res in res_ls:
            # print(res)
            doc_id = res.metadata.get('id', -1)
            src = res.metadata.get('source', '')
            #level0_name = res.metadata.get('level0_name', '')
            level1_name = res.metadata.get('level1_name', '')
            txt_len = len(res.page_content)
            ls_ch = []
            expanded = False
            if txt_len < 200:
                ls_ch = self.get_all_children(res, N=10)
                doc0 = all_doc_concat(*ls_ch)
                expanded = True
                # print(doc0)
            if expanded:
                ls_id = [key for key in self.doc_base.base.keys() if
                         (key < doc_id + 10) and (key > doc_id - 10) and (key != doc_id)]
                ls_docs = [doc0]
            else:
                ls_id = [key for key in self.doc_base.base.keys() if (key < doc_id + 10) and (key > doc_id - 10)]
                ls_docs = []

            for m in ls_id:
                doc = self.doc_base.base.get(m, Doc.empty())
                if doc.metadata.get('source', '') == src and doc.metadata.get('level1_name', '') == level1_name:
                    ls_docs.append(doc)

            tmp_ls.append(all_doc_concat(*ls_docs))
        return tmp_ls

    def view(self, ls_docs):
        for doc in ls_docs:
            print(doc)
            print(' --- ' * 3)


class EmptyLLM():
    """
     класс EmptyLLM
    """

    def __init__(self, config):
        self.config = config
        self.mod_name = 'Empty_model'
        self.llm = None
        self.tokenizer = None
        self.temperature = 0.2

    def invoke(self, prmt):
        search_config = self.config
        sys_prompt = search_config.system_prompt
        try:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prmt}]
            res0 = prmt
            res0 = "Проверка работы бота"
        except Exception as ex:
            print(ex)
            res0 = None
        return res0

    def change_temp(self, temp):
        pass


class RAG():
    def __init__(self, config):
        logger.info("RAG init")
        self.search_base = SearchBase(config)
        self.search_base.load()

        logger.info("базы загружены")
        # self.my_llm = EmptyLLM(config)


    def response(self, qst):
        logger.info("RAG response")
        additional_info = " ======= "
        full_res = self.search_base.get_response(llm=self.my_llm, qst=qst)
        logger.info(self.my_llm.temperature)
        logger.info(full_res[0])
        logger.info(doc_str(full_res[2]))
        logger.info(full_res[7])
        return full_res

