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
import bm25s
import faiss
from collections import OrderedDict
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import XLMRobertaTokenizerFast
import string

from FlagEmbedding import BGEM3FlagModel

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

'''
def level1_split(doc, tokenizer):
    """
    Разбивает документ по пунктам/подпунктам (первый уровень) и вычисляет global_token_offset для каждого чанка.

    Parameters:
         doc: Документ (объект Doc) с атрибутами page_content и metadata.
         tokenizer: Токенизатор, используемый для вычисления глобального смещения токенов.
    Returns:
         ls_doc: список объектов Doc, каждый из которых содержит текст раздела 
                 и в metadata добавлены ключи:
                   - "char_span": кортеж (start, end) глобальных символьных границ чанка,
                   - "global_token_offset": число токенов, предшествующих началу чанка.
    """
    txt = doc.page_content
    source = doc.metadata['source']

    # Регулярное выражение для разделителей вида "1. ", "1.1. ", "1.1.1. ", "1.1.1.1. "
    pat_02 = r"(\n\s*\d+\. )|(\n\s*\d+\.\d+\. )|(\n\s*\d+\.\d+\.\d+\. )|(\n\s*\d+\.\d+\.\d+\.\d+\. )"
    m_iterator = re.finditer(pat_02, txt)
    l_match = list(m_iterator)
    # Если разделителей нет, считаем весь текст одним чанком:
    metadata0 = {'source': source, 'level1_name': '', 'level1_id': '', 'char_span': (0, len(txt))}
    # Для полного текста смещение равно 0
    metadata0["global_token_offset"] = 0

    if len(l_match) > 0:
        ls_tec = []
        for m in l_match:
            idx = idx_match(m.groups()) 
            group = m.group().replace("\n", "")
            span = m.span()  # span[0] – начало разделителя
            metadata = {'source': source, 'level1_name': group, 'level1_id': idx, 'start': span[0]}
            ls_tec.append((span[0], metadata))
        
        # Подготавливаем пары: первая точка – начало чанка, вторая – начало следующего разделителя
        ls_tec1 = [(0, metadata0)] + ls_tec[:-1]
        ls2 = list(zip(ls_tec1, ls_tec))
        
        ls_doc = []
        for (start, meta_start), (next_start, _) in ls2:
            chunk_span = (start, next_start)
            chunk_length = next_start - start
            # Если не первый чанк и его длина меньше 500 символов – объединяем с предыдущим чанком
            if ls_doc and chunk_length < 500:
                prev_doc = ls_doc[-1]
                prev_start = prev_doc.metadata['char_span'][0]
                new_span = (prev_start, next_start)
                new_text = txt[prev_start:next_start]
                # Обновляем предыдущий документ: расширяем его границы и текст
                prev_doc.page_content = new_text
                prev_doc.metadata['char_span'] = new_span
                # global_token_offset остаётся без изменений (от начала предыдущего чанка)
            else:
                new_meta = meta_start.copy()
                new_meta['char_span'] = chunk_span
                # Вычисляем global_token_offset: число токенов до начала чанка
                tokens_before = tokenizer(txt[:start], add_special_tokens=False)['input_ids']
                new_meta["global_token_offset"] = len(tokens_before)
                chunk_text = txt[start:next_start]
                ls_doc.append(Doc(page_content=chunk_text, metadata=new_meta))
    else:
        ls_doc = [Doc(page_content=txt, metadata=metadata0)]
    return ls_doc
'''

def level0_split(doc, tokenizer):
    """
    Разбивает документ по нулевому уровню (верхнеуровневые блоки — статьи) и вычисляет global_token_offset для каждого чанка.
    
    Parameters:
         doc: Документ (объект Doc) с атрибутами page_content и metadata.
         tokenizer: Токенизатор, используемый для вычисления глобального смещения токенов.
    Returns:
         ls_doc: список объектов Doc, каждый из которых содержит текст статьи 
                 и в metadata добавлены ключи:
                   - "char_span": кортеж (start, end) глобальных символьных границ статьи,
                   - "global_token_offset": число токенов, предшествующих началу статьи,
                   - "level0_name": название статьи,
                   - "level0_id": идентификатор статьи.
    """
    txt = doc.page_content
    source = doc.metadata['source']
    
    # Ищем строки, начинающиеся с "Статья" (с возможными пробелами) и захватываем номер и заголовок до конца строки.
    pat_01 = r"^\s*Статья\s+(\d+(?:\.\d+)*\.\s*.+)$"
    m_iterator = re.finditer(pat_01, txt, flags=re.MULTILINE)
    ls_tec = []
    
    # Базовые метаданные для предисловия (если оно есть)
    metadata0 = {
        'source': source, 
        'level0_name': '', 
        'level0_id': '', 
        'char_span': (0, len(txt)),
        "global_token_offset": 0
    }
    
    # Собираем найденные статьи: group(1) содержит номер и заголовок.
    for m in m_iterator:
        idx = idx_match(m.groups())
        group = m.group(1).strip()
        span = m.span()  # начало найденного совпадения
        metadata = {'source': source, 'level0_name': group, 'level0_id': idx}
        ls_tec.append((span[0], metadata))
    
    # Если статьи не найдены – возвращаем весь текст как один чанк.
    if not ls_tec:
        # Вычисляем global_token_offset для всего текста (будет 0)
        tokens_before = tokenizer(txt[:0], add_special_tokens=False)['input_ids']
        metadata0["global_token_offset"] = len(tokens_before)
        return [Doc(page_content=txt, metadata=metadata0)]
    else:
        metadata0['char_span'] = (0, ls_tec[0][0])
    
    # Добавляем конечную метку, чтобы захватить текст от последнего совпадения до конца документа
    ls_tec.append((len(txt) + 1000, metadata0))
    
    # Формируем пары: первый чанк – от начала документа до первого найденного разделителя, затем между статьями.
    ls_tec1 = [(0, metadata0)] + ls_tec
    ls2 = list(zip(ls_tec1, ls_tec))
    
    ls_doc = []
    for (start, meta_start), (next_start, _) in ls2:
        chunk_span = (start, next_start)
        new_meta = meta_start.copy()
        new_meta['char_span'] = chunk_span
        # Вычисляем global_token_offset: число токенов от начала документа до позиции start
        tokens_before = tokenizer(txt[:start], add_special_tokens=False)['input_ids']
        new_meta["global_token_offset"] = len(tokens_before)
        chunk_text = txt[start:next_start]
        ls_doc.append(Doc(page_content=chunk_text, metadata=new_meta))
    #print(f"Найдено {len(ls_doc)} статей!")
    return ls_doc


def level1_split(doc, tokenizer):
    """
    Разбивает документ (чанк, полученный из level0_split) по пунктам/подпунктам (первый уровень) и вычисляет global_token_offset для каждого подраздела.
    Если doc уже содержит метаданные родительского уровня (например, char_span и global_token_offset),
    то при вычислении глобального смещения для подраздела используется базовый offset родителя.
    
    Parameters:
         doc: Документ (объект Doc) с атрибутами page_content и metadata, полученный из level0_split.
              metadata содержит: 'level0_name', 'level0_id', 'char_span' — глобальные границы родительского чанка,
              'global_token_offset' — число токенов от начала документа до начала родительского чанка.
         tokenizer: Токенизатор, используемый для вычисления глобального смещения токенов.
    Returns:
         ls_doc: список объектов Doc, каждый из которых содержит текст подраздела 
                 и в metadata добавлены ключи:
                   - "char_span": кортеж (global_start, global_end) глобальных символьных границ подраздела,
                   - "global_token_offset": число токенов от начала документа до подраздела.
                   Кроме того, сохраняются родительские метаданные уровня 0 (level0_name, level0_id).
    """
    txt = doc.page_content
    source = doc.metadata['source']
    level0_name = doc.metadata.get('level0_name', '')
    level0_id = doc.metadata.get('level0_id', '')
    
    # Извлекаем глобальные границы родительского чанка и его global_token_offset.
    parent_char_span = doc.metadata.get('char_span', (0, len(txt)))
    parent_global_token_offset = doc.metadata.get("global_token_offset", 0)

    if doc.metadata.get("level0_name"):
        meta_start = {
            'source': source, 
            'level0_name': level0_name, 
            'level0_id': level0_id, 
            'level1_name': '',
            'level1_id': '',
            'char_span': parent_char_span,
            'global_token_offset': parent_global_token_offset
        }
        return [Doc(page_content=txt, metadata=meta_start)]

    # Регулярное выражение для поиска разделителей подразделов (пунктов)
    pat_02 = r"(\n\s*\d+\. )|(\n\s*\d+\.\d+\. )|(\n\s*\d+\.\d+\.\d+\. )|(\n\s*\d+\.\d+\.\d+\.\d+\. )"
    m_iterator = re.finditer(pat_02, txt)
    l_match = list(m_iterator)
    
    metadata0 = {
        'source': source,
        'level0_name': level0_name, 
        'level0_id': level0_id, 
        'level1_name': '', 
        'level1_id': '', 
        'char_span': (parent_char_span[0], parent_char_span[1]) 
    }

    metadata0["global_token_offset"] = parent_global_token_offset
    
    # Если разделителей не найдено, возвращаем весь текст как один подраздел
    if not l_match:
        return [Doc(page_content=txt, metadata=metadata0)]
    else:
        first_point_start = l_match[0].span()[0]
        metadata0['char_span'] = (parent_char_span[0], parent_char_span[0] + first_point_start)
    
    ls_tec = [] 
    for m in l_match:
        idx = idx_match(m.groups())
        group = m.group().replace("\n", "").strip()
        span = m.span()  # Локальное смещение в тексте родительского чанка
        metadata = {
            'source': source, 
            'level0_name': level0_name, 
            'level0_id': level0_id, 
            'level1_name': group, 
            'level1_id': idx, 
            'start': span[0]
        }
        ls_tec.append((span[0], metadata))
    
    # Формируем пары: от начала текста до первого разделителя, затем между разделителями.
    ls_tec1 = [(0, metadata0)] + ls_tec[:-1]
    ls2 = list(zip(ls_tec1, ls_tec))
    ls_doc = []
    
    for (start, meta_start), (next_start, _) in ls2:
        # Преобразуем локальные смещения в глобальные, прибавляя начало родительского чанка.
        global_start = parent_char_span[0] + start
        global_end = parent_char_span[0] + next_start
        global_char_span = (global_start, global_end)
        #chunk_length = next_start - start
        # Если не первый подраздел и его длина меньше 500 символов – объединяем с предыдущим
        '''
        if ls_doc and chunk_length < 500:
            prev_doc = ls_doc[-1]
            prev_start = prev_doc.metadata['char_span'][0]
            new_span = (prev_start, global_end)
            # Вычисляем локальный диапазон для объединения:
            local_prev_start = prev_doc.metadata['char_span'][0] - parent_char_span[0]
            new_text = txt[local_prev_start: next_start]
            prev_doc.page_content = new_text
            prev_doc.metadata['char_span'] = new_span
            # global_token_offset остаётся прежним (начало предыдущего подраздела)
        else:
        '''
        new_meta = meta_start.copy()
        new_meta['char_span'] = global_char_span
        # Вычисляем global_token_offset для подраздела:
        tokens_before = tokenizer(txt[:start], add_special_tokens=False)['input_ids']
        new_meta["global_token_offset"] = parent_global_token_offset + len(tokens_before)
        chunk_text = txt[start:next_start]
        ls_doc.append(Doc(page_content=chunk_text, metadata=new_meta))
    #print(f"Найдено {len(ls_doc)} пунктов!")
    return ls_doc


def level2_split(doc, text_splitter, tokenizer):
    """
    Делит документ (чанк первого уровня, полученный из level1_split) на более мелкие части,
    вычисляя для каждого подчанка его границы в символах и преобразовывая их в индексы токенов с использованием токенизатора.
    Границы token_span возвращаются в глобальной системе (аналогично char_span).
    
    Parameters:
        doc: Объект Doc, содержащий:
            - page_content: текст чанка (часть исходного текста, полученная на уровне 1)
            - metadata: словарь, содержащий по крайней мере ключ 'source' и 'level1_name', 'level1_id',
                        а также 'char_span' — кортеж (global_start, global_end) для этого чанка.
                        Опционально может содержать 'global_token_offset' — сдвиг токенов для этого чанка.
        text_splitter: Объект с методом split_text(text) для разбиения текста на подчанки.
                       Должен также иметь атрибут chunk_overlap (количество символов перекрытия).
        tokenizer: Токенизатор, поддерживающий параметр return_offsets_mapping=True.
    
    Returns:
        ls_doc: Список объектов Doc, каждый из которых представляет подчанок с дополненными метаданными:
                - 'char_span': глобальные границы подчанка,
                - 'token_span': кортеж (token_start, token_end) в глобальной индексации токенов,
                - 'token_count': число токенов в подчанке.
    """
    txt = doc.page_content  
    source = doc.metadata.get('source', '')
    level0_name = doc.metadata.get('level0_name', '')
    level0_id = doc.metadata.get('level0_id', '')
    level1_name = doc.metadata.get('level1_name', '')
    level1_id = doc.metadata.get('level1_id', '')
    parent_char_span = doc.metadata.get('char_span', (0, len(txt)))
    parent_char_start = parent_char_span[0]
    global_token_offset = doc.metadata.get("global_token_offset", 0)

    # Получаем offsets mapping для всего текста чанка первого уровня.
    parent_encoded = tokenizer(txt, return_offsets_mapping=True)
    parent_offsets = parent_encoded.offset_mapping  # список кортежей (token_start_char, token_end_char)

    # Если metadata содержит подраздел (level1_name не пуст), то не делим дальше, иначе – используем text_splitter.
    if doc.metadata.get("level0_name"):
        ls_txt = [txt]
    else:
        ls_txt = text_splitter.split_text(txt)

    ls_doc = []
    current_pos = 0  # локальная позиция в txt
    for k, chunk in enumerate(ls_txt):
        local_start = current_pos
        local_end = current_pos + len(chunk)
        # С учётом перекрытия между подчанками
        if k < len(ls_txt) - 1:
            current_pos = local_end - 300
        #elif (local_end - local_start) < 1000:
        #    break 
        else:
            current_pos = local_end
        '''
        else:
            if (local_end - local_start) < 1000:
                break
            current_pos = local_end
        '''

        global_start = parent_char_start + local_start
        global_end = parent_char_start + local_end

        token_start = None
        token_end = None
        for i, (tok_start, tok_end) in enumerate(parent_offsets):
            if token_start is None and tok_start >= local_start:
                token_start = i
            if tok_end <= local_end:
                token_end = i + 1
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = len(parent_offsets)
        token_count = token_end - token_start

        token_span_global = (token_start + global_token_offset, token_end + global_token_offset)

        metadata = {
            'source': source,
            'level0_name': level0_name, 
            'level0_id': level0_id, 
            'level1_name': level1_name,
            'level1_id': level1_id,
            'order': k,
            'char_span': (global_start, global_end),
            'token_span': token_span_global,
            'token_count': token_count
        }
        ls_doc.append(Doc(page_content=chunk, metadata=metadata))
    return ls_doc


'''
def level2_split(doc, text_splitter, tokenizer):
    """
    Делит документ (чанк первого уровня, полученный, например, из level1_split) на более мелкие части,
    вычисляя для каждого подчанка его границы в символах и преобразовывая их в индексы токенов с использованием
    токенизатора. Границы token_span возвращаются в глобальной системе (аналогично char_span).

    Parameters:
        doc: Объект Doc, содержащий:
            - page_content: текст чанка (часть исходного текста, полученная на уровне 1)
            - metadata: словарь, содержащий по крайней мере ключ 'source' и 'level1_name', 'level1_id',
                        а также 'char_span' — кортеж (global_start, global_end) для этого чанка в исходном тексте.
                        Опционально может содержать 'global_token_offset' — сдвиг токенов для этого чанка.
        text_splitter: Объект, реализующий метод split_text(text) для разбиения текста на подчанки.
                       Должен также иметь атрибут chunk_overlap (количество символов перекрытия).
        tokenizer: Токенизатор модели, поддерживающий параметр return_offsets_mapping=True при токенизации.

    Returns:
        ls_doc: Список объектов Doc, каждый из которых представляет подчанок с дополненными метаданными:
                - 'char_span': абсолютные границы подчанка в исходном тексте,
                - 'token_span': кортеж (token_start, token_end) — границы подчанка в глобальной индексации токенов,
                - 'token_count': количество токенов в подчанке.
    """
    # Текст чанка первого уровня
    txt = doc.page_content  
    source = doc.metadata.get('source', '')
    level1_name = doc.metadata.get('level1_name', '')
    level1_id = doc.metadata.get('level1_id', '')
    # Глобальные (абсолютные) границы чанка первого уровня в исходном тексте.
    parent_char_span = doc.metadata.get('char_span', (0, len(txt)))
    parent_char_start = parent_char_span[0]
    
    # Сдвиг для глобальной токеновой индексации (если не указан, считаем, что локальные токены уже глобальные)
    global_token_offset = doc.metadata.get("global_token_offset", 0)

    # Получаем offsets mapping для всего текста чанка первого уровня.
    # Эти offsets (от начала doc.page_content) позволяют сопоставить символьные позиции с токеновыми индексами.
    parent_encoded = tokenizer(txt, return_offsets_mapping=True)
    parent_offsets = parent_encoded.offset_mapping  # список кортежей (token_start_char, token_end_char)

    # Если в metadata уже есть пункт, то делим текст на один чанк, иначе используем text_splitter.
    if doc.metadata.get("level1_name"):
        ls_txt = [txt]
    else: 
        ls_txt = text_splitter.split_text(txt)

    ls_doc = []
    current_pos = 0  # локальная позиция внутри txt
    for k, chunk in enumerate(ls_txt):
        local_start = current_pos
        local_end = current_pos + len(chunk)
        # Если не последний чанк — с учётом перекрытия
        if k < len(ls_txt) - 1:
            current_pos = local_end - 300
        else:
            if (local_end - local_start) < 1000:
                break
            current_pos = local_end

        # Вычисляем глобальные границы подчанка в исходном тексте:
        global_start = parent_char_start + local_start
        global_end = parent_char_start + local_end

        # Находим локальные токеновые границы относительно doc.page_content
        token_start = None
        token_end = None
        for i, (tok_start, tok_end) in enumerate(parent_offsets):
            # Первый токен, начинающийся не раньше, чем local_start
            if token_start is None and tok_start >= local_start:
                token_start = i
            # Токены полностью попадают в подчанок: выбираем последний, у которого конец не превышает local_end
            if tok_end <= local_end:
                token_end = i + 1  # токен с индексом i включается в диапазон
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = len(parent_offsets)
        token_count = token_end - token_start

        # Переводим локальные токеновые границы в глобальные, добавляя сдвиг
        token_span_global = (token_start + global_token_offset, token_end + global_token_offset)

        metadata = {
            'source': source,
            'level1_name': level1_name,
            'level1_id': level1_id,
            "order": k,
            'char_span': (global_start, global_end),
            'token_span': token_span_global,
            'token_count': token_count
        }
        ls_doc.append(Doc(page_content=chunk, metadata=metadata))
    return ls_doc
'''

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

    def __init__(self, config, name='doc_base', path="./data"):
        self.config = config

        self.name = name
        self.path = path
        self.fl_ls = None
        self.txt_files_names = None
        self.txt_files_dict = None
        self.doc_path = None
        self.full_docs = None
        self.base = None

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.config.late_chunking_tokenizer_name)

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
            "law_43.txt": "Инструкция Банка России от 30.06.2021 № 204-И «Об открытии, ведении и закрытии банковских счетов и счетов по вкладам (депозитам)»",
            "law_44.txt": "Решение Минфина России от 31.01.2025 N 25-67381-01850-Р<О порядке предоставления субсидии АО ДОМ.РФ для возмещения российским кредитным организациям и АО ДОМ.РФ недополученных доходов по выданным (приобретенным) жилищным (ипотечным) кредитам (займам), предоставленным гражданам РФ, имеющим детей (Версия 5)>"
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
        pattern_ot = r'(?m:^(?:от.*(?:\r?\n|$)){2,})'
        patterns = [pattern_date, pattern_dashes, pattern_caps, pattern_consultant, pattern_ot]
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
        # Инициализируем сплиттер с заданными параметрами:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

        ls_doc_01 = []
        for doc in self.full_docs:
            rr = level0_split(doc, self.tokenizer)
            ls_doc_01 += rr

        ls_doc_02 = []
        for doc in ls_doc_01:
            rr = level1_split(doc, self.tokenizer)
            ls_doc_02 += rr

        ls_doc_03 = []
        for doc in ls_doc_02:
            rr = level2_split(doc, text_splitter, self.tokenizer)
            ls_doc_03 += rr

        self.base = {}
        for k, d in enumerate(ls_doc_03):
            metadata = d.metadata.copy()
            metadata['id'] = k
            pref = (
                "## Документ: " + metadata.get('source', 'неизвестен') + "\n" +
                "## Статья: " + metadata.get('level0_name', 'неизвестен') + "\n" +
                "## Пункт или подпункт: " + metadata.get('level1_name', 'неизвестен') +
                " <#####>\n"
            )
            pref_len = len(pref)
            metadata['pref_len'] = pref_len 
            self.base[k] = Doc(page_content=(pref + d.page_content), metadata=metadata)

        print(f'splited: {len(self.base)}')
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
        # model_path = "/home/sav/work/DBRA_RAG/rank_model"
        self.model_path = model_path
        # 'DiTy/cross-encoder-russian-msmarco'
        if device != 'cuda':
            self.reranker_model = CrossEncoder(self.model_path, max_length=512, device='cpu')
        else:
            self.reranker_model = CrossEncoder(self.model_path, max_length=512, device='cuda')
        self.device = device

    def rank(self, qst, documents, k=5):
        rank_result = self.reranker_model.rank(qst, [x.page_content for x in documents])
        res_ls = [(x['score'], documents[x['corpus_id']]) for x in rank_result[:k]]
        scores, res_docs = zip(*res_ls)
        # for doc in res_docs:
        #     if doc.metadata.get('is_empty', False):
        #         print(qst)
        return res_docs


class SearchBase():
    """
        класс поиск в базе               
        готовим на основе базы документов
    """ 

    def __init__(self, config):
        self.config = config

        self.doc_base = None
        self.doc_corpus_token = None
        self.emb_model_device = self.config.emb_model_device

        self.doc_emb_model_name = self.config.doc_emb_model_name

        self.reranker_device = self.config.reranker_device

        self.doc_reranker = Reranker(self.reranker_device, self.config.doc_reranker_name) 

        self.doc_bm25_db = bm25s.BM25(k1=1.5, b=0.75)
        self.stemmer = SnowballStemmer("russian")

        self.late_chunk_embedder = BGEM3FlagModel(self.config.late_chunk_model_name, pooling_method='mean')


    def fill(self, doc_base):
        """
           заполнение поисковой базы и создание индексов
           Parameters:
            doc_base : база документов по чанкам
        """
        self.doc_base = doc_base

        nltk.download('punkt_tab')

        doc_keys = list(self.doc_base.base.keys())
        doc_keys.sort()
        doc_corpus_tokens = [[self.stemmer.stem(y) for y in
                              word_tokenize(self.doc_base.base[key].page_content.lower(), language='russian')] for key
                             in doc_keys]
        self.doc_corpus_token = doc_corpus_tokens

        self.doc_bm25_db.index(doc_corpus_tokens)
        self.doc_bm25_db.save(f'{self.config.base_path}/doc_bm25_db')


    def load(self):
        """
           загрузка базы поиска на основании сохраненных данных и сохраненных индексов
        """
        self.doc_base = Doc_Base(config=self.config, path=self.config.base_path)
        self.doc_base.load()
        self.doc_bm25_db = bm25s.BM25.load(f'{self.config.base_path}/doc_bm25_db')

    def rerank_doc(self, qst, docs, K):
        return self.doc_reranker.rank(qst, docs, K)
    
    def search_doc(self, qst, kt, doc_base):
        self.doc_base = doc_base
        self.config.reload()
        doc_tokens = [self.stemmer.stem(y) for y in word_tokenize(qst.lower(), language='russian')]
        doc_keys = list(self.doc_base.base.keys())
        doc_keys.sort()

        doc_results_bm, doc_scores_bm = self.doc_bm25_db.retrieve([doc_tokens], corpus=doc_keys, k=kt)
        ls_id = set()
        for i in range(doc_results_bm.shape[1]):
            ids, score = doc_results_bm[0, i], doc_scores_bm[0, i]
            ls_id.add(ids)

        ls_doc = [self.doc_base.base[k] for k in ls_id]

        late_chunk_res = self.search_doc_late(qst, self.doc_base)
        for elem in late_chunk_res:
            ls_doc.append(elem)

        return ls_doc

    def search_doc_late(self, qst, doc_base):
        """
        Метод поиска с использованием алгоритма late chunking с учётом полного документа (source).

        Для каждого полного документа (source):
          - Берётся полный текст из doc_base.full_docs.
          - Вычисляются токеновые эмбеддинги для полного текста.
          - Для каждого чанка (или его сегмента, если число токенов чанка больше max_seq_len),
            извлекается pooled-эмбеддинг на основе token_span.
          - Вычисляется сходство pooled-эмбеддингов с эмбеддингом запроса.
        
        В итоге выбираются топ-5 сегментов с наивысшей схожестью.

        Параметры:
            qst     - строка запроса.
            doc_base- база документов, где:
                      - doc_base.base – словарь с чанками,
                      - doc_base.full_docs – словарь, сопоставляющий source с полным текстом документа.
        Возвращает:
            Список из 5 найденных чанков (объектов Doc) с наивысшей схожестью.
        """
        self.doc_base = doc_base
        self.config.reload()
        max_seq_len = self.config.max_seq_len  # максимально допустимое число токенов на сегмент

        # Группируем чанки по source
        source_to_chunks = {}
        for key, doc in self.doc_base.base.items():
            src = doc.metadata.get("source", "")
            if src not in source_to_chunks:
                source_to_chunks[src] = []
            source_to_chunks[src].append(doc)

        candidate_chunks = []  # список вида (doc, pooled_embedding)

        # Формируем словарь, сопоставляющий source с полным текстом, исходя из doc_base.full_docs (список объектов Doc)
        source_to_full = {}
        for full_doc in self.doc_base.full_docs:
            src = full_doc.metadata.get("source", "")
            source_to_full[src] = full_doc.page_content

        # Для каждого полного документа получаем его эмбеддинги один раз
        for src, chunks in source_to_chunks.items():
            # Извлекаем полный текст документа из full_docs
            if src not in source_to_full:
                continue  # если для данного source нет полного текста, пропускаем его
            full_doc_text = source_to_full[src]
            
            # Вычисляем токеновые эмбеддинги для полного документа
            full_doc_token_embeds = self.late_chunk_embedder.encode(
                [full_doc_text],
                max_length=8192,
                return_colbert_vecs=True
            )

            # Добавляем измерение батча (shape: (1, n_tokens, emb_dim))
            full_doc_token_embeds = torch.from_numpy(full_doc_token_embeds['colbert_vecs'][0]).unsqueeze(0)

            # Для каждого чанка из данного документа извлекаем pooled-эмбеддинг по его token_span
            for doc in chunks:
                token_span = doc.metadata.get("token_span", None)
                token_count = doc.metadata.get("token_count", None)
                if token_span is None or token_count is None:
                    continue

                if token_count > int(max_seq_len):
                    for seg_start in range(token_span[0], token_span[1], int(max_seq_len)):
                        seg_end = min(seg_start + int(max_seq_len), token_span[1])
                        # Выполняем pooling: суммирование эмбеддингов по заданному диапазону с делением на число токенов сегмента
                        pooled = full_doc_token_embeds[0, seg_start:seg_end].sum(dim=0) / (seg_end - seg_start)
                        candidate_chunks.append((doc, pooled))
                else:
                    seg_start, seg_end = token_span
                    pooled = full_doc_token_embeds[0, seg_start:seg_end].sum(dim=0) / (seg_end - seg_start)
                    candidate_chunks.append((doc, pooled))

        query_embeddings = self.late_chunk_embedder.encode(
            [qst],
            max_length=8192,
            return_colbert_vecs=True
        )
    
        query_embedding_pooled = np.mean(query_embeddings['colbert_vecs'][0], axis=0)

        def cos_sim(x, y):
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            if norm_x == 0 or norm_y == 0:
                return 0.0
            return np.dot(x, y) / (norm_x * norm_y)

        # Вычисляем сходство каждого pooled-эмбеддинга с эмбеддингом запроса
        similarities = []
        for doc, pooled_emb in candidate_chunks:
            # Приводим pooled_emb к numpy, если это тензор
            if isinstance(pooled_emb, torch.Tensor):
                pooled_emb_np = pooled_emb.detach().cpu().numpy()
            else:
                pooled_emb_np = pooled_emb

            sim = cos_sim(query_embedding_pooled, pooled_emb_np)
            similarities.append(sim)

        if len(similarities) == 0:
            return []

        top_indices = [int(i) for i in np.argsort(similarities)[-30:][::-1].flatten()]
        top_chunks = [candidate_chunks[i][0] for i in top_indices]

        return top_chunks

    # Doc(page_content = x, metadata = {'source':source, 'level0_name':level0_name, 'level0_id':level0_id, 'level1_name':level1_name, 'level1_id':level1_id, "order":k, 'id' = k} )
    def add_link(self, doc):
        meta = doc.metadata
        page_content = doc.page_content
        # pref = "Информация об источнике данных \nДокумент: " + meta.get('source', 'неизвестен') + "\n" + "Раздел: " +  meta.get('level0_name', 'неизвестен') + "\n" + "Пункт: " +  meta.get('level1_name', 'неизвестен') + "\n"
        return Doc(page_content=page_content, metadata=meta)

    def llm_chat(self, llm, doc_ls, qst):
        self.config.reload()
        llm_config = self.config
        # ls_result_new = self.expand(doc_ls, window=2)
        ls_result_new = self.expand_lev(doc_ls)
        prompt2 = llm_config.prompt_template_response

        new_ls = [self.add_link(x).page_content for x in ls_result_new]
        # print(new_ls)
        prompt2 = prompt2.replace("{context}", '\n'.join(new_ls)).replace("{qst}", qst)
        # print(prompt2)
        llm.change_temp(llm_config.temperature_response_gen)
        res = llm.invoke(prompt2)
        return res, ls_result_new, prompt2
    
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

