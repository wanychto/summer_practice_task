import pandas as pd
import numpy as np
from transliterate import translit
from symspellpy import SymSpell, Verbosity #исправление опечаток
from Levenshtein import distance as lev_distance
import jellyfish #фонетическое сравнение
from collections import defaultdict
import sqlite3
import re #регулярные выражения
import os
import unicodedata
from datetime import datetime

def load_csv_safe(filename): #для безопасной загрузки цсв файлов
    try:
        return pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

class NameDeduplicator:
    def __init__(self, person_df=None):
        self.name_similarity_threshold = 0.85
        self.min_cluster_size = 2
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = 'english_dict.txt' 
        if dictionary_path is os.listdir():
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.conn = sqlite3.connect('deduplicated_authors.db')
        self.create_database()
        self.person_df = person_df
        self.db_path = 'deduplicated_authors.db'
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Подключение к базе данных {self.db_path} успешно")
            self.create_database()
        except sqlite3.Error as e:
            print(f"Ошибка подключения к базе данных: {e}")
            self.conn = None
    

    def create_database(self):
        """Создает структуру базы данных для хранения кластеров дубликатов"""
        cursor = self.conn.cursor()
        
        # Таблица с кластерами дубликатов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS author_clusters (
            cluster_id INTEGER PRIMARY KEY,
            canonical_name TEXT,
            variant_count INTEGER,
            created_at TIMESTAMP
        )
        ''')
        
        # Таблица с вариантами имен
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS name_variants (
            variant_id INTEGER PRIMARY KEY,
            cluster_id INTEGER,
            raw_name TEXT,
            normalized_name TEXT,
            source_ids TEXT,
            FOREIGN KEY (cluster_id) REFERENCES author_clusters (cluster_id)
        )
        ''')
        
        self.conn.commit()
    def normalize_names(self, name):
        try:
            name = translit(name, 'en', reversed=True)
        except:
            pass
            
        if pd.isna(name) or name is None: #обнаруживает пропуски в таблице
            return ""
        if isinstance(name, (float, int)): #проверка типа переменной
            name = str(int(name)) if isinstance(name, float) and name.is_integer() else str(name)
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8').lower()
        name = re.sub(r'[^a-z\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    def preprocess_data(self):
        if not hasattr(self, 'person_df') or self.person_df is None: #проверка атрибута у объекта
            raise ValueError("Данные не загружены (person_df is None)")
            
        if not self.conn:
            print("Невозможно обработать данные - нет подключения к БД")
            return
            
        if self.person_df is None:
            print("Ошибка: Не загружена таблица person")
            return
        if 'fio' not in self.person_df.columns:
            raise ValueError("Таблица должна содержать колонку 'fio'")
        
        # Нормализуем имена
        self.person_df['fio_clean'] = self.person_df['fio'].apply(self.normalize_names)
        
        # Кластеризация имен
        clustered_df= self.cluster_similar_names()
        
        if clustered_df.empty:
            print("Дубликаты не найдены")
            return pd.DataFrame()
        try:
            self.save_clusters_to_db(clustered_df)
            print(f"Найдено {len(clustered_df['cluster_id'].unique())} кластеров дубликатов")
            print(f"Результаты сохранены в {self.db_path}")
            
            # Проверяем что данные записались
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM author_clusters")
            count = cursor.fetchone()[0]
            print(f"Всего кластеров в базе: {count}")
            
            cursor.execute("SELECT COUNT(*) FROM name_variants")
            count = cursor.fetchone()[0]
            print(f"Всего вариантов имен в базе: {count}")
            
        except Exception as e:
            print(f"Ошибка при сохранении в базу данных: {e}")   
    def calculate_name_similarity(self, name1, name2):
        #Вычисляет комбинированную метрику схожести имен
        name1 = self.normalize_names(name1)
        name2 = self.normalize_names(name2)
        
        if not name1 or not name2:
            return 0.0
        
        # 1. Расстояние Левенштейна
        lev_sim = 1 - (lev_distance(name1, name2) / max(len(name1), len(name2)))
        # 2. Фонетическое сходство (Soundex)
        soundex1 = jellyfish.soundex(name1)
        soundex2 = jellyfish.soundex(name2)
        phonetic_sim = 1.0 if soundex1 == soundex2 else 0.0
         # 3. Сходство N-грамм- последовательность из n элементов в тексте, совпадение по косинусу
        n = 2  # биграммы
        ngrams1 = set([name1[i:i+n] for i in range(len(name1)-n+1)])
        ngrams2 = set([name2[i:i+n] for i in range(len(name2)-n+1)])
        if not ngrams1 and not ngrams2:  # Оба набора пустые
            ngram_sim = 1.0  # Считаем одинаковыми
        elif not ngrams1 or not ngrams2:  # Один набор пустой
            ngram_sim = 0.0
        else:
            intersection = len(ngrams1 & ngrams2) #пересечение
            max_len = max(len(ngrams1), len(ngrams2))
            ngram_sim = intersection / max_len
        
        # Взвешенная комбинация метрик
        total_sim = 0.5*lev_sim + 0.3*phonetic_sim + 0.2*ngram_sim #!!!!не уверена в пропорциях
        return total_sim
    def cluster_similar_names(self):
        # Группирует похожие имена в кластеры
        if self.person_df is None:
            return pd.DataFrame()
            
        # Создаем временный DataFrame для работы
        names_df = self.person_df[['fio']].copy()
        names_df['original_index'] = names_df.index
        names_df['cluster_id'] = -1
        
        # Блокировка по первым буквам для оптимизации
        blocks = defaultdict(list)
        for idx, row in names_df.iterrows(): #создаёт ключи для имён по первым 3м буквам
            name = self.normalize_names(row['fio'])
            block_key = name[:3] if len(name) >= 3 else name
            blocks[block_key].append(idx)
        
        # Поиск дубликатов внутри блоков
        current_clusters_id = 0
        visited = set()
        
        for block, indices in blocks.items():
            for i in range(len(indices)):
                if indices[i] in visited:
                    continue
                current_cluster = [indices[i]]
                current_name = names_df.iloc[indices[i]]['fio']
                
                for j in range(i+1, len(indices)):
                    candidate_name = names_df.iloc[indices[j]]['fio']
                    similarity = self.calculate_name_similarity(current_name, candidate_name)
                    
                    if similarity >= self.name_similarity_threshold:
                        current_cluster.append(indices[j])
                        visited.add(indices[j])
                
                if len(current_cluster) >= self.min_cluster_size:
                    names_df.loc[current_cluster, 'cluster_id'] = current_clusters_id
                    current_clusters_id +=1
        
        return names_df[names_df['cluster_id']!=-1]
    def save_clusters_to_db(self, clustered_df):
        # Сохраняет кластеры дубликатов в базу данных
        cursor = self.conn.cursor()
        
        for cluster_id, group in clustered_df.groupby('cluster_id'):
            # Выбираем каноническое имя (самое частое или самое длинное)
            names = group['fio'].tolist()
            canonical_name = max(set(names), key=lambda x: (names.count(x), len(x)))
            # Добавляем кластер в базу
            cursor.execute('''
            INSERT INTO author_clusters (canonical_name, variant_count, created_at)
            VALUES (?, ?, ?)
            ''', (canonical_name, len(group), datetime.now()))
            
            cluster_id = cursor.lastrowid #получаем айди только что добавленной надписи
            
            # Добавляем варианты имен
            for _, row in group.iterrows():
                source_ids = row.get('original_index')
                cursor.execute('''
                INSERT INTO name_variants (cluster_id, raw_name, normalized_name, source_ids)
                VALUES (?, ?, ?, ?)
                ''', (cluster_id, row['fio'], self.normalize_names(row['fio']), str(source_ids) if not pd.isna(source_ids) else ''))
        
        self.conn.commit()
    def process_data(self):
        # Основной метод обработки данных
        if not hasattr(self, 'db_path') or self.db_path is None:
            raise ValueError("Путь к базе данных не указан (db_path is None)")
    
        output_db = self.db_path  # Сохраняем путь для вывода
        # 2. Кластеризация данных
        try:
            clustered_df = self.cluster_similar_names()
        except Exception as e:
            print(f"Ошибка при обработке данных: {e}")
            return False
        names_df =self.cluster_similar_names()
        if clustered_df.empty:
            print("Дубликаты не найдены")
            return True
       
        self.save_clusters_to_db(clustered_df)
        cluster_count= clustered_df['cluster_id'].nunique()
        print(f"Найдено {cluster_count} кластеров дубликатов")
        print(f"Результаты сохранены в {output_db}") 
if __name__ == "__main__":
    person = load_csv_safe('person.csv')  
    if person is not None:
        print(f"Загружено {len(person)} записей из person.csv")
        
        deduplicator = NameDeduplicator(person_df=person)
        
        print("\nНачало обработки реальных данных...")
        deduplicator.process_data()
    else:
        print("Ошибка: не удалось загрузить файл person.csv")
        print("Убедитесь, что файл существует и содержит данные в формате CSV")