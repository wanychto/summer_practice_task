import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from collections import defaultdict

class SafePersonDisambiguator:
    def __init__(self, sqlite_db_path, clusters_table, authorship_csv, output_dir):
        self.sqlite_db_path = sqlite_db_path
        self.clusters_table = clusters_table
        self.authorship_csv = authorship_csv
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
        self.duplicates = []
    def load_data_in_chunks(self, chunk_size=10000):
        print("Загрузка данных частями...")
        
        # Загрузка кластеров
        with sqlite3.connect(self.sqlite_db_path) as conn:
            self.clusters = pd.read_sql(
                f"SELECT source_ids as person_id, cluster_id FROM {self.clusters_table}", 
                conn
            )
        self.clusters['person_id'] = self.clusters['person_id'].astype(str)
        
        # Загрузка authorship по частям
        chunks = pd.read_csv(self.authorship_csv, chunksize=chunk_size,
                            usecols=['person_id', 'publ_id'])
        
        self.person_to_pubs = defaultdict(set)
        self.pub_to_persons = defaultdict(set)
        
        for chunk in chunks:
            chunk['person_id'] = chunk['person_id'].astype(str)
            for _, row in chunk.iterrows():
                self.person_to_pubs[row['person_id']].add(row['publ_id'])
                self.pub_to_persons[row['publ_id']].add(row['person_id'])
            
            # Контроль памяти
            if len(self.person_to_pubs) > 500000:  # Лимит уникальных персон
                self.save_checkpoint()
                print(f"Загружено {len(self.person_to_pubs)} персон. Пауза...")
                time.sleep(10)  # Даем остыть процессору

    def save_checkpoint(self):
        checkpoint_data = {
            'duplicates': self.duplicates,
            'last_processed': datetime.now().isoformat()
        }
        pd.DataFrame(checkpoint_data).to_json(self.checkpoint_file)
        print(f"Чекпоинт сохранен в {self.checkpoint_file}")

    def find_duplicates_safely(self, batch_size=1000, cooldown=5):
        all_persons = list(self.person_to_pubs.keys())
        total = len(all_persons)
        start_time = time.time()
        
        print(f"Начало обработки {total} персон...")
        
        for i in range(0, total, batch_size):
            batch = all_persons[i:i + batch_size]
            batch_start = time.time()
            
            for j, p1 in enumerate(batch):
                # Пропускаем уже обработанные
                if any(p1 in pair for pair in self.duplicates):
                    continue
                    
               
                for p2 in all_persons[i+j+1:]:
                    # Добавляем проверку кластера
                    p1_cluster = self.clusters[self.clusters['person_id'] == p1]['cluster_id'].values
                    p2_cluster = self.clusters[self.clusters['person_id'] == p2]['cluster_id'].values
                    
                    if len(p1_cluster) > 0 and len(p2_cluster) > 0 and p1_cluster[0] == p2_cluster[0]:
                        common = self.person_to_pubs[p1] & self.person_to_pubs[p2]
                        if common:
                            self.duplicates.append((p1, p2, len(common)))
                        
                # Контроль температуры
                if time.time() - batch_start > 30:  # 30 сек на батч
                    print(f"Обработано {i+j}/{total}. Пауза...")
                    self.save_checkpoint()
                    time.sleep(cooldown)
                    batch_start = time.time()
            
            # Сохраняем каждые N батчей
            if i % (10 * batch_size) == 0:
                self.save_checkpoint()
        
        print(f"Обработка завершена за {(time.time()-start_time)/60:.1f} минут")

    # Создание тестовой базы данных (добавьте этот код перед основным выполнением)
    def create_test_database():
        # Удаляем старый тестовый файл, если существует
        if os.path.exists('test.db'):
            os.remove('test.db')
        
        # Подключаемся к базе (файл создастся автоматически)
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        
        # Создаем таблицу name_variants
        cursor.execute('''
        CREATE TABLE name_variants (
            source_ids TEXT,
            cluster_id TEXT
        )''')
        
        # Вставляем тестовые данные
        test_data = [
            ('1', 'cluster1'),
            ('2', 'cluster1'),  # дубликат
            ('3', 'cluster2'),
            ('4', 'cluster3'),
            ('5', 'cluster3')   # дубликат
        ]
        cursor.executemany('INSERT INTO name_variants VALUES (?, ?)', test_data)
        
        # Сохраняем изменения и закрываем соединение
        conn.commit()
        conn.close()
        
        # Создаем тестовый CSV файл
        with open('test_links.csv', 'w') as f:
            f.write('''person_id,publ_id
    1,pub1
    1,pub2
    2,pub1
    2,pub3
    3,pub4
    4,pub5
    5,pub5
    5,pub6
    ''')

    # Вызываем функцию создания тестовых данных
    create_test_database()
    def run_safe(self):
        try:
            self.load_data_in_chunks()
            self.find_duplicates_safely()
            
            # Сохранение результатов
            result = pd.DataFrame(self.duplicates, 
                                columns=['person1', 'person2', 'common_pubs'])
            result.to_csv(os.path.join(self.output_dir, 'publ_duplicates.csv'), index=False)
            
            print(f"Найдено {len(self.duplicates)} дубликатов")
            print(f"Результаты сохранены в {self.output_dir}")
            
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            self.save_checkpoint()
            print("Прогресс сохранен в чекпоинт. Можно продолжить позже.")
processor = SafePersonDisambiguator(
    sqlite_db_path='test.db',          # Используем тестовую базу
    clusters_table='name_variants',    # Таблица в тестовой базе
    authorship_csv='test_links.csv',   # Тестовый CSV файл
    output_dir='test_output'           # Отдельная директория для тестовых результатов
)
processor.run_safe()