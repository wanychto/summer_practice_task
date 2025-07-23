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
        
        # Загрузка links по частям
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
                time.sleep(10) 

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
    sqlite_db_path='deduplicated_authors.db',
    clusters_table='name_variants',
    authorship_csv='links.csv',
    output_dir='safe_output'
)
processor.run_safe()
