import sqlite3
import csv
import os
from contextlib import closing

def export_matched_records(db_path: str, table_name: str, id_column: str, extra_column: str,
                          csv_path: str, csv_id_column: str, output_path: str):
   
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Файл базы данных не найден: {db_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.cursor()) as cursor:
                query = f"SELECT CAST({id_column} AS TEXT) as id, {extra_column} FROM {table_name}"
                cursor.execute(query)
                db_data = {row['id']: row[extra_column] for row in cursor.fetchall()}
                
                if not db_data:
                    raise ValueError("В базе данных не найдено ни одной записи")
                print(f"Загружено {len(db_data)} записей из базы данных")
                

    except sqlite3.Error as e:
        raise ValueError(f"Ошибка при чтении базы данных: {e}")


    matched_rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            if not reader.fieldnames or csv_id_column not in reader.fieldnames:
                available = ", ".join(reader.fieldnames) if reader.fieldnames else "нет колонок"
                raise ValueError(f"Колонка '{csv_id_column}' не найдена в CSV. Доступные: {available}")
            
            for row in reader:
                csv_id = str(row[csv_id_column]).strip()
                if csv_id in db_data:
                    row[extra_column] = db_data[csv_id]
                    matched_rows.append(row)
    except Exception as e:
        raise ValueError(f"Ошибка при обработке CSV: {e}")
    if not matched_rows:
        print("Предупреждение: Не найдено ни одного совпадения!")
        return
    try:
        matched_rows.sort(key=lambda x: int(x.get(extra_column, 0)))  # 0 как значение по умолчанию
    except ValueError:
    # Если не получается преобразовать в число, сортируем как строки
        matched_rows.sort(key=lambda x: str(x.get(extra_column, "")))
    except Exception as e:
        raise ValueError(f"Ошибка при сортировке: {e}")
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as out_file:
            fieldnames = reader.fieldnames + [extra_column]
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matched_rows)
        
        print(f"Найдено {len(matched_rows)} совпадений. Результат сохранен в {output_path}")
    except Exception as e:
        raise ValueError(f"Ошибка при сохранении: {e}")

if __name__ == "__main__":
    config = {
        'db_path': 'C:\\Users\\smirn\\Documents\\summer_practice_task\\deduplicated_authors.db',
        'table_name': 'name_variants',
        'id_column': 'source_ids',  
        'extra_column':'cluster_id',
        'csv_path': 'person.csv',
        'csv_id_column': 'a',  
        'output_path': 'duplicate.csv'
    }

    try:
        export_matched_records(
            db_path=config['db_path'],
            table_name=config['table_name'],
            id_column=config['id_column'],
            extra_column=config['extra_column'],
            csv_path=config['csv_path'],
            csv_id_column=config['csv_id_column'],
            output_path=config['output_path']
        )
    except Exception as e:
        print(f"Ошибка: {e}")