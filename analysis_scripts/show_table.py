import pandas as pd
from typing import Dict, Tuple
import os
import sys

def get_data(scenario_num=None, agent_type=None, progon_number=None) -> Dict[Tuple, float]:

    help_message = """
Этот скрипт загружает сохранённые Q-таблицы, преобразует их в таблицу действий и визуализирует оптимальные направления движения агента.

Использование:
    python show_table.py scenario_num=<номер_сценария> [agent_type=<тип_агента>] [progon_num=<номер_прогона>]

Параметры:
    scenario_num  — номер сценария для загрузки (например: 1a, 2b). Обязательно.
    agent_type    — тип агента (по умолчанию 'patron').
    progon_num    — номер прогона для загрузки (по умолчанию берётся последний доступный).
    --help        — вывести это сообщение и завершить выполнение.

Примеры запуска:
    python show_table.py scenario_num=1a
    python show_table.py scenario_num=2b agent_type=altruist progon_num=5

Описание работы:
    1. Скрипт ищет сохранённые данные в папке cache/<сценарий>/progon_<номер>/table_<agent_type>.
    2. Загружает Q-таблицу и преобразует её в читаемый DataFrame.
    3. Определяет лучшее действие для каждого состояния.
    4. Визуализирует карту с указанием направлений оптимальных действий.
"""

    # Обработка помощи
    if "--help" in sys.argv or not any(arg.startswith("scenario_num") for arg in sys.argv):
        print(help_message)
        sys.exit()
    for arg in sys.argv:
        if arg.startswith("scenario_num"):
            scenario_num=arg.split("=")[1]
    for arg in sys.argv:
        if arg.startswith("progon_num"):
            progon_number=arg.split("=")[1]
    for arg in sys.argv:
        if arg.startswith("agent_type"):
            agent_type=arg.split("=")[1]
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", scenario_num)
    try_dir_base = "progon_"
    existing_folders = [f for f in os.listdir(cache_dir) if
                        f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
    if not progon_number:
        if existing_folders:
            max_i = max([int(f.split('_')[1]) for f in existing_folders])
            progon_number = max_i
        else:
            raise ValueError("Нет сохранённых прогонов для загрузки.")
    progon_folder = os.path.join(cache_dir, f"progon_{progon_number}")
    whole_path = os.path.join(progon_folder, f"table_{agent_type}_0.pkl")
    # Открываем файл и читаем данные
    print("Итоговый путь загрузки: ", whole_path)
    data = pd.read_pickle(whole_path)
    return data

def create_df(pickled_data: Dict[Tuple, float]) -> pd.DataFrame:
    # Словарь для отображения действий в направления
    action_to_direction = {
        0: 'up',  # Движение влево
        1: 'right',  # Движение вниз
        2: 'down',  # Движение вправо
        3: 'left',    # Движение вверх
        4: 'stay'   # Оставаться на месте
    }
    # Создадим набор действий (используем числовые действия для сопоставления с направлениями)
    states = set()
    actions = set()

    for state, action in pickled_data.keys():
        states.add(state)
        actions.add(action)

    # Заменим числовые действия на их строковые эквиваленты (направления)
    direction_columns = [action_to_direction[action] for action in sorted(actions)]

    # Создаем DataFrame с индексами для состояний и столбцами для направлений
    df = pd.DataFrame(index=sorted(states), columns=direction_columns)

    # Заполним DataFrame значениями Q-функции
    for (state, action), q_value in pickled_data.items():
        direction = action_to_direction[action]  # Получаем направление по действию
        df.at[state, direction] = q_value

    df['max_q'] = df.idxmax(axis=1)

    # Инферируем объекты для правильного типа данных
    df = df.infer_objects()

    print(df)

    return df

def visualize_grid(df: pd.DataFrame, cell_size=5) -> None:
    # Получаем уникальные координаты
    states = df.index
    grid_size_x = max(state[0] for state in states) + 1
    grid_size_y = max(state[1] for state in states) + 1
    
    # Символы для направлений
    direction_symbols = {
        'up': '▲',
        'down': '▼',
        'left': '◀--',
        'right': '--▶'
    }

    # Печатаем заголовок с номерами столбцов
    header = "   |" + "".join([f"{i}".center(cell_size) + "|" for i in range(grid_size_x)])
    print(header)
    print("  " + "--".join(["-" * cell_size for _ in range(grid_size_x)]))

    # Печатаем каждую строку сетки
    for i in range(grid_size_y):
        row_first = ""
        row_second = ""
        row_third = ""
        for j in range(grid_size_x):
            state = (j, i)
            if state in df.index:
                direction = df.at[state, 'max_q']
            else:
                direction = None
            
            symbol = direction_symbols.get(direction, '...')
            # Форматируем строки вывода для каждого направления
            if symbol == '▲':   # Up
                row_first += f"▲".center(cell_size) + "|"
                row_second += f"¦".center(cell_size) + "|"
                row_third += f"¦".center(cell_size) + "|"
            elif symbol == '▼':  # Down
                row_first += f"¦".center(cell_size) + "|"
                row_second += f"¦".center(cell_size) + "|"
                row_third += f"▼".center(cell_size) + "|"
            elif symbol == '◀--':  # Left
                row_first += "   ".center(cell_size) + "|"
                row_second += "◀--".center(cell_size) + "|"
                row_third += "   ".center(cell_size) + "|"
            elif symbol == '--▶':  # Right
                row_first += "   ".center(cell_size) + "|"
                row_second += "--▶".center(cell_size) + "|"
                row_third += "   ".center(cell_size) + "|"
            else:  # Empty or unknown direction
                row_first += "".join("·" * cell_size).center(cell_size) + "|"
                row_second += "".join("·" * cell_size).center(cell_size) + "|"
                row_third += "".join("·" * cell_size).center(cell_size) + "|"

        # Печатаем строки для текущего ряда
        print(f"   |{row_first}")
        print(f" {i} |{row_second}")
        print(f"   |{row_third}")
        print("  " + "--".join(["-" * cell_size for _ in range(grid_size_x)]))

pickled_data = get_data()
df = create_df(pickled_data)
visualize_grid(df)
