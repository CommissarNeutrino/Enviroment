import pandas as pd
import os
import numpy as np
import sys

def visualize_grid(df, cell_size=5):
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

map_type = "1a"
agent_type = "patron"
progon_number = None
if "no_learn" in sys.argv:
    learning_needed = False
if "no_test" in sys.argv:
    testing_needed = False
for arg in sys.argv:
    if arg.startswith("map_type"):
        map_type=arg.split("=")[1]
for arg in sys.argv:
    if arg.startswith("progon_num"):
        progon_number=arg.split("=")[1]
for arg in sys.argv:
    if arg.startswith("agent_type"):
        agent_type=arg.split("=")[1]


action_to_direction = {
    0: np.array([0, -1]),
    1: np.array([1, 0]),
    2: np.array([0, 1]),
    3: np.array([-1, 0]),
    4: np.array([0, 0]),
}
# Словарь для отображения действий в направления
action_to_direction = {
    0: 'up',  # Движение влево
    1: 'right',  # Движение вниз
    2: 'down',  # Движение вправо
    3: 'left',    # Движение вверх
    4: 'stay'   # Оставаться на месте
}

cache_dir = f"cache/{map_type}"
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
print(whole_path)
data = pd.read_pickle(whole_path)

# Создадим набор действий (используем числовые действия для сопоставления с направлениями)
actions = set(action for (state, action) in data.keys())
states = set(state for (state, action) in data.keys())

# Заменим числовые действия на их строковые эквиваленты (направления)
direction_columns = [action_to_direction[action] for action in sorted(actions)]

# Создаем DataFrame с индексами для состояний и столбцами для направлений
df = pd.DataFrame(index=sorted(states), columns=direction_columns)

# Заполним DataFrame значениями Q-функции
for (state, action), q_value in data.items():
    direction = action_to_direction[action]  # Получаем направление по действию
    df.at[state, direction] = q_value

df['max_q'] = df.idxmax(axis=1)


# Инферируем объекты для правильного типа данных
df = df.infer_objects()

print(df)

visualize_grid(df)
