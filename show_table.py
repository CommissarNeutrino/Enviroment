import pandas as pd
import os
import numpy as np
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

cache_dir = "cache/2b"
try_dir_base = "progon_"
existing_folders = [f for f in os.listdir(cache_dir) if
                    f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
if existing_folders:
    max_i = max([int(f.split('_')[1]) for f in existing_folders])
    progon_number = max_i
else:
    raise ValueError("Нет сохранённых прогонов для загрузки.")
progon_folder = os.path.join(f"cache/2c/progon_5")

# Открываем файл и читаем данные
print(progon_number)
data = pd.read_pickle(f"{progon_folder}/table_altruist_0.pkl")

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

# Инферируем объекты для правильного типа данных
df = df.infer_objects()

print(df)
