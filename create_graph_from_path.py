import pickle
import matplotlib.pyplot as plt

def load_from_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None

def plot_graph(data):
    # Определяем тип данных для определения типа графика
    if isinstance(data[0], (int, float)):
        plt.plot(range(len(data)), data)
    else:
        print("Неизвестный тип данных для построения графика.")
        return
    plt.tight_layout()
    plt.show()

def main(path: str):
    loaded_data = load_from_pickle(path)
    if loaded_data is not None:
        print("Данные успешно загружены из файла.")
        plot_graph(loaded_data)
    else:
        print("Не удалось загрузить данные из файла.")

if __name__ == "__main__":
    path = "cache/3a/progon_2/0.001_0.05/table_data.pkl"
    main(path)
