import random
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('ggplot')



# Функция для вычисления длины пути для данной перестановки городов
def calculate_path_length(cities, permutation):
    path_length = 0
    for i in range(len(permutation) - 1):
        city1 = permutation[i]
        city2 = permutation[i + 1]
        path_length += cities[int(city1)][int(city2)]
    if path_length == 0:
        a = 1
    return path_length

# Функция для создания начальной популяции
def initialize_population(num_cities, population_size, city0):
    population = []
    for _ in range(population_size):
        permutation = [0]*(num_cities)
        permutation[0] = city0
        for i in range(city0):
            permutation[i + 1] = i
        for i in range(city0 + 1, num_cities):
            permutation[i] = i
        random.shuffle(permutation)
        for i in range(num_cities):
            if permutation[i] == city0:
                permutation[i], permutation[0] = permutation[0], permutation[i]
        permutation.append(permutation[0])
        population.append(permutation)
    return population



# Функция для построения linkage tree
def build_linkage_tree(cities):
    dependency_matrix = calculate_dependency_matrix(cities)
    linkage_matrix = linkage(dependency_matrix, method='average')
    dendrogram(linkage_matrix)
    # clusters = cut_tree(linkage_matrix, height=2)
    threshold = determine_threshold(dependency_matrix)
    return linkage_matrix, threshold


def generate_crossover_mask(linkage_matrix, threshold):
    # Используем linkage matrix и пороговое значение, чтобы создать маску для кроссовера
    num_genes = len(linkage_matrix)*2 + 1
    mask_crossover = np.zeros((num_genes, num_genes), dtype=bool)

    for row in linkage_matrix:
        if row[2] < threshold:
            mask_crossover[int(row[0]), int(row[1])] = True
            mask_crossover[int(row[1]), int(row[0])] = True

    return mask_crossover

def generate_mutation_mask(linkage_tree, mutation_rate):

    """
    Генерация маски мутации на основе дерева связей и уровня мутации.

    Параметры:
    - linkage_tree: numpy.ndarray, дерево связей
    - mutation_rate: float, уровень мутации (вероятность мутации для каждого гена)

    Возвращает:
    - mutation_mask: numpy.ndarray, маска мутации
    """
    num_genes = len(linkage_tree) * 2 + 1  # Общее количество генов

    # Инициализация маски мутации
    mutation_mask = np.zeros(num_genes, dtype=int)

    # Проход по дереву связей
    for node in linkage_tree:
        # Устанавливаем 1 в маске мутации для генов, с уровнем мутации ниже заданного порога
        if np.random.rand() < mutation_rate:
            mutation_mask[int(node[0])] = 1
        if np.random.rand() < mutation_rate:
            mutation_mask[int(node[1])] = 1

    return mutation_mask



def apply_crossover_mask(ind1, ind2, mask_crossover):
    # Применяем маску кроссовера к двум индивидуумам
    # Проходим через каждый ген в индивидууме и меняем значения, если маска для кроссовера указывает на обмен
    ch1 = ind1
    ch2 = ind2
    for i in range(len(ind1)):
        if mask_crossover[int(ind1[i]), int(ind2[i])]:
            ch1[i], ch2[i] = ind2[i], ind1[i]
    return ch1, ch2


# Применение маски мутации к индивидууму
def apply_mutation_mask(individual, mask_mutation):
    mutated_indices = np.nonzero(mask_mutation[1:-1])[0] + 1  # +1 для коррекции индексов
    mutated_genes = np.zeros(len(individual) - 2)
    # Получаем значения генов для мутации
    for i in range(len(mutated_genes)):
        mutated_genes[i] = individual[mutated_indices[i]]
    # Применяем мутацию с перемешиванием
    np.random.shuffle(mutated_genes)
    # Возвращаем мутированный массив, вставляя краевые гены
    mutated_individual = np.concatenate(([individual[0]], mutated_genes, [individual[-1]]))
    return mutated_individual

def determine_threshold(dependency_matrix):
    threshold = np.mean(dependency_matrix)
    return threshold


def calculate_dependency_matrix(cities):
    cities = np.array(cities)
    N = len(cities)
    dist_matrix = np.zeros((len(cities), len(cities)))

    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = cities[i, j]
    condensed_dist_matrix = squareform(dist_matrix)
    return condensed_dist_matrix


# Функция для выбора новой популяции
def select_population(population, offspring, population_size, cities):
    combined_population = population + offspring
    combined_population.sort(key=lambda x: calculate_path_length(cities, x))
    new_population = combined_population[:population_size]
    return new_population

def evaluate_population(population, cities):
    fitness_values = []
    for individual in population:
        # Ваш код для оценки качества каждого индивида
        fitness = calculate_path_length(cities, individual)
        fitness_values.append(fitness)
    return fitness_values



def LTFGA(cities, population_size, city0):

    num_cities = len(cities)
    build_linkage_tree(cities)

    max_stagnation = 5
    current_stagnation = 0
    cnt_generations = 0

    test_min = np.array([])
    test_max = np.array([])
    test_mean = np.array([])

    confidence_level = 95
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile


    # Инициализация начальной популяции
    population = initialize_population(num_cities, population_size, city0)
    # print(population)
    # linkage_matrix = build_linkage_tree(cities)[0]
    # plt.figure()
    # dendrogram(linkage_matrix)
    # plt.show()
    best_fitness = evaluate_population(population, cities)

    while current_stagnation < max_stagnation:
    # Оценка приспособленности текущей популяции
        cnt_generations += 1
        fitness_scores = [1 / calculate_path_length(cities, permutation) for permutation in population]
        # test_min = np.append(test_min, min([calculate_path_length(cities, permutation) for permutation in population]))
        # test_max = np.append(test_max, max([calculate_path_length(cities, permutation) for permutation in population]))

        test_min = np.append(test_min, np.percentile([calculate_path_length(cities, permutation) for permutation in population], lower_percentile))
        test_max = np.append(test_max, np.percentile([calculate_path_length(cities, permutation) for permutation in population], upper_percentile))
        test_mean = np.append(test_mean, np.mean([calculate_path_length(cities, permutation) for permutation in population]))

    # Построение linkage tree
        linkage_tree = build_linkage_tree(cities)[0]

    # Создание новой популяции
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
            child1 = apply_crossover_mask(parent1, parent2, generate_crossover_mask(build_linkage_tree(cities)[0], build_linkage_tree(cities)[1]))[0]
            child2 = apply_crossover_mask(parent1, parent2, generate_crossover_mask(build_linkage_tree(cities)[0], build_linkage_tree(cities)[1]))[1]

            mutated_child1 = apply_mutation_mask(child1, generate_mutation_mask(build_linkage_tree(cities)[0], build_linkage_tree(cities)[1]))
            mutated_child2 = apply_mutation_mask(child2, generate_mutation_mask(build_linkage_tree(cities)[0], build_linkage_tree(cities)[1]))
            offspring.append(mutated_child1)
            offspring.append(mutated_child2)

        # Выбор новой популяции
        population = select_population(population, offspring, population_size, cities)
        # print(population)
        new_fitness = evaluate_population(population, cities)
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            current_stagnation = 0
        else:
            current_stagnation += 1

    # Нахождение лучшего решения
    best_solution = min(population, key=lambda x: calculate_path_length(cities, x))
    best_path_length = calculate_path_length(cities, best_solution)

    return best_solution, best_path_length, (test_min, test_mean, test_max), cnt_generations


# Параметры задачи
cities3 = np.array([
    [0, 10, 15],
    [10, 0, 20],
    [15, 20, 0]
])

cities4 = [
    [0, 2, 15, 6],
    [2, 0, 7, 3],
    [15, 7, 0, 12],
    [6, 3, 12, 0]
]

cities5 = [
    [0, 2, 3, 5, 8],
    [2, 0, 4, 6, 2],
    [3, 4, 0, 7, 9],
    [5, 6, 7, 0, 3],
    [8, 2, 9, 3, 0]
]

cities6 = np.array([
    [0, 10, 15, 20, 25, 30],
    [10, 0, 35, 40, 45, 50],
    [15, 35, 0, 55, 60, 65],
    [20, 40, 55, 0, 75, 80],
    [25, 45, 60, 75, 0, 90],
    [30, 50, 65, 80, 90, 0]
])

cities7 = [
    [0, 2, 3, 5, 8, 4, 6],
    [2, 0, 4, 6, 2, 9, 1],
    [3, 4, 0, 7, 9, 5, 2],
    [5, 6, 7, 0, 3, 8, 5],
    [8, 2, 9, 3, 0, 1, 7],
    [4, 9, 5, 8, 1, 0, 4],
    [6, 1, 2, 5, 7, 4, 0],
]

cities8 = np.array([
    [0, 10, 15, 20, 25, 30, 35, 40],
    [10, 0, 45, 50, 55, 60, 65, 70],
    [15, 45, 0, 75, 80, 85, 90, 95],
    [20, 50, 75, 0, 100, 105, 110, 115],
    [25, 55, 80, 100, 0, 120, 125, 130],
    [30, 60, 85, 105, 120, 0, 140, 145],
    [35, 65, 90, 110, 125, 140, 0, 150],
    [40, 70, 95, 115, 130, 145, 150, 0]
])

cities9 = np.array([
    [0, 10, 15, 20, 25, 30, 35, 40, 45],
    [10, 0, 50, 55, 60, 65, 70, 75, 80],
    [15, 50, 0, 85, 90, 95, 100, 105, 110],
    [20, 55, 85, 0, 115, 120, 125, 130, 135],
    [25, 60, 90, 115, 0, 140, 145, 150, 155],
    [30, 65, 95, 120, 140, 0, 160, 165, 170],
    [35, 70, 100, 125, 145, 160, 0, 180, 185],
    [40, 75, 105, 130, 150, 165, 180, 0, 190],
    [45, 80, 110, 135, 155, 170, 185, 190, 0]
])

all_cities = [cities3,cities4,cities5,cities6,cities7,cities8,cities9]

polulation_size = 10
city0 = 1

# best_solution, best_path_length, test, cnt_generations = LTFGA(cities5, polulation_size, city0)
#
# print("Лучшая перестановка городов:", best_solution)
# print("Длина пути:", best_path_length)
# print("Количество поколений:", cnt_generations)





def get_stats_values(cities, polulation_size,  city0):
    # best_solution, best_path_length, test, cnt_generations = LTFGA(cities, polulation_size, city0)
    # a, b, c = test
    # fix, ax = plt.subplots()
    # ax.fill_between(np.arange(len(a)), a, c, linewidth=0)
    # ax.plot(a, color='red', linewidth=1)
    # ax.plot(c, color='red', linewidth=1)
    # ax.plot(b, color='green', linewidth=2)
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Target function')
    # plt.show()

    test_min = test_max = test_mean = np.array([])
    for i in range(10):
        best_solution, best_path_length, test, cnt_generations = LTFGA(cities, polulation_size, city0)
        test_min_, test_mean_, test_max_ = test
        test_min = np.append(test_min, test_min_)
        test_mean = np.append(test_mean, test_mean_)
        test_max = np.append(test_max, test_max_)

    test_min.sort()
    test_mean.sort()
    test_max.sort()

    test_min = test_min[::-1]
    test_mean = test_mean[::-1]
    test_max = test_max[::-1]

    fix, ax = plt.subplots()
    ax.fill_between(np.arange(len(test_min)), test_min, test_max, alpha=.7, linewidth=0)
    ax.plot(test_max, color = 'red', linewidth = 1)
    ax.plot(test_min, color = 'red', linewidth = 1)
    ax.plot(test_mean, color = 'green', linewidth = 2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Target function')
    plt.show()


def get_plot():
    dimension = [3, 4, 5, 6, 7, 8, 9]
    cnt_generations = [LTFGA(city, 20, 1)[3] for city in all_cities]
    fig, ax = plt.subplots()
    ax.plot(dimension, cnt_generations)
    plt.show()



get_stats_values(cities9, polulation_size, city0)

# get_plot()