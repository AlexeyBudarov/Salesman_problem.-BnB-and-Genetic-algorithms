import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class TravelingSalesmanProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.min_cost = float('inf')
        self.best_path = []
        self.explored_nodes = set()
        self.current_path = []
        self.all_distance = np.array([])

    def branch_and_bound(self, current_city, current_cost):
        if len(self.current_path) == self.num_cities:
            final_cost = current_cost + self.distance_matrix[current_city][self.current_path[0]]
            self.all_distance = np.append(self.all_distance, final_cost)
            if final_cost < self.min_cost:
                self.min_cost = final_cost
                self.best_path = self.current_path + [self.current_path[0]]
            return

        for next_city in range(self.num_cities):
            if next_city not in self.current_path:
                new_path = self.current_path + [next_city]
                new_cost = current_cost + self.distance_matrix[current_city][next_city]

                if new_cost < self.min_cost:
                    self.current_path = new_path
                    self.explored_nodes.add(tuple(self.current_path))
                    self.branch_and_bound(next_city, new_cost)
                    self.current_path.pop()

    def solve(self):
        start_city = 0
        self.current_path = [start_city]
        self.explored_nodes.add(tuple(self.current_path))
        self.branch_and_bound(start_city, 0)

    def solve_with_all_paths(self):
        start_city = 0
        self.current_path = [start_city]
        self.explored_nodes.add(tuple(self.current_path))

        all_paths = []  # Хранение всех маршрутов
        all_distances_per_iteration = []  # Хранение массивов длин маршрутов на каждой итерации

        def branch_and_bound_with_paths(current_city, current_cost):
            distances = []  # Хранение длин маршрутов для данной итерации

            if len(self.current_path) == self.num_cities:
                final_cost = current_cost + self.distance_matrix[current_city][self.current_path[0]]
                distances.append(final_cost)
                all_distances_per_iteration.append(distances)
                all_paths.append(self.current_path.copy())

                return

            for next_city in range(self.num_cities):
                if next_city not in self.current_path:
                    new_path = self.current_path + [next_city]
                    new_cost = current_cost + self.distance_matrix[current_city][next_city]
                    distances.append(new_cost)

                    if new_cost < self.min_cost:
                        self.current_path = new_path
                        self.explored_nodes.add(tuple(self.current_path))
                        branch_and_bound_with_paths(next_city, new_cost)
                        self.current_path.pop()

        branch_and_bound_with_paths(start_city, 0)

        # Постройте график длин маршрутов для каждого маршрута
        dist = np.array([])
        mean = np.array([])
        confidence_level = 95
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile

        up_per_array = np.array([])
        low_per_array = np.array([])

        for iteration, distances in enumerate(all_distances_per_iteration):
            dist = np.append(dist, distances)
            if iteration % 10 == 0:

                mean = np.append(mean, dist.mean())
                up_per_array = np.append(up_per_array, np.percentile(dist, upper_percentile))
                low_per_array = np.append(low_per_array, np.percentile(dist, lower_percentile))
                dist = np.array([])

        return low_per_array, mean, up_per_array


    def visualize_tree(self):
        G = nx.DiGraph()

        for node in self.explored_nodes:
            for i in range(len(node) - 1):
                G.add_edge(node[i], node[i + 1], weight=self.distance_matrix[node[i]][node[i + 1]])

        pos = nx.spring_layout(G, seed=42)
        edge_labels = {(i, j): self.distance_matrix[i][j] for i, j in G.edges()}
        nx.draw(G, pos, with_labels=True, node_size=700, font_size=10, node_color="lightblue", verticalalignment= 'center', arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, verticalalignment='center')
        nx.draw_networkx_nodes(G, pos, nodelist=self.current_path, node_color="red", node_size=700)





if __name__ == "__main__":
    cities4 = np.array([
        [0, 2, 15, 6],
        [2, 0, 7, 3],
        [15, 7, 0, 12],
        [6, 3, 12, 0]
    ])

    cities5 = np.array([
        [0, 2, 3, 5, 8],
        [2, 0, 4, 6, 2],
        [3, 4, 0, 7, 9],
        [5, 6, 7, 0, 3],
        [8, 2, 9, 3, 0]
    ])

    cities7 = np.array([
        [0, 2, 3, 5, 8, 4, 6],
        [2, 0, 4, 6, 2, 9, 1],
        [3, 4, 0, 7, 9, 5, 2],
        [5, 6, 7, 0, 3, 8, 5],
        [8, 2, 9, 3, 0, 1, 7],
        [4, 9, 5, 8, 1, 0, 4],
        [6, 1, 2, 5, 7, 4, 0],
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

    distance_matrix = cities9

    tsp = TravelingSalesmanProblem(distance_matrix)
    tsp.solve()

    low_per_array = mean = up_per_array = np.array([])

    for i in range(10):
        a, b, c = tsp.solve_with_all_paths()
        low_per_array = np.append(low_per_array, a)
        mean = np.append(mean, b)
        up_per_array = np.append(up_per_array, c)


    up_per_array.sort()
    low_per_array.sort()
    mean.sort()
    mean = mean[::-1]
    up_per_array = up_per_array[::-1]
    low_per_array = low_per_array[::-1]

    fig, ax = plt.subplots()
    ax.fill_between(np.arange(len(mean)), low_per_array, up_per_array, alpha=.7, linewidth=0)
    ax.plot(up_per_array, color = 'red', linewidth = 1)
    ax.plot(low_per_array, color = 'red', linewidth = 1)
    ax.plot(mean, color = 'black', linewidth = 2)

    plt.show()


    # Добавляем рёбра из корневой вершины ко всем остальным
    for city in range(1, tsp.num_cities):
        tsp.explored_nodes.add((0, city))

    tsp.visualize_tree()

    print("Optimal Path:", tsp.best_path)
    print("Optimal Cost:", tsp.min_cost)

    plt.show()