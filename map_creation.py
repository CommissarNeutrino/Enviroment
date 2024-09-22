class Map_Creation():

    def select_scenary(self, map_type):
        agent_patron_start_zone = []
        agent_altruist_start_zone = []
        walls_positions = set()
        target_location = (4, 0)
        match map_type:
            case 1:
                self.size_x = 5
                self.size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                doors_positions = {}
            case 2:
                self.size_x = 5
                self.size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions.update[(1, 0), (1, 1), (4, 1)]
                doors_positions = {}
            case 3:
                self.size_x = 5
                self.size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions.update[(1, 0), (1, 1), (4, 1)]
                doors_positions = {(1, 2): (3, 1)}
            case 4:
                self.size_x = 7
                self.size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions.update[(1, 0), (1, 1), (4, 1), (5, 0)]
                doors_positions = {(1, 2): (3, 1), (4, 2): (3, 0)}
        return agent_patron_start_zone, agent_altruist_start_zone, target_location, walls_positions, doors_positions