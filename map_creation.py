from typing import Optional

class Map_Creation():

    def select_scenary(self, map_type, path_to_patron_q_table: Optional[str] = None, path_to_altruist_q_table: Optional[str] = None):
        # scenary_type = map_type[0]
        # pod_scenary = map_type[1]
        agent_patron_start_zone = []
        agent_altruist_start_zone = []
        walls_positions = set()
        doors_positions = {}
        target_location = (4, 0)
        match map_type:
            case "1":
                size_x = 5
                size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                # match pod_scenary:
                #     case "1":
                #         agent_patron_status = "trainig"
                #         agent_altruist_status = "not_there"
                #     case "2":
                #         agent_patron_status = "trainig"
                #         agent_altruist_status = "random"
            case 2:
                size_x = 5
                size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions = set([(1, 0), (1, 1), (4, 1)])
                # match pod_scenary:
                #     case "1":
                #         agent_patron_status = "training"
                #         agent_altruist_status = "not_there"       
                #     case "2":
                #         agent_patron_status = "training"
                #         agent_altruist_status = "random"
                #     case "3":
                #         agent_patron_status = "trained"
                #         agent_altruist_status = "training"
            case 3:
                size_x = 5
                size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions = set([(1, 0), (1, 1), (4, 1)])
                doors_positions = {(1, 2): (3, 1)}
                # match pod_scenary:
                #     case "1":
                #         agent_patron_status = "training"
                #         agent_altruist_status = "random"
                #     case "2":
                #         agent_patron_status = "trained"
                #         agent_altruist_status = "training"
            case 4:
                size_x = 7
                size_y = 3
                agent_patron_start_zone.extend([(0, 0), (0, 1), (0, 2)])
                agent_altruist_start_zone.extend([(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)])
                walls_positions = set([(1, 0), (1, 1), (4, 1), (5, 0)])
                doors_positions = {(1, 2): (3, 1), (4, 2): (3, 0)}
                # match pod_scenary:
                #     case "1":
                #         agent_patron_status = "training"
                #         agent_altruist_status = "not_there"
                #     case "2":
                #         agent_patron_status = "training"
                #         agent_altruist_status = "random"              
                #     case "3":
                #         agent_patron_status = "trained"
                #         agent_altruist_status = "training"
        return size_x, size_y, agent_patron_start_zone, agent_altruist_start_zone, target_location, walls_positions, doors_positions