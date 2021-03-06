from re import S
import unittest
from typing import List, Tuple, Union, Dict

from environment_agent_modules import (
    SwarmAgent,
    Direction,
    Turn,
    ObjectType,
    RelativePosition,
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
)


class swarm_agent_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = create_nonclustered_tile_grid(height=5, width=5)
        self.clusted_enviro_obs_useful = (
            create_clustered_inital_observation_useful_tile_grid(
                width=5, height=5, ratio_of_white_to_black_tiles=0.52
            )
        )
        self.clusted_enviro_obs_not_useful = (
            create_clustered_inital_observation_not_useful_tile_grid(
                width=5, height=5, ratio_of_white_to_black_tiles=0.52
            )
        )
        self.swarm_agent = SwarmAgent(starting_cell=self.tiled_enviro[(0, 0)])

    def test_visited_cell_is_visited(self):
        self.assertEqual(
            self.swarm_agent.current_cell, (0, 0), "current cell not correct"
        )

    def test_visit_cell_again_cause_no_error(self):
        with self.subTest():
            self.swarm_agent.occupy_cell(self.tiled_enviro[(0, 0)])

    def test_occupied_cell_returns_is_occupied(self):
        self.assertEqual(
            self.tiled_enviro[(0, 0)]["agent"],
            self.swarm_agent,
            "occupation status not correct",
        )

    def test_another_member_visting_cell_cannot_enter_it(self):
        second_swarm_agent = SwarmAgent(starting_cell=self.tiled_enviro[(0, 0)])
        self.assertNotEqual(
            second_swarm_agent.current_cell,
            (0, 0),
            "visiting cell that is occupied not working",
        )

    def test_leaving_cell_makes_it_no_longer_occupied(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.assertEqual(
            self.tiled_enviro[(0, 0)]["agent"],
            None,
            "occupation status not correct on leaving",
        )

    def test_occupy_leave_cell_allows_other_agent_to_occupy(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        second_swarm_agent = SwarmAgent(starting_cell=self.tiled_enviro[(0, 0)])
        with self.subTest():
            self.assertEqual(
                self.tiled_enviro[(0, 0)]["agent"],
                second_swarm_agent,
                "occupation status not correct on re-entry",
            )
            self.assertEqual(
                second_swarm_agent.current_cell,
                (0, 0),
                "cell not added to visited list",
            )

    def test_when_leaving_cell_it_is_added_to_visited_list(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.assertIn(
            (0, 0),
            self.swarm_agent.cells_visited,
            "cell when left not added to visited list",
        )

    def test_current_cell_not_in_visited_cell_return_reward_of_1(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro[(1, 0)])
        self.assertEqual(
            self.swarm_agent.return_navigation_reward(),
            2,
            "postive reward not returning correctly",
        )

    def test_current_cell_in_visited_cell_return_reward_of_neg_1(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        self.swarm_agent.occupy_cell(self.tiled_enviro[(0, 0)])
        self.assertEqual(
            self.swarm_agent.return_navigation_reward(),
            -1,
            "zero reward not returning correctly",
        )

    def start_on_different_cell(
        self, different_starting_cell: Tuple[int, int]
    ) -> SwarmAgent:
        self.swarm_agent.leave_cell(self.tiled_enviro[self.swarm_agent.current_cell])
        return SwarmAgent(starting_cell=self.tiled_enviro[different_starting_cell])

    def succesful_forward_step_tester(
        self,
        new_cell: Tuple[int, int],
        agent_direction: Direction = Direction.RIGHT,
        different_starting_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        starting_cell = self.swarm_agent.current_cell

        self.swarm_agent.current_direction_facing = agent_direction.value

        self.swarm_agent.forward_step(self.tiled_enviro)

        self.swarm_agent.add_cell_to_visited_list(starting_cell)

        with self.subTest():
            self.assertEqual(
                self.swarm_agent.current_cell,
                new_cell,
                f"current cell for {agent_direction} facing forward step not working",
            )
            self.assertIn(
                starting_cell,
                self.swarm_agent.cells_visited,
                f"cells visited for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro[starting_cell]["agent"],
                None,
                f"occupation for cell left for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro[new_cell]["agent"],
                self.swarm_agent,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_navigation_reward(),
                2,
                "correct reward not returning after succesful step forward",
            )

    def test_step_forward_agent_facing_right(self):
        self.succesful_forward_step_tester(new_cell=(0, 1))

    def test_step_forward_agent_facing_down(self):
        self.succesful_forward_step_tester(
            new_cell=(1, 0), agent_direction=Direction.DOWN
        )

    def test_step_forward_from_tile_0_1_agent_facing_left(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0),
            agent_direction=Direction.LEFT,
            different_starting_cell=(0, 1),
        )

    def test_step_forward_from_tile_1_0_agent_facing_up(self):
        self.succesful_forward_step_tester(
            new_cell=(0, 0),
            agent_direction=Direction.UP,
            different_starting_cell=(1, 0),
        )

    def unsuccesful_forward_step_tester(
        self,
        different_starting_cell: Tuple[int, int] = None,
        agent_direction: Direction = Direction.RIGHT,
        pre_occupied_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        starting_cell = self.swarm_agent.current_cell

        self.swarm_agent.current_direction_facing = agent_direction.value

        self.swarm_agent.forward_step(self.tiled_enviro)

        self.swarm_agent.add_cell_to_visited_list(starting_cell)

        with self.subTest():
            self.assertEqual(
                self.swarm_agent.current_cell,
                starting_cell,
                f"current cell for {agent_direction} facing forward step not working",
            )
            self.assertIn(
                starting_cell,
                self.swarm_agent.cells_visited,
                f"cells visited for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.tiled_enviro[starting_cell]["agent"],
                self.swarm_agent,
                f"occupation for current cell for {agent_direction} facing forward step not working",
            )
            self.assertEqual(
                self.swarm_agent.return_navigation_reward(),
                -1,
                "correct reward not returning after unsuccesful step forward",
            )
            if pre_occupied_cell:
                self.assertNotIn(
                    pre_occupied_cell,
                    self.swarm_agent.cells_visited,
                    "previously occupied cell in visited cells list",
                )

    def test_step_forward_on_to_occupied_tile(self):
        SwarmAgent(starting_cell=self.tiled_enviro[(0, 1)])
        self.unsuccesful_forward_step_tester(pre_occupied_cell=(0, 1))

    def wall_and_corner_tester(
        self,
        directions: List[Direction],
        different_starting_cell: Tuple[int, int] = (0, 0),
    ) -> None:
        for direction in directions:
            self.unsuccesful_forward_step_tester(
                different_starting_cell=different_starting_cell,
                agent_direction=direction,
            )

    def test_step_forward_into_top_left_corner(self):
        self.wall_and_corner_tester(directions=[Direction.LEFT, Direction.UP])

    def test_step_forward_into_top_right_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(0, 4), directions=[Direction.RIGHT, Direction.UP]
        )

    def test_step_forward_into_bottom_left_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 0), directions=[Direction.LEFT, Direction.DOWN]
        )

    def test_step_forward_into_bottom_right_corner(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 4), directions=[Direction.RIGHT, Direction.DOWN]
        )

    def test_step_forward_into_top_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(0, 1), directions=[Direction.UP]
        )

    def test_step_forward_into_left_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(1, 0), directions=[Direction.LEFT]
        )

    def test_step_forward_into_right_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(1, 4), directions=[Direction.RIGHT]
        )

    def test_step_forward_into_bottom_wall(self):
        self.wall_and_corner_tester(
            different_starting_cell=(4, 1), directions=[Direction.DOWN]
        )

    def rotation_tester(
        self,
        rotations: List[Turn],
        correction_directions: List[Direction],
        starting_direction: Direction = Direction.UP,
    ) -> None:
        self.swarm_agent.current_direction_facing = starting_direction.value
        for rotation, correction_direction in zip(rotations, correction_directions):
            self.swarm_agent.turn(turn_type=rotation.value)
            self.assertEqual(
                self.swarm_agent.current_direction_facing,
                correction_direction.value,
                f"Turning to the {rotation} to ended up in {correction_direction} didn't work",
            )

    def test_from_up_positon_turn_right_turn_left(self):
        self.rotation_tester(
            rotations=[Turn.RIGHT, Turn.LEFT],
            correction_directions=[Direction.RIGHT, Direction.UP],
        )

    def test_from_down_positon_turn_left_turn_right(self):
        self.rotation_tester(
            starting_direction=Direction.DOWN,
            rotations=[Turn.LEFT, Turn.RIGHT],
            correction_directions=[Direction.RIGHT, Direction.DOWN],
        )

    def test_from_up_positon_full_clockwise_turn(self):
        self.rotation_tester(
            rotations=[Turn.RIGHT] * 4,
            correction_directions=[
                Direction.RIGHT,
                Direction.DOWN,
                Direction.LEFT,
                Direction.UP,
            ],
        )

    def test_from_up_positon_full_anticlockwise_turn(self):
        self.rotation_tester(
            rotations=[Turn.LEFT] * 4,
            correction_directions=[
                Direction.LEFT,
                Direction.DOWN,
                Direction.RIGHT,
                Direction.UP,
            ],
        )

    def state_tester(
        self,
        correct_navigation_states: List[Union[ObjectType, RelativePosition]],
        obj_description: str,
        agent_direction: Direction = Direction.RIGHT,
        different_starting_cell: Tuple[int, int] = None,
    ):
        if different_starting_cell:
            self.swarm_agent = self.start_on_different_cell(
                different_starting_cell=different_starting_cell
            )

        self.swarm_agent.current_direction_facing = agent_direction.value

        current_navigation_states = self.swarm_agent.get_navigation_states(
            self.tiled_enviro
        )

        for current_state, correct_state in zip(
            current_navigation_states, correct_navigation_states
        ):
            self.assertEqual(
                current_state,
                correct_state.value,
                (
                    f"{correct_state} not returned when facing {agent_direction} at {self.swarm_agent.current_cell} observing {obj_description}"
                ),
            )

    def return_sense_tester_agent(
        self, starting_cell: Dict[str, any], sense: bool
    ) -> SwarmAgent:
        return SwarmAgent(starting_cell=starting_cell, sensing=int(sense))

    def test_agent_when_sensing_updates_num_of_white_tiles_when_on_white_tile(self):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_useful[(0, 0)], sense=True
        )

        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 1, "num_of_white_tiles should be 1"
        )

    def test_agent_when_sensing_updates_num_of_white_tiles_when_on_white_tile_twice(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_useful[(0, 0)], sense=True
        )
        swarm_agent.turn(1)
        swarm_agent.forward_step(tile_grid=self.clusted_enviro_obs_useful)
        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 2, "num_of_white_tiles should be 2"
        )

    def test_agent_when_not_sensing_doesnt_update_num_of_white_tiles_when_on_white_tile(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_useful[(0, 0)], sense=False
        )

        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 0, "num_of_white_tiles should be 0"
        )

    def test_agent_when_not_sensing_doesnt_update_num_of_white_tiles_when_on_white_tile_twice(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_useful[(0, 0)], sense=False
        )
        swarm_agent.turn(1)
        swarm_agent.forward_step(tile_grid=self.clusted_enviro_obs_useful)
        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 0, "num_of_white_tiles should be 0"
        )

    def test_agent_when_sensing_doesnt_update_num_of_white_tiles_when_on_black_tile(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_not_useful[(0, 0)], sense=True
        )

        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 0, "num_of_white_tiles should be 0"
        )

    def test_agent_when_sensing_updates_num_of_white_tiles_when_on_black_then_white_tile(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_not_useful[(0, 0)], sense=True
        )
        swarm_agent.turn(1)
        swarm_agent.forward_step(tile_grid=self.clusted_enviro_obs_useful)
        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 1, "num_of_white_tiles should be 2"
        )

    def test_agent_when_not_sensing_doesnt_update_num_of_white_tiles_when_on_black_tile(
        self,
    ):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_not_useful[(0, 0)], sense=False
        )

        self.assertEqual(
            swarm_agent.num_of_white_cells_observed, 0, "num_of_white_tiles should be 0"
        )

    def test_return_opinion_sensing_true_returns_none(self):
        self.assertEqual(
            self.swarm_agent.return_opinion(), None, "return_opinion should be None"
        )

    def test_return_opinion_sensing_false_returns_int(self):
        self.swarm_agent.sensing = 0
        self.swarm_agent.perform_navigation_action(0, self.tiled_enviro)
        self.assertIsInstance(
            self.swarm_agent.return_opinion(), int, "return_opinion should be int"
        )

    def test_return_opinion_returns_1_majorty_white(self):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_useful[(0, 0)], sense=True
        )
        swarm_agent.perform_navigation_action(2, self.clusted_enviro_obs_useful)
        for i in range(3):
            swarm_agent.perform_navigation_action(0, self.clusted_enviro_obs_useful)
        swarm_agent.sensing = 0
        self.assertEqual(
            swarm_agent.return_opinion(), 1, "return_opinion should be 1 for white"
        )

    def test_return_opinion_returns_0_majorty_black(self):
        swarm_agent = self.return_sense_tester_agent(
            starting_cell=self.clusted_enviro_obs_not_useful[(0, 0)], sense=True
        )
        swarm_agent.perform_navigation_action(2, self.clusted_enviro_obs_not_useful)
        for i in range(3):
            swarm_agent.perform_navigation_action(0, self.clusted_enviro_obs_not_useful)
        swarm_agent.sensing = 0
        self.assertEqual(
            swarm_agent.return_opinion(), 0, "return_opinion should be 0 for black"
        )

    def test_return_opinion_returns_1_when_55_white(self):
        self.swarm_agent.num_of_white_cells_observed = 11
        self.swarm_agent.num_of_cells_observed = 20
        self.swarm_agent.sensing = 0
        self.assertEqual(
            self.swarm_agent.return_opinion(), 1, "return_opinion should be 1 for white"
        )

    def test_return_opinion_returns_0_when_45_white(self):
        self.swarm_agent.num_of_white_cells_observed = 9
        self.swarm_agent.num_of_cells_observed = 20
        self.swarm_agent.sensing = 0
        self.assertEqual(
            self.swarm_agent.return_opinion(), 0, "return_opinion should be 0 for black"
        )

    def test_recieve_local_opinions_all_filled(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        swarm_agent = SwarmAgent(starting_cell=self.tiled_enviro[(1, 1)])

        for i in range(3):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[0, i], sensing=0)
            agent.num_of_white_cells_observed += 1
            agent.num_of_cells_observed += 1
        for i in range(3):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[2, i], sensing=0)
            agent.num_of_cells_observed += 1
        for i in range(0, 3, 2):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[1, i], sensing=0)
            agent.num_of_white_cells_observed += int(bool(i))
            agent.num_of_cells_observed += 1

        swarm_agent.recieve_local_opinions(tile_grid=self.tiled_enviro)

        self.assertAlmostEqual(
            swarm_agent.calculated_collective_opinion,
            0.448157,
            msg="internal_collective_opinion should be 0.495",
            places=9,
        )

    def test_recieve_local_opinions_partially_filled_and_sensing_agnets(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])
        swarm_agent = SwarmAgent(starting_cell=self.tiled_enviro[(1, 1)])

        for i in range(3):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[0, i], sensing=1)
            agent.num_of_white_cells_observed += 1
            agent.num_of_cells_observed += 1
        for i in range(1):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[2, i], sensing=0)
            agent.num_of_cells_observed += 1
        for i in range(0, 3, 2):
            agent = SwarmAgent(starting_cell=self.tiled_enviro[1, i], sensing=0)
            agent.num_of_white_cells_observed += int(bool(i))
            agent.num_of_cells_observed += 1

        swarm_agent.recieve_local_opinions(tile_grid=self.tiled_enviro)

        self.assertAlmostEqual(
            swarm_agent.calculated_collective_opinion,
            0.4545,
            msg="internal_collective_opinion should be 0.495",
            places=9,
        )

    def test_recieve_local_opinions_in_top_left_corner(self):
        other_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(0, 1)], sensing=0
        )
        other_corner_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(4, 4)], sensing=0
        )

        other_swarm_agent.num_of_white_cells_observed += 3
        other_swarm_agent.num_of_cells_observed += 3
        other_corner_swarm_agent.num_of_cells_observed += 1

        self.swarm_agent.recieve_local_opinions(tile_grid=self.tiled_enviro)

        self.assertAlmostEqual(
            self.swarm_agent.calculated_collective_opinion,
            0.55,
            msg="internal_collective_opinion should be 0.55",
            places=9,
        )

    def test_recieve_local_opinions_in_bottom_right_corner(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])

        top_left_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(0, 0)], sensing=0
        )

        bottom_right_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(4, 4)], sensing=1
        )

        other_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(3, 4)], sensing=0
        )

        top_left_swarm_agent.num_of_white_cells_observed += 3
        top_left_swarm_agent.num_of_cells_observed += 3
        other_swarm_agent.num_of_cells_observed += 1

        bottom_right_swarm_agent.recieve_local_opinions(tile_grid=self.tiled_enviro)

        self.assertAlmostEqual(
            bottom_right_swarm_agent.calculated_collective_opinion,
            0.45,
            msg="internal_collective_opinion should be 0.45",
            places=9,
        )

    def test_recieve_local_opinions_in_bottom_right_corner(self):
        self.swarm_agent.leave_cell(self.tiled_enviro[(0, 0)])

        top_left_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(0, 0)], sensing=0
        )

        bottom_right_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(4, 4)], sensing=1
        )

        other_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(3, 4)], sensing=0
        )

        top_left_swarm_agent.num_of_white_cells_observed += 3
        top_left_swarm_agent.num_of_cells_observed += 3
        other_swarm_agent.num_of_cells_observed += 1

        bottom_right_swarm_agent.recieve_local_opinions(tile_grid=self.tiled_enviro)

        self.assertAlmostEqual(
            bottom_right_swarm_agent.calculated_collective_opinion,
            0.45,
            msg="internal_collective_opinion should be 0.45",
            places=9,
        )

    def test_swarm_agent_equation_based_opinion_weights(self):
        self.swarm_agent.calculated_collective_opinion = 0.75
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(1), 0.075
        )
        self.swarm_agent.calculated_collective_opinion = 0.25
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(1), 0.025
        )
        self.swarm_agent.calculated_collective_opinion = 0.75
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(0), 0.025
        )
        self.swarm_agent.calculated_collective_opinion = 0.25
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(0), 0.075
        )

    def test_swarm_agent_equation_based_opinion_weights_with_different_opinion_weighting(
        self,
    ):
        self.swarm_agent.max_new_opinion_weighting = 0.25
        self.swarm_agent.calculated_collective_opinion = 0.75
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(1), 0.1875
        )
        self.swarm_agent.calculated_collective_opinion = 0.25
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(1), 0.0625
        )
        self.swarm_agent.calculated_collective_opinion = 0.75
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(0), 0.0625
        )
        self.swarm_agent.calculated_collective_opinion = 0.25
        self.assertEqual(
            self.swarm_agent.return_opinion_weight_based_on_equation(0), 0.1875
        )

    def test_swarm_agent_equation_based_opinion_weights_update_collective_opinion(self):
        temp_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(0, 1)],
            opinion_weighting_method="equation_based",
        )
        temp_swarm_agent.calculated_collective_opinion = 0.75
        temp_swarm_agent.update_collective_opinion(1)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.76875)

        temp_swarm_agent.calculated_collective_opinion = 0.25
        temp_swarm_agent.update_collective_opinion(1)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.26875)

        temp_swarm_agent.calculated_collective_opinion = 0.75
        temp_swarm_agent.update_collective_opinion(0)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.73125)

        temp_swarm_agent.calculated_collective_opinion = 0.25
        temp_swarm_agent.update_collective_opinion(0)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.23125)

    def test_swarm_agent_equation_based_opinion_weights_with_different_opinion_weighting_update_collective_opinion(
        self,
    ):
        temp_swarm_agent = SwarmAgent(
            starting_cell=self.tiled_enviro[(0, 1)],
            opinion_weighting_method="equation_based",
            max_new_opinion_weighting=0.25,
        )
        temp_swarm_agent.calculated_collective_opinion = 0.75
        temp_swarm_agent.update_collective_opinion(1)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.796875)

        temp_swarm_agent.calculated_collective_opinion = 0.25
        temp_swarm_agent.update_collective_opinion(1)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.296875)

        temp_swarm_agent.calculated_collective_opinion = 0.75
        temp_swarm_agent.update_collective_opinion(0)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.703125)

        temp_swarm_agent.calculated_collective_opinion = 0.25
        temp_swarm_agent.update_collective_opinion(0)
        self.assertEqual(temp_swarm_agent.calculated_collective_opinion, 0.203125)


if __name__ == "__main__":
    unittest.main()
