from typing import Tuple
import cv2
import numpy as np
import tkinter as tk
from environment_agent_modules import (
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    SwarmAgent,
    Direction,
    TileColour,
)
import random
from helper_files import return_list_of_coordinates_column_by_columns

TILE_SIZE = 25
MID_POINT_OFFSET = round(TILE_SIZE / 2)
root = tk.Tk()
SCREEN_MID_POINT = (
    round(root.winfo_screenwidth() / 2),
    round(root.winfo_screenheight() / 2),
)


def setup_tile_grid_for_display(tile_grid: np.ndarray) -> np.ndarray:
    display_grid = np.zeros(
        (
            tile_grid.shape[0] * TILE_SIZE,
            tile_grid.shape[1] * TILE_SIZE,
            3,
        ),
        np.uint8,
    )

    for row in range(tile_grid.shape[0]):
        for col in range(tile_grid.shape[1]):
            if TileColour(tile_grid[row, col]["colour"]) == TileColour.WHITE:
                display_grid[
                    row * TILE_SIZE : row * TILE_SIZE + TILE_SIZE,
                    col * TILE_SIZE : col * TILE_SIZE + TILE_SIZE,
                ] = (
                    255,
                    255,
                    255,
                )

    return display_grid


def map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
    agent_direction: int, agent_position: Tuple[int, int]
) -> Tuple[int, int]:
    return {
        Direction.UP.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row),
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
        ),
        Direction.RIGHT.value: lambda row, column: (
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
            (column, row - MID_POINT_OFFSET),
        ),
        Direction.DOWN.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
            (column - MID_POINT_OFFSET, row),
        ),
        Direction.LEFT.value: lambda row, column: (
            (column, row - MID_POINT_OFFSET),
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
        ),
    }[agent_direction](*agent_position)


def return_arrow_colour_based_on_agent_state(
    swarm_agent: SwarmAgent,
) -> Tuple[int, int, int]:
    if not (swarm_agent.committed_to_opinion):
        if not (swarm_agent.sensing):
            return (0, 0, 255)

        return (0, 255, 0)

    return (255, 0, 0)


def draw_swarm_agent_on_tile_grid(
    display_tile_grid: np.ndarray,
    swarm_agent: SwarmAgent,
) -> np.ndarray:
    (
        start_point,
        end_point,
    ) = map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
        agent_direction=swarm_agent.current_direction_facing,
        agent_position=[(value + 1) * TILE_SIZE for value in swarm_agent.current_cell],
    )

    return cv2.arrowedLine(
        display_tile_grid,
        start_point,
        end_point,
        return_arrow_colour_based_on_agent_state(swarm_agent),
        2,
        tipLength=0.4,
    )


def move_and_show_window(x: int, y: int, winname: str, img: np.ndarray) -> None:
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


def main() -> None:
    tile_grid = create_nonclustered_tile_grid(25, 25, ratio_of_white_to_black_tiles=0.6)

    list_of_locations = return_list_of_coordinates_column_by_columns(
        num_of_columns=25, num_of_rows=25
    )

    swarm_agents = [
        SwarmAgent(
            starting_cell=(tile_grid[list_of_locations.pop(0)]),
            current_direction_facing=1,
            needs_models_loaded=True,
        )
        for _ in range(25)
    ]

    display_location = (
        round(SCREEN_MID_POINT[0] - (tile_grid.shape[1] * TILE_SIZE / 2)),
        round(SCREEN_MID_POINT[1] - (tile_grid.shape[0] * TILE_SIZE / 1.5)),
    )

    for i in range(200):
        img = setup_tile_grid_for_display(tile_grid)

        for agent in swarm_agents:
            draw_swarm_agent_on_tile_grid(img, agent)
            agent.navigate(tile_grid)

        move_and_show_window(*display_location, winname="Tile Grid", img=img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
