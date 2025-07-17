
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib import transforms
import csv
import numpy as np
import random
import cv2

class Roundabout():
    def __init__(self):
        self.routes = {}
        self.routes_om = {}
        with open('routes.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['Name']
                x = float(row['X'])
                y = float(row['Y'])
                observation_mode = row['observation_mode']
                if name not in self.routes:
                    self.routes[name] = []
                    self.routes_om[name] = []
                self.routes[name].append([x, y])
                self.routes_om[name].append(observation_mode)
        for name in self.routes:
            self.routes[name] = np.array(self.routes[name])
            self.routes_om[name] = np.array(self.routes_om[name])

class Car():
    def __init__(self, entrance, exit, confidence=0.5):
        self.entrance = entrance
        self.exit = exit

        self.observation_mode = observation_modes['drive']
        self.last_ob_mode = self.observation_mode.copy()

        self.path = roundabout.routes[entrance + exit]
        self.position = self.path[0].copy()
        self.speed = 0
        self.angle = np.arctan2(self.path[1][1] - self.path[0][1], self.path[1][0] - self.path[0][0])
        self.length = 4.7  # meters
        self.width = 1.9   # meters
        self.confidence = confidence

        self.place_in_queue()

    def __str__(self):
        return f"Car(entrance={self.entrance}, exit={self.exit}, confidence={self.confidence})"

    def __repr__(self):
        return self.__str__()

    def get_corners(self):
        # Car's local coordinates (centered at self.position)
        dx = self.length / 2
        dy = self.width / 2
        corners_local = np.array([
            [ dx,  dy],  # front right
            [ dx, -dy],  # front left
            [-dx, -dy],  # rear left
            [-dx,  dy],  # rear right
        ])
        # Rotation matrix
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        # Transform to global coordinates
        corners_global = self.position + corners_local @ R.T
        return corners_global

    def place_in_queue(self):
        if len(cars) == 0:
            return
        for car in cars:
            if car is not self:
                distance = np.linalg.norm(car.position - self.position)
                if distance < 4.0:
                    self.position -= 6.0 * np.array([np.cos(car.angle), np.sin(car.angle)])
                    self.place_in_queue()
                    break

    # Dynamics
    def update_position(self):
        # Update the position based on speed and angle
        self.position[0] += np.cos(self.angle) * self.speed * dt
        self.position[1] += np.sin(self.angle) * self.speed * dt

    def enact_driver_controls(self, throttle, brake, turn):
        # Update the speed based on acceleration
        max_accel = 15.0
        max_brake = 50.0
        self.speed += throttle * max_accel * dt
        self.speed -= brake * max_brake * dt
        self.speed = max(0, self.speed)  # Speed cannot be negative
        self.angle += self.speed * turn * dt

    @property
    def path_following(self):
        # Find the closest point on the path to the current position
        distances = np.linalg.norm(self.path - self.position, axis=1)
        return np.argmin(distances)

    def set_drive_controls(self, cars=None):
        # Find the closest point on the path
        distances = [np.linalg.norm(point - self.position) for point in self.path]
        closest_point_index = np.argmin(distances)
        self.closest_point = self.path[closest_point_index]

        self.observation_mode = observation_modes[roundabout.routes_om[self.entrance + self.exit][closest_point_index]]

        # Target point selection
        if distances[closest_point_index] > 4.0 and closest_point_index < len(self.path) - 1:
            self.target_point = self.closest_point
        else:
            target_idx = closest_point_index
            for i in range(closest_point_index + 1, len(self.path)):
                if np.linalg.norm(self.path[i] - self.position) >= 2.0:
                    target_idx = i
                    break
            self.target_point = self.path[target_idx]

        # Calculate direction and distance to target
        direction = self.target_point - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction /= distance

        yield_factor = 0.0
        if cars is not None:
           for car in [car for car in cars if car is not self]:
               distance = np.linalg.norm(car.position - self.position)
               if distance < 30.0:
                   relative_position = np.array([
                       [np.cos(-self.angle), -np.sin(-self.angle)],
                       [np.sin(-self.angle),  np.cos(-self.angle)]
                   ]) @ (car.position - self.position)

                   observation_mode_index = np.round(np.array([
                       -relative_position[1] / 3 + 10,
                       -relative_position[0] / 3 + 10
                   ])).astype(int)

                   # Stay within bounds
                   if 0 <= observation_mode_index[0] <= 20 and 0 <= observation_mode_index[1] <= 20:
                    yield_factor_temp = self.observation_mode[observation_mode_index[1], observation_mode_index[0]]
                    if yield_factor_temp > yield_factor:
                           yield_factor = yield_factor_temp

        # Throttle logic: 0 (no throttle), 1 (full throttle)
        self.throttle = min(np.clip(distance - 0.5, 0.0, 1.0),
                       np.clip(speed_limit - self.speed, 0.0, 1.0),
                       np.clip(1 - 2*yield_factor, 0.0, 1.0))

        # Brake logic: 0 (no brake), 1 (full brake)
        self.brake = max(np.clip(1.2 - distance, 0.0, 1.0),
                    np.clip(self.speed - speed_limit, 0.0, 1.0),
                    np.clip(2*yield_factor-1, 0.0, 1.0))

        # Steering logic: -1 (left), 1 (right)
        angle_to_target = np.arctan2(direction[1], direction[0])
        turn = angle_to_target - self.angle
        # Normalize turn to [-pi, pi]
        turn = (turn + np.pi) % (2 * np.pi) - np.pi
        turn = np.clip(turn / np.pi, -1.0, 1.0)

        # Mark the car for deletion if it has reached the end of its path
        if closest_point_index >= len(self.path) - 1 and np.linalg.norm(self.position - self.path[-1]) < 3.0:
            del self
            return

        # Apply controls
        self.enact_driver_controls(self.throttle, self.brake, turn)

if __name__ == "__main__":
    dt = 1/30  # Time step for simulation in seconds
    speed_limit = 30 * 0.44704
    observation_modes = {
        'queue': cv2.imread('kernel_queuing.png', cv2.IMREAD_GRAYSCALE)/255.0,
        'merge': cv2.imread('kernel_merging.png', cv2.IMREAD_GRAYSCALE)/255.0,
        'drive': cv2.imread('kernel_driving.png', cv2.IMREAD_GRAYSCALE)/255.0,
        'test': cv2.imread('kernel_test.png', cv2.IMREAD_GRAYSCALE)/255.0,
        'lane_change_left': cv2.imread('kernel_lane_change_left.png', cv2.IMREAD_GRAYSCALE)/255.0
    }
    show_ob_fields = True

    roundabout = Roundabout()
    cars = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-33.3, 33.3)
    ax.set_ylim(-33.3, 33.3)
    ax.set_title("Live Car Simulation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal')

    img = mpimg.imread('Background.jpg')
    ax.imshow(img, extent=[-33.3, 33.3, -33.3, 33.3], aspect='auto', zorder=0)

    car_patches = []
    ob_field_patches = []

    def add_car():
        entrances = ['1', '2', '3', '4']
        entrance = random.choice(entrances)
        exits = [e for e in entrances if e != entrance]
        exit = random.choice(exits)
        confidence = random.uniform(0, 1)
        car = Car(entrance=entrance, exit=exit, confidence=confidence)
        cars.append(car)
        car_patch = plt.Polygon(car.get_corners(), closed=True, color='w', alpha=0.4)
        ax.add_patch(car_patch)
        car_patches.append(car_patch)

        if show_ob_fields:
            patches = []
            ob = car.observation_mode
            grid_size = 3.0
            offset = -10 * grid_size
            for i in range(21):
                for j in range(21):
                    val = ob[-1-j, -1-i]
                    if val == 0:
                        continue
                    x = offset + j * grid_size
                    y = offset + i * grid_size
                    dx = x * np.cos(car.angle) - y * np.sin(car.angle)
                    dy = x * np.sin(car.angle) + y * np.cos(car.angle)
                    world_x = car.position[0] + dx
                    world_y = car.position[1] + dy
                    rect = Rectangle(
                        (world_x - grid_size/2, world_y - grid_size/2),
                        grid_size, grid_size,
                        color='white', alpha=val * 0.5, zorder=2
                    )
                    t = transforms.Affine2D().rotate_around(world_x, world_y, car.angle)
                    rect.set_transform(t + ax.transData)
                    ax.add_patch(rect)
                    patches.append(rect)
            ob_field_patches.append(patches)

    add_car()

    def init():
        for i, car in enumerate(cars):
            car_patches[i].set_xy(car.get_corners())
            if show_ob_fields:
                patches = ob_field_patches[i]
                ob = car.observation_mode
                grid_size = 3.0
                offset = -10 * grid_size
                k = 0
                for ii in range(21):
                    for jj in range(21):
                        val = ob[-1-jj, -1-ii]
                        if val == 0:
                            continue
                        x = offset + jj * grid_size
                        y = offset + ii * grid_size
                        dx = x * np.cos(car.angle) - y * np.sin(car.angle)
                        dy = x * np.sin(car.angle) + y * np.cos(car.angle)
                        world_x = car.position[0] + dx
                        world_y = car.position[1] + dy
                        if k >= len(patches):
                            rect = Rectangle(
                                (world_x - grid_size/2, world_y - grid_size/2),
                                grid_size, grid_size,
                                color='white', alpha=val * 0.5, zorder=2
                            )
                            ax.add_patch(rect)
                            patches.append(rect)
                        else:
                            rect = patches[k]
                            rect.set_xy((world_x - grid_size/2, world_y - grid_size/2))
                            rect.set_alpha(val * 0.5)
                        t = transforms.Affine2D().rotate_around(world_x, world_y, car.angle)
                        rect.set_transform(t + ax.transData)
                        k += 1
        return car_patches + sum(ob_field_patches, []) if show_ob_fields else car_patches

    def animate(frame):
        if len(cars) < 25:
            add_car()
        to_remove = []
        for i, car in enumerate(cars):
            car.set_drive_controls(cars=cars)
            car.update_position()
            car_patches[i].set_xy(car.get_corners())
            car_patches[i].set_color((
                np.clip(1 - car.throttle, 0, 1),
                np.clip(1 - car.brake, 0, 1),
                np.clip(1 - car.brake - car.throttle, 0, 1)
            ))

            if show_ob_fields:
                # Check if observation mode has changed (add a new attribute to track last one)
                if not hasattr(car, 'last_ob_mode') or not np.array_equal(car.observation_mode, car.last_ob_mode):
                    # Remove old patches
                    for p in ob_field_patches[i]:
                        p.remove()
                    ob_field_patches[i] = []

                patches = ob_field_patches[i]
                ob = car.observation_mode
                grid_size = 3.0
                offset = -10 * grid_size
                k = 0
                for ii in range(21):
                    for jj in range(21):
                        val = ob[-1-jj, -1-ii]
                        if val <= 0.5:
                            continue
                        x = offset + jj * grid_size
                        y = offset + ii * grid_size
                        dx = x * np.cos(car.angle) - y * np.sin(car.angle)
                        dy = x * np.sin(car.angle) + y * np.cos(car.angle)
                        world_x = car.position[0] + dx
                        world_y = car.position[1] + dy
                        if k >= len(patches):
                            rect = Rectangle(
                                (world_x - grid_size/2, world_y - grid_size/2),
                                grid_size, grid_size,
                                color='white', alpha=val * 0.5, zorder=2
                            )
                            rect.set_edgecolor(None)
                            ax.add_patch(rect)
                            patches.append(rect)
                        else:
                            rect = patches[k]
                            rect.set_xy((world_x - grid_size/2, world_y - grid_size/2))
                            rect.set_alpha(np.clip((val-0.5) * 0.5,0,1))
                        t = transforms.Affine2D().rotate_around(world_x, world_y, car.angle)
                        rect.set_transform(t + ax.transData)
                        k += 1
                car.last_ob_mode = car.observation_mode.copy()

            if np.linalg.norm(car.position - car.path[-1]) < 1.0:
                to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            car_patches[idx].remove()
            del car_patches[idx]
            del cars[idx]
            if show_ob_fields:
                for p in ob_field_patches[idx]:
                    p.remove()
                del ob_field_patches[idx]

        return car_patches + sum(ob_field_patches, []) if show_ob_fields else car_patches

    ani = animation.FuncAnimation(
        fig, animate, frames=5000, init_func=init,
        interval=dt * 1000, blit=True, repeat=False
    )

    plt.show()
