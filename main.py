
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib import transforms
import csv
import numpy as np
import random
import cv2

# Simulation parameters
dt = 1/30  
speed_limit = 30 * 0.44704
observation_modes = {
    'queue': cv2.imread('kernel_queuing.png', cv2.IMREAD_GRAYSCALE)/255.0,
    'merge': cv2.imread('kernel_merging.png', cv2.IMREAD_GRAYSCALE)/255.0,
    'drive': cv2.imread('kernel_driving.png', cv2.IMREAD_GRAYSCALE)/255.0,
    'test': cv2.imread('kernel_test.png', cv2.IMREAD_GRAYSCALE)/255.0,
    'lane_change_left': cv2.imread('kernel_lane_change_left.png', cv2.IMREAD_GRAYSCALE)/255.0
}

class Roundabout():
    def __init__(self):
        self.routes = {}
        self.routes_om = {}
        self.cars = []
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

    def add_car(self):
        entrances = ['1', '2', '3', '4']
        entrance = random.choice(entrances)
        exits = [e for e in entrances if e != entrance]
        exit = random.choice(exits)
        confidence = random.uniform(0, 1)
        car = Car(road=self, entrance=entrance, exit=exit, confidence=confidence)
        self.cars.append(car)
    
class Car():
    def __init__(self, road, entrance, exit, confidence=0.5):
        self.road = road
        self.entrance = entrance
        self.exit = exit

        self.observation_mode = observation_modes['drive']
        self.last_ob_mode = self.observation_mode.copy()

        self.path = self.road.routes[entrance + exit]
        self.position = self.path[0].copy()
        self.speed = 0
        self.angle = np.arctan2(self.path[1][1] - self.path[0][1], self.path[1][0] - self.path[0][0])
        self.length = 4.7
        self.width = 1.9
        self.confidence = confidence

        self.time_elapsed = 0.0

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
        if len(self.road.cars) == 0:
            return
        for car in self.road.cars:
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

    def set_drive_controls(self):
        self.time_elapsed += dt

        # Find the closest point on the path
        distances = [np.linalg.norm(point - self.position) for point in self.path]
        closest_point_index = np.argmin(distances)
        self.closest_point = self.path[closest_point_index]

        self.observation_mode = observation_modes[self.road.routes_om[self.entrance + self.exit][closest_point_index]]

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
        if self.road.cars is not None:
           for car in [car for car in self.road.cars if car is not self]:
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

class QuickRun():
    def __init__(self, timesteps=1000):
        self.road = Roundabout()
        self.timesteps = timesteps
        self.times_elapsed = []
        self.last_five_speeds = []

        for _ in range(25):
            self.road.add_car()

        self.run_simulation()
    
    def run_simulation(self):
        for _ in range(self.timesteps):

            average_speed = np.mean([car.speed for car in self.road.cars]) if self.road.cars else 0
            self.last_five_speeds.append(average_speed)
            if len(self.last_five_speeds) > 5:
                self.last_five_speeds = self.last_five_speeds[1:]
                if max(self.last_five_speeds) == 0:
                    print("All cars have stopped. Ending simulation.")
                    return False, self.times_elapsed
                
            # Check for collisions
            for i, car1 in enumerate(self.road.cars):
                corners1 = self.get_corners(car1.position, car1.length, car1.width, car1.angle)
                for j, car2 in enumerate(self.road.cars):
                    if i >= j:
                        continue
                    corners2 = self.get_corners(car2.position, car2.length, car2.width, car2.angle)
                    if self.rectangles_intersect(corners1, corners2):
                        print(f"Collision detected between cars. Ending simulation at time {_ * dt}.")
                        return False, self.times_elapsed


            if len(self.road.cars) < 25:
                self.road.add_car()

            for car in self.road.cars:
                car.set_drive_controls()
                car.update_position()
                if np.linalg.norm(car.position - car.path[-1]) < 1.0:
                    self.times_elapsed.append(car.time_elapsed)
                    self.road.cars.remove(car)
        print(f"All times for cars: {self.times_elapsed}")
        print(f"Simulation complete with average time elapsed: {np.mean(self.times_elapsed):.2f} seconds.")
        return True, self.times_elapsed

    def get_corners(self, pos, length, width, angle):
        """Returns the 4 corners of a rotated rectangle centered at pos"""
        dx = length / 2
        dy = width / 2

        # Rectangle in local space
        corners = np.array([
            [-dx, -dy],
            [-dx,  dy],
            [ dx,  dy],
            [ dx, -dy]
        ])

        # Rotation matrix
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        return np.dot(corners, rotation.T) + pos
    
    def rectangles_intersect(self, rect1, rect2):
        """Check if two rectangles (given by 4 corners) intersect using SAT"""
        def get_axes(corners):
            return [corners[i] - corners[i - 1] for i in range(4)]

        def project(corners, axis):
            projections = np.dot(corners, axis)
            return [min(projections), max(projections)]

        axes = get_axes(rect1) + get_axes(rect2)
        axes = [axis / np.linalg.norm(axis) for axis in axes]

        for axis in axes:
            proj1 = project(rect1, axis)
            proj2 = project(rect2, axis)
            if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
                return False  # No overlap
        return True  # Overlap found!



class LivePlotter2d():
    def __init__(self, timesteps=1000):
        self.road = Roundabout()
        self.timesteps = timesteps
        self.show_ob_fields = True
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-33.3, 33.3)
        self.ax.set_ylim(-33.3, 33.3)
        self.ax.set_title("Live Car Simulation")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_aspect('equal')

        img = mpimg.imread('Background.jpg')
        self.ax.imshow(img, extent=[-33.3, 33.3, -33.3, 33.3], aspect='auto', zorder=0)

        self.car_patches = []
        self.ob_field_patches = []

    def init_animation(self):
        for i, car in enumerate(self.road.cars):
            self.car_patches[i].set_xy(car.get_corners())
            if self.show_ob_fields:
                patches = self.ob_field_patches[i]
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
                            self.ax.add_patch(rect)
                            patches.append(rect)
                        else:
                            rect = patches[k]
                            rect.set_xy((world_x - grid_size/2, world_y - grid_size/2))
                            rect.set_alpha(val * 0.5)
                        t = transforms.Affine2D().rotate_around(world_x, world_y, car.angle)
                        rect.set_transform(t + self.ax.transData)
                        k += 1
        return self.car_patches + sum(self.ob_field_patches, []) if self.show_ob_fields else self.car_patches
    
    def animate(self, frame):
        if len(self.road.cars) < 25:
            self.road.add_car()
            self._make_car_patch(self.road.cars[-1])
        to_remove = []
        for i, car in enumerate(self.road.cars):
            car.set_drive_controls()
            car.update_position()
            self.car_patches[i].set_xy(car.get_corners())
            self.car_patches[i].set_color((
                np.clip(1 - car.throttle, 0, 1),
                np.clip(1 - car.brake, 0, 1),
                np.clip(1 - car.brake - car.throttle, 0, 1)
            ))

            if self.show_ob_fields:
                # Check if observation mode has changed (add a new attribute to track last one)
                if not hasattr(car, 'last_ob_mode') or not np.array_equal(car.observation_mode, car.last_ob_mode):
                    # Remove old patches
                    for p in self.ob_field_patches[i]:
                        p.remove()
                    self.ob_field_patches[i] = []

                patches = self.ob_field_patches[i]
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
                            self.ax.add_patch(rect)
                            patches.append(rect)
                        else:
                            rect = patches[k]
                            rect.set_xy((world_x - grid_size/2, world_y - grid_size/2))
                            rect.set_alpha(np.clip((val-0.5) * 0.5,0,1))
                        t = transforms.Affine2D().rotate_around(world_x, world_y, car.angle)
                        rect.set_transform(t + self.ax.transData)
                        k += 1
                car.last_ob_mode = car.observation_mode.copy()

            if np.linalg.norm(car.position - car.path[-1]) < 1.0:
                to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            self.car_patches[idx].remove()
            del self.car_patches[idx]
            del self.road.cars[idx]
            if self.show_ob_fields:
                for p in self.ob_field_patches[idx]:
                    p.remove()
                del self.ob_field_patches[idx]

        return self.car_patches + sum(self.ob_field_patches, []) if self.show_ob_fields else self.car_patches

    def _make_car_patch(self, car):
        car_patch = plt.Polygon(car.get_corners(), closed=True, color='w', alpha=0.4)
        self.ax.add_patch(car_patch)
        self.car_patches.append(car_patch)

        if self.show_ob_fields:
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
                    rect.set_transform(t + self.ax.transData)
                    self.ax.add_patch(rect)
                    patches.append(rect)
            self.ob_field_patches.append(patches)

    def go(self):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.timesteps, init_func=self.init_animation,
            interval=dt * 1000, blit=True, repeat=False
        )

        plt.show()

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class LivePlotter3d():
    def __init__(self, timesteps=1000):
        self.road = Roundabout()
        self.timesteps = timesteps
        self.show_ob_fields = False
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-33.3, 33.3)
        self.ax.set_ylim(-33.3, 33.3)
        self.ax.set_zlim(0, 20)
        self.ax.set_title("Live Car Simulation 3D")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.car_patches = []
        self.ob_field_patches = []

    def add_car_patch(self, car):
        patch = self._make_car_patch(car)
        self.car_patches.append(patch)

        if self.show_ob_fields:
            ob_patches = self._make_ob_field_patches(car)
            self.ob_field_patches.append(ob_patches)
        else:
            self.ob_field_patches.append([])

    def _make_car_patch(self, car):
        corners = car.get_corners()  # shape (4,2)
        verts = [[(x, y, 0.5) for x, y in corners]]  # Z = 0.5 height
        poly = Poly3DCollection(verts, facecolors='black', linewidths=0.5, alpha=0.4)
        self.ax.add_collection3d(poly)
        return poly

    def _make_ob_field_patches(self, car):
        patches = []
        ob = car.observation_mode
        grid_size = 3.0
        offset = -10 * grid_size
        for i in range(21):
            for j in range(21):
                val = ob[-1-j, -1-i]
                if val <= 0.5:
                    continue
                x = offset + i * grid_size
                y = offset + j * grid_size
                dx = x * np.cos(car.angle) - y * np.sin(car.angle)
                dy = x * np.sin(car.angle) + y * np.cos(car.angle)
                world_x = car.position[0] + dx
                world_y = car.position[1] + dy
                z = 1.0  # hover above car
                square = [
                    (world_x - grid_size/2, world_y - grid_size/2, z),
                    (world_x + grid_size/2, world_y - grid_size/2, z),
                    (world_x + grid_size/2, world_y + grid_size/2, z),
                    (world_x - grid_size/2, world_y + grid_size/2, z)
                ]
                poly = Poly3DCollection([square], color='black', alpha=np.clip((val - 0.5) * 0.5, 0, 1))
                self.ax.add_collection3d(poly)
                patches.append(poly)
        return patches

    def init_animation(self):
        self.car_patches.clear()
        self.ob_field_patches.clear()

        for car in self.road.cars:
            self.car_patches.append(self._make_car_patch(car))
            self.ob_field_patches.append(self._make_ob_field_patches(car) if self.show_ob_fields else [])
        return self.car_patches + sum(self.ob_field_patches, [])

    def animate(self, frame):
        if len(self.road.cars) < 25:
            self.road.add_car()
            self.add_car_patch(self.road.cars[-1])
        to_remove = []

        for i, car in enumerate(self.road.cars):
            car.set_drive_controls()
            car.update_position()
            corners = car.get_corners()
            verts = [[(x, y, 0.5) for x, y in corners]]
            self.car_patches[i].set_verts(verts)

            if self.show_ob_fields:
                for p in self.ob_field_patches[i]:
                    p.remove()
                self.ob_field_patches[i] = self._make_ob_field_patches(car)

            if np.linalg.norm(car.position - car.path[-1]) < 1.0:
                to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            self.car_patches[idx].remove()
            del self.car_patches[idx]
            del self.road.cars[idx]
            if self.show_ob_fields:
                for p in self.ob_field_patches[idx]:
                    p.remove()
                del self.ob_field_patches[idx]

        return self.car_patches + sum(self.ob_field_patches, [])

    def go(self):
        ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.timesteps,
            init_func=self.init_animation, interval=dt * 1000,
            blit=False, repeat=False
        )
        plt.show()

if __name__ == "__main__":
    quick_run = QuickRun(timesteps=2000)
    #plotter_2d = LivePlotter2d(timesteps=2000)
    #plotter_2d.go()
    #plotter_3d = LivePlotter3d(timesteps=500)
    #plotter_3d.go()