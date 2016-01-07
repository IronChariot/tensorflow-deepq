"""
Details of AsteroidGame:
- Hero can be always in the middle (movement makes everything ELSE move) or free
- Hero can accelerate forwards and has momentum
- Hero has a current direction and field of view (so looking around has value!)
- Inputs are:
    closest object in each sight line
    distance to object in sight line
    object velocity relative to sight line
    hero's velocity relative to the direction he faces
- Hero outputs are:
    shoot ahead (instant laser, range slightly shorter than sight radius)
    accelerate forward
    turn clockwise
    turn anticlockwise
- Environment consists of:
    asteroids which harm hero on contact,
    and can be shot to produce smaller asteroids and gems
    asteroids are a random size between min and max
    asteroids below a certain size are completely destroyed when shot
- Rewards are:
    Large positive reward from collecting gems
    Large negative reward from being hit by asteroids
    (penalty scales with size of asteroid)

TODO:
    Fix bug which makes hero unable to see beyond walls when not fixed to centre
    Make it clearer when hero is moving, when hero is fixed in the centre (background could move?)
    Fix whatever bug is causing the hero not to flash red when colliding with an asteroid
"""

import math
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, speed, obj_type, radius, settings):
        """Essentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = radius

        self.obj_type = obj_type
        self.position = position
        self.speed    = speed
        self.bounciness = 1.0

    def wall_collisions(self, hero_speed):
        """Update position upon collision with the wall."""
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and (self.speed[dim] - hero_speed[dim]) < 0:
                self.position[dim] = world_size[dim] - self.radius - 1
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and (self.speed[dim] - hero_speed[dim]) > 0:
                self.position[dim] = self.radius + 1

    def move(self, dt, hero_speed):
        """Move as if dt seconds passed"""
        self.position += dt * (self.speed - hero_speed)
        self.position = Point2(*self.position)

    def translate(self, x, y):
        """ Add the vector (x, y) to current object position. """
        self.position += Vector2(x, y)
        self.position = Point2(*self.position)

    def step(self, dt, hero_speed):
        """Move and react to walls."""
        self.wall_collisions(hero_speed)
        self.move(dt, hero_speed)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, temp_color=None):
        """Return svg object for this item."""
        if temp_color:
            color = temp_color
        else:
            color = self.settings["colors"][self.obj_type]
        return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)

class AsteroidGame(object):
    def __init__(self, settings):
        """Initialize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]

        self.hero = GameObject(Point2(*self.settings["hero_initial_position"]),
                               Vector2(*self.settings["hero_initial_speed"]),
                               "hero",
                               self.settings["hero_radius"],
                               self.settings)
        # Hero direction is a number between 0 and num_observation_lines_total - 1 (0 is directly right)
        self.hero_direction = self.settings["hero_initial_direction"]
        self.unit_vectors = []
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines_total"], endpoint=False):
            u_vector = Point2(math.cos(angle), math.sin(angle))
            self.unit_vectors.append(u_vector)

        self.objects = []
        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                self.spawn_object(obj_type)

        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []
        self.just_shot = False
        self.just_got_hit = False

        # every observation_line sees one of objects and
        # two numbers representing speed of the object (if applicable)
        self.eye_observation_size = len(self.settings["objects"]) + 2
        # additionally there are two numbers representing agents own speed.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2

        # self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1]]]
        self.num_actions = 4  # Accelerate, turn clockwise, turn anticlockwise, shoot

        self.objects_eaten = defaultdict(lambda: 0)
        self.objects_shot = defaultdict(lambda: 0)

    def perform_action(self, action_id):
        """Change speed or direction of hero"""
        assert 0 <= action_id < self.num_actions
        if action_id == 0:  # Accelerate in direction hero is facing
            self.just_shot = False
            self.hero.speed *= self.settings["hero_momentum"]
            self.hero.speed += self.unit_vectors[self.hero_direction] * self.settings["delta_v"]
        if action_id == 1:  # Turn clockwise
            self.just_shot = False
            self.hero_direction += 1
            if self.hero_direction > 39: self.hero_direction = 0
            self.observation_lines = self.generate_observation_lines()
        if action_id == 2:  # Turn anticlockwise
            self.just_shot = False
            self.hero_direction -= 1
            if self.hero_direction < 0: self.hero_direction = 39
            self.observation_lines = self.generate_observation_lines()
        if action_id == 3:  # Shoot laser
            self.just_shot = True
            shooting_distance = self.settings["laser_line_length"]
            relevant_objects = [obj for obj in self.objects
                                if obj.position.distance(self.hero.position) < shooting_distance]
            shooting_line = LineSegment2(self.hero.position,
                                         self.hero.position +
                                            (self.unit_vectors[self.hero_direction] * shooting_distance))
            # Sort relevant objects from closest to hero to farthest
            relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))
            for obj in relevant_objects:
                if shooting_line.distance(obj.position) < obj.radius:
                    self.objects_shot[obj.obj_type] += 1
                    if obj.obj_type == 'asteroids':
                        # Split up large asteroids and spawn gems
                        if (obj.radius / 2) > self.settings["min_asteroid_radius"]:
                            self.spawn_cluster(2, 2, obj.radius, obj.position, obj.speed)
                    self.objects.remove(obj)
                    respawnable = ['asteroids']
                    if (obj.obj_type in respawnable and
                        len([thing for thing in self.objects if thing.obj_type == obj.obj_type])
                        < self.settings["num_objects"][obj.obj_type]):
                        self.spawn_object(obj.obj_type)
                    break


    def spawn_object(self, obj_type):
        """Spawn object of a given type and add it to the objects array"""
        if obj_type == 'asteroids':
            radius = np.random.uniform(self.settings["min_asteroid_radius"], self.settings["max_asteroid_radius"])
        else:
            radius = self.settings["object_radius"]
        collision_distance = self.hero.radius + radius
        valid_position = False
        while valid_position == False:
            position = np.random.uniform([radius, radius], np.array(self.size) - radius)
            position = Point2(float(position[0]), float(position[1]))
            if self.squared_distance(self.hero.position, position) > (collision_distance ** 2):
                valid_position = True
        max_speed = np.array(self.settings["maximum_speed"])
        speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        speed = Vector2(float(speed[0]), float(speed[1]))
        self.objects.append(GameObject(position, speed, obj_type, radius, self.settings))

    def spawn_cluster(self, num_asteroids, num_gems, radius, position, speed):
        """ Creates a certain number of smaller asteroids and gems based on radius and position. """
        new_radius = radius / 2
        # Choose random starting point for aligning children:
        start = np.random.randint(0, self.settings["num_observation_lines_total"] / num_asteroids)
        # Get as many unit vectors as there are asteroid children:
        gap = self.settings["num_observation_lines_total"] / num_asteroids
        directions = range(start, self.settings["num_observation_lines_total"], gap)
        vectors = [self.unit_vectors[d] for d in directions]
        positions = [Point2(*(position + (vect * new_radius))) for vect in vectors]
        speeds = [Vector2(*((vect * new_radius) + speed)) for vect in vectors]
        for i in range(num_asteroids):
            self.objects.append(GameObject(positions[i], speeds[i], 'asteroids', new_radius, self.settings))

        # For the gems, get some random vectors for the positions, speed is original asteroid speed
        gem_radius = self.settings["gem_radius"]
        for i in range(num_gems):
            pos = Point2(*(position +
                           (self.unit_vectors[np.random.randint(0, self.settings["num_observation_lines_total"])]
                            * gem_radius)))
            self.objects.append(GameObject(pos, speed, 'gems', gem_radius, self.settings))

    def step(self, dt):
        """Simulate all the objects for a given amount of time.

        Also resolve collisions with the hero"""
        if self.settings["keep_hero_in_middle"]:
            for obj in self.objects:
                obj.step(dt, self.hero.speed)
        else:
            for obj in self.objects + [self.hero]:
                obj.step(dt, Vector2(0.0, 0.0))
        self.resolve_collisions()

    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def resolve_collisions(self):
        """If hero touches, hero eats. Also reward gets updated."""
        hero_radius = self.hero.radius
        to_remove = []
        to_bounce = []
        self.just_got_hit = False
        for obj in self.objects:
            collision_distance = hero_radius + obj.radius
            if self.squared_distance(self.hero.position, obj.position) <= (collision_distance ** 2):
                if obj.obj_type == 'gems':
                    to_remove.append(obj)
                else:
                    to_bounce.append(obj)
        for obj in to_remove:
            self.objects.remove(obj)
            self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type]
        for obj in to_bounce:
            self.just_got_hit = True
            self.objects_eaten[obj.obj_type] += 1
            self.object_reward += self.settings["object_reward"][obj.obj_type] * obj.radius
            # Do elastic collisions to work out new velocities of object and hero
            # Mass is assumed to be radius^2 (discs as opposed to spheres)
            # Elastic collision, so both energy and momentum is conserved
            dx = self.hero.position[0] - obj.position[0]
            dy = self.hero.position[1] - obj.position[1]
            angle = math.atan2(dy, dx) + 0.5 * math.pi
            h_mass = hero_radius ** 2
            o_mass = obj.radius ** 2
            h_x_vel = self.hero.speed[0]
            h_y_vel = self.hero.speed[1]
            o_x_vel = obj.speed[0]
            o_y_vel = obj.speed[1]
            h_angle = np.arctan2(h_y_vel, h_x_vel)
            o_angle = np.arctan2(o_y_vel, o_x_vel)
            h_speed = ((h_x_vel ** 2) + (h_y_vel ** 2)) ** 0.5
            o_speed = ((o_x_vel ** 2) + (o_y_vel ** 2)) ** 0.5
            total_mass = h_mass + o_mass

            vector1_h = np.array([h_angle, h_speed * (h_mass - o_mass) / total_mass])
            vector2_h = np.array([angle, 2 * o_speed * o_mass / total_mass])
            (h_angle, h_speed) = tuple(vector1_h + vector2_h) # Add the two vectors above

            vector1_o = np.array([o_angle, o_speed * (o_mass - h_mass) / total_mass])
            vector2_o = np.array([angle + math.pi, 2 * h_speed * h_mass / total_mass])
            (o_angle, o_speed) = tuple(vector1_o + vector2_o) # Add the two vectors above

            h_xv = h_speed * np.cos(h_angle)
            h_yv = h_speed * np.sin(h_angle)
            o_xv = o_speed * np.cos(o_angle)
            o_yv = o_speed * np.sin(o_angle)

            self.hero.speed = Vector2(float(h_xv), float(h_yv))
            obj.speed = Vector2(float(o_xv), float(o_yv))

            collision_distance = hero_radius + obj.radius
            x_displ = -np.sin(angle)
            y_displ = np.cos(angle)
            while self.squared_distance(self.hero.position, obj.position) <= (collision_distance ** 2):
                obj.translate(float(x_displ), float(y_displ))

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing or an object.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"])
        max_speed_x, max_speed_y = self.settings["maximum_speed"]
        max_speed = ((max_speed_x ** 2) + (max_speed_y ** 2)) ** 0.5

        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation        = np.zeros(self.observation_size)
        observation_offset = 0
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))
            # Unit vector for observation line:
            line_direction = self.hero_direction - (self.settings["num_observation_lines"] / 2) + i
            if line_direction < 0:
                line_direction += self.settings["num_observation_lines_total"]
            elif line_direction >= self.settings["num_observation_lines_total"]:
                line_direction -= self.settings["num_observation_lines_total"]
            line_vector = self.unit_vectors[line_direction]
            # Get unit vector perpendicular to observation line:
            p_line_direction = line_direction + (self.settings["num_observation_lines_total"] / 4)
            if p_line_direction < 0:
                p_line_direction += self.settings["num_observation_lines_total"]
            elif p_line_direction >= self.settings["num_observation_lines_total"]:
                p_line_direction -= self.settings["num_observation_lines_total"]
            p_line_vector = self.unit_vectors[p_line_direction]

            observed_object = None
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < obj.radius:
                    observed_object = obj
                    break
            object_type_id = None
            speed_x, speed_y = 0, 0
            rel_speed_along_line = 0.0
            rel_speed_across_line = 0.0
            proximity = 0
            if observed_object is not None: # object seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                rel_speed_along_line = line_vector.dot(Vector2(speed_x - self.hero.speed[0], speed_y - self.hero.speed[1]))
                rel_speed_across_line = p_line_vector.dot(Vector2(speed_x - self.hero.speed[0], speed_y - self.hero.speed[1]))
                intersection_segment = obj.as_circle().intersect(observation_line)
                assert intersection_segment is not None
                try:
                    proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                except AttributeError:
                    proximity = observable_distance
            for object_type_idx_loop in range(num_obj_types):
                observation[observation_offset + object_type_idx_loop] = 1.0
            if object_type_id is not None:
                observation[observation_offset + object_type_id] = proximity / observable_distance
            # rel_speed_across_line = ((speed_x ** 2) + (speed_y ** 2) - (rel_speed_along_line ** 2)) ** 0.5
            observation[observation_offset + num_obj_types] =     rel_speed_along_line / max_speed
            observation[observation_offset + num_obj_types + 1] = rel_speed_across_line / max_speed
            assert num_obj_types + 2 == self.eye_observation_size, "{} and {}".format(num_obj_types, self.eye_observation_size)
            observation_offset += self.eye_observation_size

        # Realign hero speed to be relative to the direction it is facing:
        facing_uv = self.unit_vectors[self.hero_direction]
        sideways_dir = self.hero_direction + (self.settings["num_observation_lines_total"] / 4)
        if sideways_dir >= self.settings["num_observation_lines_total"]:
            sideways_dir -= self.settings["num_observation_lines_total"]
        sideways_uv = self.unit_vectors[sideways_dir]
        forward_v = facing_uv.dot(self.hero.speed)
        sideways_v = sideways_uv.dot(self.hero.speed)
        # sideways_v = ((self.hero.speed[0] ** 2) + (self.hero.speed[1] ** 2) - (forward_v ** 2)) ** 0.5
        observation[observation_offset]     = forward_v / max_speed
        observation[observation_offset + 1] = sideways_v / max_speed
        assert observation_offset + 2 == self.observation_size

        return observation

    def collect_reward(self):
        """Return accumulated object eating score"""
        total_reward = self.object_reward
        self.object_reward = 0
        self.collected_rewards.append(total_reward)
        return total_reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        for i in range(self.hero_direction - (self.settings["num_observation_lines"] / 2),
                       self.hero_direction + (self.settings["num_observation_lines"] / 2) + 1):
            if i < 0: j = 40 + i
            elif i >= 40: j = i - 40
            else: j = i
            result.append( LineSegment2(Point2(0.0, 0.0),
                                        self.unit_vectors[j] * self.settings["observation_line_length"]) )
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            "reward       = %.1f" % (sum(recent_reward)/len(recent_reward),),
            "objects eaten => %s" % (objects_eaten_str,),
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))

        for i, line in enumerate(self.observation_lines):
            if self.just_shot and i == (self.settings["num_observation_lines"] / 2):
                scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                                   line.p2 + self.hero.position + Point2(10,10), color='red'))
            else:
                scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                                   line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects:
            scene.add(obj.draw())

        if self.just_got_hit:
            scene.add(self.hero.draw('red'))
        else:
            scene.add(self.hero.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene
