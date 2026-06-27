import msgParser
import carState
import carControl
import numpy as np
import model
import csv
import time

class Driver(object):
    def __init__(self, stage):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.model = model.RacingNet()
        self.current_gear = 1
        self.gear_speed_limits = {
            0: (0, 0),      # Neutral
            1: (0, 45),     # Gear 1: 0–45 km/h
            2: (46, 95),    # Gear 2: 46–95 km/h
            3: (96, 140),   # Gear 3: 96–140 km/h
            4: (141, 185),  # Gear 4: 141–185 km/h
            5: (186, 230),  # Gear 5: 186–230 km/h
            -1: (-10, 0)    # Reverse: -10–0 km/h (TORCS uses -1 for reverse)
        }
        self.data_buffer = []
        self.csv_initialized = False
        self.accelerating = False
        self.steer = 0.0
        self.steer_target = 0.0

    def init(self):
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})

    def drive(self, msg, gear_shift_trigger, steer_trigger, accel_trigger, brake_trigger):
        self.state.setFromMsg(msg)
        state_vector = self.get_state_vector()
        if state_vector is None:
            return self.control.toMsg()

        accel = 0.0
        brake = 0.0

        # Get current speed
        raw_speedX = self.state.getSpeedX() or 0.0
        speedX = raw_speedX * 1.2
        min_speed, max_speed = self.gear_speed_limits[self.current_gear]
        target_speed = max_speed if self.current_gear != -1 else min_speed

        # Debug: Log raw message and speeds
        print(f"Raw msg: {msg[:100]}...")
        print(f"Raw speedX: {raw_speedX:.2f} m/s, Converted speedX: {speedX:.2f} km/h, Gear: {self.current_gear}")
        print(f"Target speed: {target_speed:.2f} km/h, Accel trigger: {accel_trigger}, Brake trigger: {brake_trigger}")

        # Handle gear selection
        if gear_shift_trigger in ['1', '2', '3', '4', '5', '6']:
            new_gear = int(gear_shift_trigger) if gear_shift_trigger != '6' else -1  # Use -1 for reverse
            min_new, max_new = self.gear_speed_limits[new_gear]
            if new_gear > self.current_gear and abs(speedX - min_new) < 5 and new_gear != -1:  # Shift up
                self.current_gear = new_gear
                target_speed = max_new
                self.accelerating = True
                print(f"Shifted up to gear {new_gear}, target speed: {target_speed} km/h")
            elif new_gear < self.current_gear and abs(speedX - max_speed) < 5 and self.current_gear != -1:  # Shift down
                self.current_gear = new_gear
                target_speed = max_new
                self.accelerating = True
                print(f"Shifted down to gear {new_gear}, target speed: {target_speed} km/h")
            elif new_gear == -1:  # Reverse
                self.current_gear = -1
                target_speed = -10
                self.accelerating = False
                print(f"Set reverse gear (-1), target speed: {target_speed} km/h")
            elif new_gear == self.current_gear:  # Same gear
                target_speed = max_speed if new_gear != -1 else -10
                self.accelerating = True if new_gear != -1 else False
                print(f"Same gear {new_gear}, target speed: {target_speed} km/h")
            else:  # Invalid shift
                target_speed = max_speed if self.current_gear != -1 else -10
                print(f"Invalid gear shift to {new_gear}, staying in gear {self.current_gear}, target speed: {target_speed} km/h")

        self.control.setGear(self.current_gear)

        # Handle braking
        if brake_trigger == 'down':
            accel = 0.0
            brake = 1.0
            self.accelerating = False
            print(f"Braking applied, speed: {speedX:.2f} km/h")
        else:
            # Handle acceleration
            if self.current_gear == -1:  # Reverse gear
                if accel_trigger == 'up':
                    if speedX > -10:  # Accelerate backward
                        accel = 1.0
                        brake = 0.0
                        print(f"Accelerating backward, speed: {speedX:.2f} km/h, target: -10 km/h")
                    else:  # Maintain -10 km/h
                        accel = 0.3  # Increased to ensure maintenance
                        brake = 0.0
                        print(f"Maintaining reverse speed: {speedX:.2f} km/h")
                else:  # No up key, coast
                    accel = 0.0
                    brake = 0.0
                    print(f"No acceleration, coasting, speed: {speedX:.2f} km/h")
            else:  # Forward gears
                if self.accelerating and speedX < (max_speed - 1):  # Auto-accelerate
                    accel = 1.0
                    brake = 0.0
                    print(f"Auto-accelerating, speed: {speedX:.2f} km/h, target: {max_speed} km/h")
                elif abs(speedX - max_speed) <= 2:  # Maintain max speed
                    accel = 0.3  # Increased to counteract drag
                    brake = 0.0
                    self.accelerating = False
                    print(f"Maintaining speed: {speedX:.2f} km/h")
                else:  # Coast
                    accel = 0.0
                    brake = 0.0
                    print(f"No acceleration, coasting, speed: {speedX:.2f} km/h")

        # Handle steering with smoothing
        if steer_trigger == 'left':
            self.steer_target = 0.7
            print(f"Steering left, target: {self.steer_target:.2f}")
        elif steer_trigger == 'right':
            self.steer_target = -0.7
            print(f"Steering right, target: {self.steer_target:.2f}")
        else:
            self.steer_target = np.clip(self.model.predict(state_vector)[0], -0.7, 0.7)
            print(f"No steer input, model steer: {self.steer_target:.2f}")

        self.steer += (self.steer_target - self.steer) * 0.2
        self.steer = np.clip(self.steer, -0.7, 0.7)
        print(f"Current steer: {self.steer:.2f}")

        self.control.setSteer(self.steer)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.control.setClutch(0.0)
        self.control.setFocus(0)
        self.control.setMeta(0)

        # Save data for training
        reward = self.compute_reward()
        data_entry = self.prepare_data_entry(state_vector, self.steer, accel, brake, reward)
        self.data_buffer.append(data_entry)
        if len(self.data_buffer) >= 100:
            self.save_data()

        return self.control.toMsg()

    def get_state_vector(self):
        track = self.state.getTrack() or [0.0] * 19
        opponents = self.state.getOpponents() or [200.0] * 36
        angle = self.state.getAngle() or 0.0
        speedX = self.state.getSpeedX() or 0.0
        trackPos = self.state.getTrackPos() or 0.0
        rpm = self.state.getRpm() or 0.0

        track = [min(t / 200.0, 1.0) for t in track]
        opponents = [min(o / 200.0, 1.0) for o in opponents]
        angle = angle / np.pi
        speedX = speedX / 300.0
        trackPos = np.clip(trackPos, -1.0, 1.0)
        rpm = rpm / 10000.0

        return np.array(track + opponents + [angle, speedX, trackPos, rpm])

    def compute_reward(self):
        speedX = self.state.getSpeedX() or 0.0
        trackPos = self.state.getTrackPos() or 0.0
        angle = self.state.getAngle() or 0.0
        opponents = self.state.getOpponents() or [200.0] * 36

        reward = speedX / 300.0
        reward -= abs(trackPos) * 0.5
        reward -= abs(angle / np.pi) * 0.3
        if any(o < 5.0 for o in opponents):
            reward -= 0.5
        return reward

    def prepare_data_entry(self, state_vector, steer, accel, brake, reward):
        timestamp = self.state.getCurLapTime() or 0.0
        angle = self.state.getAngle() or 0.0
        curLapTime = self.state.getCurLapTime() or 0.0
        damage = self.state.getDamage() or 0.0
        distFromStart = self.state.getDistFromStart() or 0.0
        distRaced = self.state.getDistRaced() or 0.0
        focus = self.state.focus or [-1.0] * 5
        fuel = self.state.getFuel() or 0.0
        gear = self.current_gear
        lastLapTime = self.state.lastLapTime or 0.0
        opponents = self.state.getOpponents() or [200.0] * 36
        racePos = self.state.getRacePos() or 0
        rpm = self.state.getRpm() or 0.0
        speedX = self.state.getSpeedX() or 0.0
        speedY = self.state.getSpeedY() or 0.0
        speedZ = self.state.getSpeedZ() or 0.0
        track = self.state.getTrack() or [0.0] * 19
        trackPos = self.state.getTrackPos() or 0.0
        wheelSpinVel = self.state.wheelSpinVel or [0.0] * 4
        z = self.state.getZ() or 0.0

        data = {
            'timestamp': timestamp,
            'angle': angle,
            'curLapTime': curLapTime,
            'damage': damage,
            'distFromStart': distFromStart,
            'distRaced': distRaced,
            'fuel': fuel,
            'gear': gear,
            'lastLapTime': lastLapTime,
            'racePos': racePos,
            'rpm': rpm,
            'speedX': speedX,
            'speedY': speedY,
            'speedZ': speedZ,
            'trackPos': trackPos,
            'z': z,
            'steer': steer,
            'accel': accel,
            'brake': brake,
            'reward': reward
        }
        for i, val in enumerate(focus):
            data[f'focus_{i}'] = val
        for i, val in enumerate(opponents):
            data[f'opponents_{i}'] = val
        for i, val in enumerate(track):
            data[f'track_{i}'] = val
        for i, val in enumerate(wheelSpinVel):
            data[f'wheelSpinVel_{i}'] = val
        return data

    def save_data(self):
        fieldnames = [
            'timestamp', 'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced',
            'fuel', 'gear', 'lastLapTime', 'racePos', 'rpm', 'speedX', 'speedY', 'speedZ',
            'trackPos', 'z'
        ]
        fieldnames.extend([f'focus_{i}' for i in range(5)])
        fieldnames.extend([f'opponents_{i}' for i in range(36)])
        fieldnames.extend([f'track_{i}' for i in range(19)])
        fieldnames.extend([f'wheelSpinVel_{i}' for i in range(4)])
        fieldnames.extend(['steer', 'accel', 'brake', 'reward'])

        mode = 'w' if not self.csv_initialized else 'a'
        with open('telemetry_log.csv', mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            writer.writerows(self.data_buffer)
        self.data_buffer = []

    def train_model(self):
        self.save_data()
        self.model.train()

    def onShutDown(self):
        self.save_data()
        self.model.save()

    def onRestart(self):
        self.current_gear = 1
        self.data_buffer = []
        self.accelerating = False
        self.steer = 0.0
        self.steer_target = 0.0