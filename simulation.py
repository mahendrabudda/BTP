"""
simulation.py
=============
Pygame traffic simulation, signal decisions driven by RLBridge.

Fix vs previous version
------------------------
The old control loop set `currentGreen = green_indices[0]` (only the
FIRST index of a {right,left} or {down,up} pair), but Vehicle.move()
gated free-flow movement on an exact index match against `currentGreen`
— so only one of the two lanes in a "green" pair actually ever got
unrestricted green in the vehicle physics, while the other silently
just queued. This version represents the signal state as a phase
(0 = horizontal green, 1 = vertical green) and checks *phase
membership*, so both lanes in the green pair behave correctly.

Control loop
------------
Every CHECK_INTERVAL seconds the bridge is asked for keep(0)/switch(1).
A switch triggers a DEFAULT_YELLOW-second yellow phase for the
outgoing pair before the new pair goes green — the only place signal
timing differs from the headless TrafficEnv (which abstracts the
yellow transition into a fixed reward penalty rather than simulating
it in wall-clock time).

Run:
    python simulation.py                 # loads models/dqn_checkpoint_final.pt, frozen policy
    python simulation.py --train-live    # experimental: fine-tune live (see rl_bridge.py)

Controls:
    Close window -> saves model checkpoint (if --train-live) and exits.
"""

import argparse
import random
import sys
import threading
import time

import pygame

from config import ENV_CONFIG
from rl_bridge import RLBridge

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train-live", action="store_true",
                     help="Experimental: fine-tune the network live during the demo "
                          "instead of running a frozen, trained policy.")
parser.add_argument("--model", type=str, default="models/dqn_checkpoint_final.pt")
args = parser.parse_args()

# ── signal timing (wall-clock) ────────────────────────────────────────────────
DEFAULT_YELLOW = 3
CHECK_INTERVAL = 1   # seconds per decision step -- matches one TrafficEnv step

# ── globals ───────────────────────────────────────────────────────────────────
currentPhase  = 0     # 0 = horizontal (right+left) green, 1 = vertical (down+up) green
currentYellow = 0

speeds = {"car": 2.25, "bus": 1.8, "truck": 1.8, "bike": 2.5}

PHASE_OF_DIR = {"right": 0, "left": 0, "down": 1, "up": 1}
PHASE_GROUPS = {0: ["right", "left"], 1: ["down", "up"]}

x = {
    "right": [0, 0, 0],
    "down":  [755, 727, 697],
    "left":  [1400, 1400, 1400],
    "up":    [602, 627, 657],
}
y = {
    "right": [348, 370, 398],
    "down":  [0, 0, 0],
    "left":  [498, 466, 436],
    "up":    [800, 800, 800],
}

vehicles = {
    "right": {0: [], 1: [], 2: [], "crossed": 0},
    "down":  {0: [], 1: [], 2: [], "crossed": 0},
    "left":  {0: [], 1: [], 2: [], "crossed": 0},
    "up":    {0: [], 1: [], 2: [], "crossed": 0},
}

vehicleTypes     = {0: "car", 1: "bus", 2: "truck", 3: "bike"}
directionNumbers = {0: "right", 1: "down", 2: "left", 3: "up"}

vehicleCrossedCount = {"car": 0, "bus": 0, "truck": 0, "bike": 0}
totalCrossed        = 0

signalCoods       = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods  = [(530, 210), (810, 210), (810, 550), (530, 550)]
stopLines         = {"right": 590, "down": 330, "left": 800, "up": 535}
defaultStop       = {"right": 580, "down": 320, "left": 810, "up": 545}
stoppingGap       = 15
movingGap         = 15

pygame.init()
simulation = pygame.sprite.Group()

bridge = RLBridge(
    model_path=args.model,
    online_learning=args.train_live,
    save_path=args.model,
    save_every=500,
)


# ── classes ───────────────────────────────────────────────────────────────────

class Vehicle(pygame.sprite.Sprite):

    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane             = lane
        self.vehicleClass     = vehicleClass
        self.speed            = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction        = direction
        self.x                = x[direction][lane]
        self.y                = y[direction][lane]
        self.crossed          = 0

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        try:
            path       = f"images/{direction}/{vehicleClass}.png"
            self.image = pygame.image.load(path)
        except pygame.error:
            self.image = pygame.Surface((30, 20))
            color_map  = {"car": (100,180,255), "bus": (255,160,60),
                          "truck": (220,80,80),  "bike": (80,210,80)}
            self.image.fill(color_map.get(vehicleClass, (200, 200, 200)))

        prev = vehicles[direction][lane]
        if len(prev) > 1 and prev[self.index - 1].crossed == 0:
            p = prev[self.index - 1]
            if direction == "right":
                self.stop = p.stop - p.image.get_rect().width  - stoppingGap
            elif direction == "left":
                self.stop = p.stop + p.image.get_rect().width  + stoppingGap
            elif direction == "down":
                self.stop = p.stop - p.image.get_rect().height - stoppingGap
            elif direction == "up":
                self.stop = p.stop + p.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        if direction == "right":
            x[direction][lane] -= self.image.get_rect().width  + stoppingGap
        elif direction == "left":
            x[direction][lane] += self.image.get_rect().width  + stoppingGap
        elif direction == "down":
            y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == "up":
            y[direction][lane] += self.image.get_rect().height + stoppingGap

        simulation.add(self)

    def move(self):
        d   = self.direction
        w   = self.image.get_rect().width
        h   = self.image.get_rect().height
        ln  = self.lane
        idx = self.index
        is_green = (PHASE_OF_DIR[d] == currentPhase and currentYellow == 0)

        if d == "right":
            if self.crossed == 0 and self.x + w > stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.x + w <= self.stop or self.crossed == 1 or is_green) and
                (idx == 0 or self.x + w < vehicles[d][ln][idx-1].x - movingGap)
            )
            if can_move:
                self.x += self.speed

        elif d == "down":
            if self.crossed == 0 and self.y + h > stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.y + h <= self.stop or self.crossed == 1 or is_green) and
                (idx == 0 or self.y + h < vehicles[d][ln][idx-1].y - movingGap)
            )
            if can_move:
                self.y += self.speed

        elif d == "left":
            if self.crossed == 0 and self.x < stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.x >= self.stop or self.crossed == 1 or is_green) and
                (idx == 0 or self.x > vehicles[d][ln][idx-1].x +
                 vehicles[d][ln][idx-1].image.get_rect().width + movingGap)
            )
            if can_move:
                self.x -= self.speed

        elif d == "up":
            if self.crossed == 0 and self.y < stopLines[d]:
                self._markCrossed()
            can_move = (
                (self.y >= self.stop or self.crossed == 1 or is_green) and
                (idx == 0 or self.y > vehicles[d][ln][idx-1].y +
                 vehicles[d][ln][idx-1].image.get_rect().height + movingGap)
            )
            if can_move:
                self.y -= self.speed

    def _markCrossed(self):
        global totalCrossed
        self.crossed                            = 1
        vehicles[self.direction]["crossed"]    += 1
        vehicleCrossedCount[self.vehicleClass] += 1
        totalCrossed                           += 1


# ── helpers ───────────────────────────────────────────────────────────────────

def getVehicleCount(direction):
    return sum(1 for lane in range(3) for v in vehicles[direction][lane]
               if v.crossed == 0)


# ── signal control loop ───────────────────────────────────────────────────────

def _run_yellow_phase(outgoing_phase):
    """Yellow for the outgoing pair; no vehicles in either pair get free-flow green."""
    global currentYellow
    currentYellow = 1
    time.sleep(DEFAULT_YELLOW)
    currentYellow = 0


def control_loop():
    global currentPhase

    while True:
        time.sleep(CHECK_INTERVAL)

        action = bridge.choose(vehicles, last_cycle_seconds=float(CHECK_INTERVAL))
        bridge.feedback(vehicles)   # no-op unless --train-live

        if action == 1:   # switch requested (and legal, per bridge's own masking)
            outgoing = currentPhase
            _run_yellow_phase(outgoing)
            currentPhase = 1 - currentPhase


# ── vehicle generator ─────────────────────────────────────────────────────────

def generateVehicles():
    while True:
        vehicle_type     = random.randint(0, 3)
        lane_number      = random.randint(1, 2)
        direction_number = random.choices([0, 1, 2, 3], weights=[25, 25, 25, 25])[0]
        Vehicle(
            lane_number,
            vehicleTypes[vehicle_type],
            direction_number,
            directionNumbers[direction_number],
        )
        time.sleep(random.uniform(1.5, 3.5))


# ── HUD drawing ───────────────────────────────────────────────────────────────

def drawVehicleCountPanel(screen, font, bold_font):
    px, py, pw, ph = 10, 620, 270, 170
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (255, 255, 255), (px, py, pw, ph), 2)
    screen.blit(bold_font.render("VEHICLES CROSSED", True, (255, 220, 0)),
                (px + 10, py + 8))
    pygame.draw.line(screen, (255, 255, 255), (px+5, py+32), (px+pw-5, py+32), 1)
    colors = {
        "car":   (100, 180, 255),
        "bus":   (255, 160,  60),
        "truck": (220,  80,  80),
        "bike":  ( 80, 210,  80),
    }
    for i, vt in enumerate(["car", "bus", "truck", "bike"]):
        txt = font.render(f"  {vt.capitalize()} : {vehicleCrossedCount[vt]}",
                          True, colors[vt])
        screen.blit(txt, (px + 10, py + 40 + i * 26))
    pygame.draw.line(screen, (255,255,255), (px+5, py+144), (px+pw-5, py+144), 1)
    screen.blit(bold_font.render(f"  TOTAL : {totalCrossed}", True, (255,255,255)),
                (px + 10, py + 148))


def drawDirectionCounts(screen, font, bold_font):
    positions  = {
        "right": (10, 450),
        "left":  (1150, 450),
        "down":  (560, 10),
        "up":    (560, 768),
    }
    dir_labels = {
        "right": "-> Waiting",
        "left":  "<- Waiting",
        "down":  "v  Waiting",
        "up":    "^  Waiting",
    }
    for direction, pos in positions.items():
        count = getVehicleCount(direction)
        color = (100,255,100) if count == 0 else (255,220,0) if count <= 4 else (255,80,80)
        txt   = bold_font.render(f"{dir_labels[direction]}: {count}", True, color)
        tw, th = txt.get_size()
        bg     = pygame.Surface((tw + 10, th + 6), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        screen.blit(bg,  (pos[0] - 5, pos[1] - 3))
        screen.blit(txt,  pos)


def drawRLHUD(screen, font, bold_font):
    m = bridge.metrics
    px, py, pw, ph = 1095, 530, 295, 280

    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((0, 20, 40, 215))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (0, 200, 255), (px, py, pw, ph), 2)

    screen.blit(bold_font.render("DQN AGENT", True, (0, 220, 255)), (px+10, py+8))
    pygame.draw.line(screen, (0,200,255), (px+5, py+28), (px+pw-5, py+28), 1)

    mode_txt   = "LIVE TRAINING" if m["online_learning"] else "FROZEN POLICY (eval)"
    mode_color = (255, 180, 0)   if m["online_learning"] else (0, 255, 120)
    screen.blit(font.render(mode_txt, True, mode_color), (px+10, py+34))

    action_lbl = "Horizontal (R+L)" if m["action"] == 0 else "Vertical   (D+U)"
    screen.blit(font.render(f"Phase: {action_lbl}", True, (180,200,255)), (px+10, py+54))

    pygame.draw.line(screen, (0,200,255), (px+5, py+72), (px+pw-5, py+72), 1)
    screen.blit(font.render("Q-values + Wait:", True, (180,180,180)), (px+10, py+76))

    qv         = m["q_values"]
    waits      = m["wait_steps"]
    best       = qv.index(max(qv))
    q_min      = min(qv)
    q_range    = max(qv) - q_min + 1e-9
    bar_labels = ["->R", "vD ", "<-L", "^U "]
    bar_colors = [(0,200,255), (255,160,60), (100,255,100), (200,100,255)]

    for i in range(4):
        bx      = px + 10
        by      = py + 94 + i * 36
        bar_len = int((qv[i] - q_min) / q_range * 90)
        col     = bar_colors[i]

        if i == best:
            pygame.draw.rect(screen, col,           (bx, by, max(bar_len, 4), 13))
            pygame.draw.rect(screen, (255,255,255), (bx, by, max(bar_len, 4), 13), 1)
        else:
            dark = (col[0]//3, col[1]//3, col[2]//3)
            pygame.draw.rect(screen, dark,          (bx, by, max(bar_len, 4), 13))

        marker    = " <" if i == best else ""
        txt_color = col if i == best else (140, 140, 140)
        screen.blit(
            font.render(f"{bar_labels[i]} Q:{qv[i]:.1f}{marker}", True, txt_color),
            (bx + 98, by),
        )

        wait_val = waits[i]
        wait_len = int(min(wait_val / 60.0, 1.0) * 90)
        wait_col = (255, 60, 60) if wait_val > 12 else \
                   (255,200,  0) if wait_val >  6 else (60, 200, 60)
        pygame.draw.rect(screen, (40,  40, 40), (bx, by+16, 90, 7))
        pygame.draw.rect(screen, wait_col,      (bx, by+16, max(wait_len, 2), 7))
        screen.blit(
            font.render(f"wait:{int(wait_val)}s", True, wait_col),
            (bx + 98, by + 14),
        )

    pygame.draw.line(screen, (0,200,255), (px+5, py+258), (px+pw-5, py+258), 1)
    screen.blit(
        font.render(
            f"Reward:{m['total_reward']:.0f}  Steps:{m['steps']}",
            True, (200, 200, 200),
        ),
        (px + 10, py + 262),
    )


# ── main ──────────────────────────────────────────────────────────────────────

thread1 = threading.Thread(target=control_loop, daemon=True)
thread1.start()

screen = pygame.display.set_mode((1400, 800))

try:
    background = pygame.image.load("images/intersection.png")
except pygame.error:
    background = pygame.Surface((1400, 800))
    background.fill((40, 40, 40))

try:
    redSignal    = pygame.image.load("images/signals/red.png")
    yellowSignal = pygame.image.load("images/signals/yellow.png")
    greenSignal  = pygame.image.load("images/signals/green.png")
except pygame.error:
    def _make_signal(color):
        s = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (20, 20), 18)
        return s
    redSignal    = _make_signal((220,  40,  40))
    yellowSignal = _make_signal((220, 180,  40))
    greenSignal  = _make_signal(( 40, 200,  40))

font      = pygame.font.Font(None, 26)
bold_font = pygame.font.Font(None, 28)

pygame.display.set_caption("Traffic Intersection — DQN Controlled")

thread2 = threading.Thread(target=generateVehicles, daemon=True)
thread2.start()

clock = pygame.time.Clock()
signal_indices = {0: [0, 2], 1: [1, 3]}   # for display purposes only

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if args.train_live:
                bridge.agent.save_model(args.model)
            pygame.quit()
            sys.exit()

    screen.blit(background, (0, 0))

    green_now = signal_indices[currentPhase]
    for i in range(4):
        if i in green_now:
            img = yellowSignal if currentYellow == 1 else greenSignal
        else:
            img = redSignal
        screen.blit(img, signalCoods[i])

    for vehicle in simulation:
        screen.blit(vehicle.image, [vehicle.x, vehicle.y])
        vehicle.move()

    drawVehicleCountPanel(screen, font, bold_font)
    drawDirectionCounts(screen, font, bold_font)
    drawRLHUD(screen, font, bold_font)

    pygame.display.update()
    clock.tick(60)
