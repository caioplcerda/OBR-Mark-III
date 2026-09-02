# OBR Mark III — Autonomous Rescue Robot

![OBR Mark III closing its gripper on a rescue ball](hardware/photos/robot-gripper-rescue-ball.jpg)

An autonomous robot built for the **Olimpíada Brasileira de Robótica (OBR) 2025**, Brazil's
national robotics olympiad. It follows a line, reads green intersection markers, avoids
obstacles, collects rescue balls with a rack-and-pinion gripper, and is tuned live from a
browser while it drives.

Every structural part was designed in Fusion 360 and 3D printed in PLA. The tyres are custom
silicone, cast in a printed two-part mould, because printed wheels would not hold the track.

---

## What it does

**Line following under a PID controller with optical encoder feedback.** Each wheel reports
its own speed, so the controller corrects for the two motors never being quite identical —
the usual reason a differential-drive robot drifts on a straight line.

**Adaptive look-ahead.** The vision system does not just find the line under the robot; it
scans forward along the expected path and feeds a predicted trajectory into the controller,
so the robot slows into corners instead of overshooting them.

**Green intersection markers.** OBR arenas mark junctions with green patches indicating which
branch to take. The vision pipeline segments them in HSV, decides the turn, and executes a
180° reversal when two markers appear at once.

**Rescue-ball retrieval.** A rack-and-pinion gripper driven by an MG996R closes on the ball
and lifts it into an onboard reservoir.

**Camera mounted upside down.** The physical layout forced a 180° camera rotation; the
software rotates every frame before processing rather than fighting it mechanically.

**Thick-line support.** Morphological operations and contour analysis handle track lines up
to 20 mm wide, which break naive centroid-following.

**Live tuning over the network.** A Flask + Socket.IO server streams the robot's own camera
view to a browser and exposes PID gains, HSV thresholds and speed limits as live controls.
Tuning happens while the robot drives, with no reflashing and no reboot.

---

## The robot

| | |
|---|---|
| ![Front](hardware/photos/robot-front.jpg) | ![Three-quarter](hardware/photos/robot-three-quarter.jpg) |
| Front — WS2812 indicator rings, ultrasonic sensor, silicone tyres | Reservoir loaded, gripper raised |

![Side view with voltmeters and power switch](hardware/photos/robot-side-instrumentation.jpg)

Four panel voltmeters report pack and rail voltages at a glance, next to the main power
switch and a printed ventilation grille — useful when you are debugging in a pit between
rounds and need to know the battery state without a laptop.

---

## Architecture

```
src/
├── main.py               State machine — orchestrates behaviour and mode transitions
├── vision.py             Camera pipeline: rotation, thresholding, line and marker detection
├── line_follower.py      PID control, adaptive look-ahead, error computation
├── hardware_control.py   Motor, encoder and servo abstraction over the TB6612FNG drivers
├── web_stream.py         Flask + Socket.IO live video and remote calibration
├── led_control.py        WS2812 status indicators
├── rcj2014_port.py       Scan and error logic ported from the RoboCup Junior 2014 C++ reference
└── index.html            Calibration interface
```

Roughly 1,200 lines of Python across the core modules.

---

## Bill of materials

### Compute and sensing
| Part | Qty |
|---|---|
| Raspberry Pi 5, 4 GB | 1 |
| Picamera3 Wide (mounted inverted) | 1 |
| Ultrasonic distance sensor | 1 |
| Optical encoder speed-sensor modules | 2 |

### Drive
| Part | Qty |
|---|---|
| DC motor, metal reduction gearbox, reversible | 2 |
| TB6612FNG dual H-bridge driver | 2 |
| Hex hub, 12 mm, for 8 mm shaft | 2 |
| Metal caster ball | 1 |
| Silicone tyres, cast in a printed mould | 2 |

### Actuators
| Part | Qty |
|---|---|
| TowerPro MG996R metal-gear servo (gripper) | 1 |
| SG90 9 g micro servo | 3 |
| Circular servo flange | 3 |

### Power
| Part | Qty |
|---|---|
| 18650 Li-Ion cell, 3.7 V (2550 mAh / Samsung 30Q 3000 mAh) | 8 |
| 4-cell 18650 holder, SMT PCB | 2 |
| BMS protection board, 5S 25 A 21 V | 1 |
| BMS protection board, 3S | 1 |
| MP1584 mini DC-DC step-down, 3 A | 5 |
| Mini-560 DC-DC step-down, 5 V | 1 |
| Adjustable current/voltage regulator | 2 |
| 3-digit digital voltmeter | 4 |
| 6A8 rectifier diode, 6 A 800 V | 2 |
| 1000 µF 16 V electrolytic capacitor | 1 |
| 100 pF 50 V ceramic capacitor | 10 |
| R13-507 16 mm momentary switch, 6 A | 1 |
| Dual 18650 charger | 1 |

### Indication and thermal
| Part | Qty |
|---|---|
| WS2812 addressable RGB module, 4-bit | 2 |
| WS2812 addressable RGB module, 7-bit | 1 |
| 40 × 40 × 10 mm fan | 2 |
| Self-adhesive heatsink | 5 |

### Mechanical
| Part | Qty |
|---|---|
| PLA filament — full chassis, gripper, mounts, wheel mould | — |
| M3 × 25 mm Philips screw | 50 |
| M3 nut | 30 |
| M2.5 × 6 mm Philips screw | 50 |
| M3 × 5 × 4.2 heat-set threaded insert | 1 kit |
| 6-way locking connector | 2 |

Threaded brass inserts were heat-set into the printed parts rather than tapping screws
directly into PLA, so the assembly survives repeated disassembly between competition rounds.

---

## CAD

**[`hardware/cad/obr-mark-iii.stl`](hardware/cad/obr-mark-iii.stl) — click to open it in
GitHub's interactive 3D viewer.** Rotate and zoom the assembly in the browser; nothing to
install.

Also committed: `obr-mark-iii.3mf`, print-ready with units and metadata preserved.
67,842 triangles.

### On the tyres

Printed PLA wheels slip on the OBR track surface, especially on inclines. Rather than buy
rubber wheels that would not fit the chassis geometry, we modelled a two-part mould, printed
it, and cast the tyres in silicone. The result grips the track and can be recast in minutes
if one tears mid-competition.

---

## Running it

The modules import each other flatly, and `web_stream.py` serves `index.html` from its own
directory, so run from inside `src/`:

```bash
pip install -r requirements.txt
cd src
python main.py
```

The calibration interface is then at `http://<robot-ip>:5000`.

Hardware-dependent imports (`RPi.GPIO`, `picamera2`, `libcamera`) are guarded, so the vision
and control modules can be exercised on a development machine without the robot attached.

---

## Repository layout

| Path | |
|---|---|
| `src/` | Flight code — the modules above |
| `hardware/cad/` | Fusion 360 exports: STL and 3MF |
| `hardware/photos/` | Robot photographs |
| `experiments/` | Bring-up scripts kept from development: servo sweeps, LED and SPI tests, encoder checks, earlier vision attempts |
| `docs/` | Wiring and tuning notes |

`experiments/` is deliberately preserved. Getting three servos, an SPI LED strip and two
encoders working reliably took a lot of small scripts, and that iteration is part of the
project.

---

## Built by

Caio Lacerda — [github.com/caioplcerda](https://github.com/caioplcerda) ·
[linkedin.com/in/caio-lacerda](https://linkedin.com/in/caio-lacerda-61b7081b9)
