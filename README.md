# OBR 2025 — Autonomous Line-Following Robot

<!-- HERO: replace with hardware/photos/robot-hero.jpg once photos are in -->

An autonomous robot built for the **Olimpíada Brasileira de Robótica (OBR) 2025** — Brazil's
national robotics olympiad. It follows a line, reads green intersection markers, avoids
obstacles, and is tuned live from a browser while it drives.

Chassis, wheels and mounts were designed in Fusion 360 and 3D printed. The tyres are custom
silicone, cast in a printed mould, because the stock printed wheels could not hold the track.

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

**Camera mounted upside down.** The physical layout forced a 180° camera rotation; the
software rotates every frame before processing rather than fighting it mechanically.

**Thick-line support.** Morphological operations and contour analysis handle track lines up
to 20 mm wide, which break naive centroid-following.

**Live tuning over the network.** A Flask + Socket.IO server streams the robot's own camera
view to a browser and exposes PID gains, HSV thresholds and speed limits as live controls.
Tuning happens while the robot drives, with no reflashing and no reboot.

---

## Architecture

```
src/
├── main.py               State machine — orchestrates behaviour and mode transitions
├── vision.py             Camera pipeline: rotation, thresholding, line and marker detection
├── line_follower.py      PID control, adaptive look-ahead, error computation
├── hardware_control.py   Motor, encoder and servo abstraction over the TB6612FNG driver
├── web_stream.py         Flask + Socket.IO live video and remote calibration
├── led_control.py        Status LEDs
├── rcj2014_port.py       Scan and error logic ported from the RoboCup Junior 2014 C++ reference
└── index.html            Calibration interface
```

Roughly 1,200 lines of Python across the core modules.

---

## Hardware

| | |
|---|---|
| Compute | Raspberry Pi 5, 4 GB |
| Camera | Picamera3 Wide, mounted inverted |
| Motor driver | TB6612FNG dual H-bridge |
| Drive | 2 × DC motors with optical encoders |
| Actuators | 3 × servos — gripper and rescue-ball reservoir |
| Chassis | 3D printed PLA, M3 bolted assembly |
| Tyres | Silicone, cast in a 3D-printed mould |

### On the tyres

Printed PLA wheels slip on the OBR track surface, especially on inclines. Rather than buy
rubber wheels that would not fit the chassis geometry, we modelled a two-part mould, printed
it, and cast the tyres in silicone. The result grips the track and can be recast in minutes
if one tears mid-competition.

### CAD

**[`hardware/cad/obr-mark-iii.stl`](hardware/cad/obr-mark-iii.stl) — click to open it in
GitHub's interactive 3D viewer.** Rotate and zoom the full assembly in the browser; nothing
to install.

Also committed: `obr-mark-iii.3mf`, print-ready with units and metadata preserved.
Model bounding box 272 × 363 × 190 mm, 67,842 triangles.

<!-- Fusion 360 public share link goes here -->

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
| `hardware/cad/` | Fusion 360 exports: STL and renders |
| `hardware/photos/` | Robot, electronics, arena |
| `experiments/` | Bring-up scripts kept from development: servo sweeps, LED and SPI tests, encoder checks, earlier vision attempts |
| `docs/` | Wiring and tuning notes |

`experiments/` is deliberately preserved. Getting three servos, an SPI LED strip and two
encoders working reliably took a lot of small scripts, and that iteration is part of the
project.

---

## Built by

Caio Lacerda — [github.com/caioplcerda](https://github.com/caioplcerda) ·
[linkedin.com/in/caio-lacerda](https://linkedin.com/in/caio-lacerda-61b7081b9)
