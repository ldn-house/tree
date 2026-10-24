# Home Assistant

The tree joins Home Assistant through MQTT discovery (`firmware/app/ha.py`): one device, "Christmas Tree", with the light, settings, diagnostics and the takeover entities below. This folder holds YAML to import for things discovery can't set up: the F1 takeover blueprint, scripts for takeovers, and the changes other automations need to coexist with takeovers.

Entity IDs below assume the device is called "Christmas Tree" (`light.christmas_tree`, `binary_sensor.christmas_tree_takeover`). If you renamed it, or put it in an area that prefixes names, check the IDs in Settings > Devices.

## The light

`light.christmas_tree` is an RGB light (JSON schema):

- **On/off and brightness** work as before. Brightness 100% means "as bright as the 1.5 A power cap allows".
- **Effect** selects an effect, or `auto` for the auto cycle. Effects marked `AUTO = False` (`solid`, the `f1_*` ones) are in the list but never auto-cycled.
- **Colour** sets the selected effect's colour (its `color` param) when it has one, like `lighthouse` or `spiral_3d`. Otherwise it switches to the `solid` effect in that colour. The light reports the colour of the selected effect, or the `solid` colour when the effect has none.
- A command with both an effect and a colour, as a scene restore sends, selects the effect; the colour only sets that effect's colour, if it has one. So a scene snapshot of the tree restores the effect it was showing.

Each effect's colour params also appear as `text` entities ("Lighthouse color", ...) that take `#RRGGBB`.

## Takeovers

A takeover shows its own effect for a while, over whatever the tree is doing, then hands the tree back. The tree's own state (power, brightness, effect, params) is never changed by a takeover, so ending one restores it, and nothing about it is saved: a reboot ends every takeover.

Send a JSON command to the MQTT topic `xmas_tree/takeover/set` (not retained), or `POST /takeover` with the HTTP token:

```json
{"id": "f1", "effect": "f1_yellow", "color": [255, 255, 0], "brightness": 255, "priority": 50, "ttl_s": 180}
```

| Field | Meaning |
|---|---|
| `id` | Name of this takeover (lowercase letters, digits, `_`; default `default`). One takeover per id. |
| `effect` | Any registered effect, or `off` for a dark tree. |
| `params` | That effect's params; colours as `"#RRGGBB"`, `[r, g, b]` or `0xRRGGBB`. Unspecified params take the effect's defaults, not the tree's saved ones. |
| `color` | Shorthand for `params.color`. |
| `brightness` | 0-255. Default: the tree's brightness. |
| `speed` | 0.1-5, default 1.0 (the tree's speed setting doesn't apply). |
| `priority` | 0-100, default 50. The highest active takeover shows; on a tie, the most recent. At most 4 at once. |
| `ttl_s` | 1-3600, default 120. A dead man's switch: the takeover ends by itself unless it's refreshed, so a crashed Home Assistant never leaves the tree stuck. |
| `restart` | `true` plays the effect from its start again even if it's already showing. |

Other commands: `{"id": "f1", "action": "refresh", "ttl_s": 180}` extends a running takeover's TTL (and does nothing if it has ended), `{"id": "f1", "action": "end"}` ends one, and `{"action": "end_all"}` ends them all.

Sending the same command again only extends its TTL; new params are applied to the running effect without restarting it, and a new effect starts from its beginning.

**While a takeover shows:**

- The light reports what the tree shows: on, with the takeover's effect and brightness. The tree shows a takeover even when it's off, and goes back to off afterwards.
- Turning the light **off**, choosing an **effect** or a **colour**, or pressing **End takeover** ends every takeover and *dismisses* it: commands for that id are refused until you turn the light on again, the automation sends `end` for it, or 4 hours pass. So turning the tree off during a race keeps it off for the rest of that session.
- Changing the **brightness** dims or brightens the takeover too.
- Other settings (speed, params, floor, auto-cycle) change the tree's own state and show once the takeover ends.
- Calibration outranks takeovers. A takeover sent during calibration waits underneath, its TTL running.

**Entities:** `binary_sensor.christmas_tree_takeover` is on while a takeover shows, with the showing one's `id`, `effect`, `params`, `brightness`, `priority` and remaining `ttl_s` as attributes. The "End takeover" button ends them all. The Running effect sensor's `mode` attribute is `takeover` (and `takeover` names the id). `GET /` → `takeover` and `GET /takeover` give the full list.

[`scripts.yaml`](scripts.yaml) has `script.christmas_tree_takeover` and `script.christmas_tree_end_takeover` for calling this from your own automations.

## F1 takeover

[`blueprints/christmas_tree_f1_takeover.yaml`](blueprints/christmas_tree_f1_takeover.yaml) hands the tree to Formula 1 using [F1 Sensor](https://github.com/Nicxe/f1_sensor) 6.0. Import it (Settings > Automations & scenes > Blueprints > Import, with the file's raw GitHub URL), then create an automation from it and pick the TV media players and the tree's Takeover sensor.

What it does, only while the TV is on and the F1 calendar's current session is one of the chosen ones (Practice 3, qualifying, sprint qualifying, sprint and race by default):

| When | The tree shows |
|---|---|
| Formation lap (Race and Sprint) | Dark |
| Lights-out, or the session clock starting | `f1_start`: a green sweep up the tree, then a steady green |
| Track status CLEAR / YELLOW / VSC / SC | `f1_start` green / `f1_yellow` / amber `lighthouse` / faster orange `lighthouse` |
| RED (session live or suspended) | `f1_red`, a slow red pulse |
| Chequered flag (session finished) | `f1_chequered` for 15 s, `f1_winner` in P1's team colour for 60 s, `spiral_3d` in the team colour and white for 15 s, then back to what the tree was doing. After practice, only `f1_chequered` |
| TV off for a minute | Back to what the tree was doing |

Flags (except red) only change while the session is live and its clock is running. When the clock starts again (lights-out, a restart after a red flag, the next part of qualifying) the current flag is applied again, and the green sweep replays. Otherwise the tree keeps what it last showed, and the automation refreshes it every minute (TTL 3 minutes).

**Lights-out timing.** F1 Sensor turns `sensor.f1_session_status` to `live` at lights-out itself, not at the formation lap, so only lights-out can be timed exactly. Its one pre-start signal, `binary_sensor.f1_formation_start`, needs F1TV authentication and isn't delayed with the live delay. So the darkness starts at the session's scheduled start in the F1 calendar, which is the formation-lap time, plus F1 Sensor's live delay (`number.f1_live_delay`). Race-control messages containing "FORMATION LAP" (including "EXTRA FORMATION LAP", and the announcements of a delayed start) start it again. The darkness lasts until lights-out, 15 minutes at most.

Replay mode (`sensor.f1_replay_status`) isn't used.

## Coexisting with other automations

Automations that turn the tree off or restore a scene would end the F1 takeover and dismiss it for the rest of the session. Make them stand down while a takeover shows:

```yaml
conditions:
  - condition: state
    entity_id: binary_sensor.christmas_tree_takeover
    state: "off"
```

For an automation that turns the tree off while something is watched on the TV and restores a scene afterwards ("film mode"):

1. Add the condition above to both the branch that turns the tree off and the branch that restores the scene. While a takeover shows, the light reports on, so a restore guarded by "the tree is off" already skips; the condition makes it explicit.
2. Add a trigger for the end of a takeover, so watching that carries on afterwards still turns the tree off:

   ```yaml
   triggers:
     - trigger: state
       id: takeover_ended
       entity_id: binary_sensor.christmas_tree_takeover
       to: "off"
       for:
         seconds: 10
   ```

   and handle `takeover_ended` like "watching started", with the same "something is being watched" check as a condition, since this trigger doesn't imply it.

[`examples/tv_on_condition.yaml`](examples/tv_on_condition.yaml) is a condition for gating other F1 automations (room lights, say) on the TV being on, the same test the blueprint uses.
