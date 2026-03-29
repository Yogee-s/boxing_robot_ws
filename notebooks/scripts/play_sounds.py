"""Play all sound effects with descriptions."""
import os
from IPython.display import Audio, display, HTML

sounds_dir = 'src/boxbunny_gui/assets/sounds'

sound_descriptions = {
    'bell_start.wav':         'Rings at the start of each round',
    'bell_end.wav':           'Rings when the round ends',
    'button_click.wav':       'Plays on any GUI button press',
    'coach_notification.wav': 'Plays when the AI coach has feedback',
    'countdown_beep.wav':     'Short beep during 3-2-1 countdown',
    'countdown_go.wav':       'Plays on "GO!" after countdown',
    'impact.wav':             'Triggered on confirmed punch impact',
    'reaction_stimulus.wav':  'Stimulus flash sound in reaction drill',
    'rest_start.wav':         'Signals the beginning of a rest period',
    'session_complete.wav':   'Celebratory sound when session finishes',
}

print("=== Full Sound Suite Playback ===")
print("Press the play button on each widget to hear the sound.\n")

for f in sorted(os.listdir(sounds_dir)):
    if not f.endswith('.wav'):
        continue
    full_path = os.path.join(sounds_dir, f)
    label = f.replace('.wav', '').replace('_', ' ').title()
    desc = sound_descriptions.get(f, '')
    display(HTML(
        f'<div style="background:#1A1A1A;padding:8px 12px;'
        f'border-radius:8px;margin:4px 0;font-family:monospace;'
        f'color:#E0E0E0">'
        f'<b style="color:#00E676">{label}</b>'
        f'<span style="color:#9E9E9E;margin-left:12px">{desc}</span>'
        f'</div>'
    ))
    display(Audio(full_path, autoplay=False))

print("\nAll sounds loaded. Click play on each to test audio output.")
