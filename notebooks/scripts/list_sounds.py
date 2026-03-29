"""List all available sound effects with a sample playback."""
import os
import sys

sys.path.insert(0, 'src/boxbunny_gui')

sounds_dir = 'src/boxbunny_gui/assets/sounds'
print("=== Available Sound Effects ===")
print(f"Directory: {os.path.abspath(sounds_dir)}\n")

total_size = 0
count = 0
for f in sorted(os.listdir(sounds_dir)):
    if f.endswith('.wav'):
        full_path = os.path.join(sounds_dir, f)
        size = os.path.getsize(full_path)
        total_size += size
        count += 1
        label = f.replace('.wav', '').replace('_', ' ').title()
        print(f"  {label:.<30} {f:>25}  ({size:,} bytes)")

print(f"\n  Total: {count} files, {total_size:,} bytes")

from IPython.display import Audio, display
sample = os.path.join(sounds_dir, 'bell_start.wav')
print(f"\nPlaying sample: bell_start.wav")
display(Audio(sample, autoplay=True))
