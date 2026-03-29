Inter Font - Manual Installation Required
==========================================

The Inter font could not be downloaded automatically (no internet access).

To install manually:
1. Visit: https://github.com/rsms/inter/releases
2. Download the latest Inter release ZIP (e.g., Inter-4.0.zip)
3. Extract the following files from the ZIP:
   - Inter-Variable.ttf
   - Inter-Variable-Italic.ttf
4. Place them in this directory:
   src/boxbunny_gui/assets/fonts/

Alternatively, if you have the ZIP already:
   unzip -o Inter-4.0.zip "*.ttf" -d src/boxbunny_gui/assets/fonts/

The GUI will fall back to system sans-serif fonts if Inter is not available.
