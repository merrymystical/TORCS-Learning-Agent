from pynput import keyboard
import time

def on_press(key):
    try:
        if key.char == 'h':
            with open('gear_command.txt', 'w') as f:
                f.write('H')
            print("Gear up triggered (H pressed)")
        elif key.char == 'r':
            with open('gear_command.txt', 'w') as f:
                f.write('R')
            print("Gear down triggered (R pressed)")
    except AttributeError:
        pass

# Set up the listener to capture global keyboard events
with keyboard.Listener(on_press=on_press) as listener:
    print("Keyboard listener started. Press 'H' to gear up or 'R' to gear down while the game is active. Press Ctrl+C to stop.")
    listener.join()