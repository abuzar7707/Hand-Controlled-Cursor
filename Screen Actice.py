import pyautogui as py
import time as t
import keyboard as key

pressed = 1
while True:
    if key.is_pressed('a'):
        while pressed != 0:
            py.press('space')
            t.sleep(0)

            if key.is_pressed('s'):
                pressed = 0
    t.sleep(2)
    if pressed == 0:
        end = input("Exit(Y / N): ")
        if end == "Y":
            break
