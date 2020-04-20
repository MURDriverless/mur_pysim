import tkinter as tk


class MouseController:
    root = tk.Tk()
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    SCREEN_HEIGHT_HALF = SCREEN_HEIGHT / 2
    SCREEN_WIDTH_HALF = SCREEN_WIDTH / 2

    @classmethod
    def position(cls):
        """
        Gets the current mouse position.

        x = pixel distance from centre side of screen
        y_pos = pixel distance upwards of screen centre line
        y_neg = pixel distance downwards from screen centre line

        RETURNS:
            - float: x,
            - float: y_pos,
            - float: y_neg
        """
        x_raw = cls.root.winfo_pointerx()
        y_raw = cls.root.winfo_pointery()

        x = (x_raw - cls.SCREEN_WIDTH_HALF) / 1000

        if cls.SCREEN_HEIGHT_HALF - y_raw <= 0:
            y_neg = (y_raw - cls.SCREEN_HEIGHT_HALF) / 1000
            y_pos = 0

        else:
            y_neg = 0
            y_pos = (cls.SCREEN_HEIGHT_HALF - y_raw) / 1000

        return x, y_pos, y_neg

    @classmethod
    def reset(cls):
        """ Resets the mouse position to centre """
        #pyautogui.moveTo(cls.SCREEN_WIDTH / 2, cls.SCREEN_HEIGHT / 2)
        pass


