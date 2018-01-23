"""
Game: Maze

The explorer (Red rectangle) keep away from hell (Black rectangles) and find the way to the terminal(yellow bin circle)

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].


The relevant RL script Q_Brain_Maze is located in directory Q_learning

Reference: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

""" Base config """
unit = 80  # the pixels (how big)
maze_height = 5  # Height
maze_width = 5  # Width
height_pix = maze_height * unit
width_pix = maze_width * unit


"""
Maze Game Environment
"""


class Maze_env(tk.Tk, object):

    def __init__(self):
        super(Maze_env, self).__init__()
        self.canvas = None
        self.title("Maze Game")  # the window's title
        self.geometry('{0}x{1}'.format(width_pix*10, height_pix*10))  # The window's size: (4*40) * (4*40) pixels
        self.actionSet = ("up", "down", "left", "right")  # The actions can be taken in this game
        self.create_env()

    def create_env(self):
        self.canvas = tk.Canvas(self, bg="white", width=width_pix, height=height_pix)

        """ create grids """
        for col in range(0, width_pix, unit):
            """
            Draw the vertical lines first (画竖线)
            """
            x0, y0 = col, 0  # From (x0,0)
            x1, y1 = col, height_pix  # To (x1,height_pix)
            self.canvas.create_line(x0,y0,x1,y1)

        for row in range(0, height_pix, unit):
            """
            Draw the horizontal lines (画横线)
            """
            x0,y0 = 0, row
            x1,y1 = width_pix, row
            self.canvas.create_line(x0,y0,x1,y1)

        """
        Hells
        """
        h1_x0,h1_y0 = unit * 1, unit * 3
        h1_x1,h1_y1 = unit * 2, unit * 4
        self.hell1 = self.canvas.create_rectangle(h1_x0,h1_y0,h1_x1,h1_y1,fill="black")

        h2_x0,h2_y0 = unit * 2, unit * 2
        h2_x1,h2_y1 = unit * 3, unit * 3
        self.hell2 = self.canvas.create_rectangle(h2_x0,h2_y0,h2_x1,h2_y1,fill="black")

        h3_x0,h3_y0 = unit * 2, unit * 4
        h3_x1,h3_y1 = unit * 3, unit * 5
        self.hell3 = self.canvas.create_rectangle(h3_x0, h3_y0, h3_x1, h3_y1, fill="black")

        """
        Terminal
        """
        t1_x0, t1_y0 = unit * 2 + 5, unit * 3 + 5
        t1_x1,  t1_y1 = unit * 3 - 5, unit * 4 - 5
        self.terminal = self.canvas.create_oval(t1_x0, t1_y0, t1_x1, t1_y1, fill="yellow")

        """
        Explorer (Randomly reset)
        """
        exp_loaction = self.reset_exp()
        #print(exp_loaction)
        """
        Pack all
        """
        self.canvas.pack()

    def reset_exp(self):
        """
        Explorer : randomly put into the game
        """
        while True:
            except_pos_list = [(unit * 1, unit * 3), (unit * 2, unit * 4),
                               (unit * 2, unit * 2), (unit * 3, unit * 3),
                               (unit * 3, unit * 5),
                               (unit * 2, unit * 3),
                               (unit * 3, unit * 4)]
            horizontal_random_pos = np.random.randint(1, maze_width)  # random horizontal position
            ver_random_pos = np.random.randint(1, maze_height)  # random vertical position

            hrz_x0 = unit * horizontal_random_pos
            ver_y0 = unit * ver_random_pos
            hrz_x1 = unit * (horizontal_random_pos + 1)
            ver_y1 = unit * (ver_random_pos + 1)

            e1_x0, e1_y0 = hrz_x0 + 5, ver_y0 + 5  # the point(x0,y0) of the explorer being a circle
            e1_x1, e1_y1 = hrz_x1 - 5, ver_y1 - 5  # the point(x1,y1) of the explorer being a circle

            if ((hrz_x0, ver_y0) in except_pos_list) and ((hrz_x1, ver_y1) in except_pos_list):
                continue  # avoiding the explorer is initially located at the hells or terminal
            else:
                # print("{0} / {1} / {2} / {3}".format(hrz_x0,ver_y0,hrz_x1,ver_y1))
                self.explorer = self.canvas.create_oval(e1_x0, e1_y0, e1_x1, e1_y1, fill="red")
                break

        return self.canvas.coords(self.explorer)

    def update_env(self,action):
        self.canvas.move(self.explorer,0,80)

    def feedback(self):
        pass
        #TODO





""" Testing """

if __name__ == '__main__':
    env = Maze_env()
    b = tk.Button(env, text='move', command=env.update_env(1)).pack()
    env.mainloop()



