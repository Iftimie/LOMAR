from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import sys
import subprocess
import time


interface = "wlp2s0"


# You can add or change the functions to parse the properties of each AP (cell)
# below. They take one argument, the bunch of text describing one cell in iwlist
# scan and return a property of that cell.

def get_name(cell):
    return matching_line(cell, "ESSID:")[1:-1]


def get_quality(cell):
    quality = matching_line(cell, "Quality=").split()[0].split('/')
    return str(int(round(float(quality[0]) / float(quality[1]) * 100))).rjust(3) + " %"


def get_channel(cell):
    return matching_line(cell, "Channel:")


def get_signal_level(cell):
    # Signal level is on same line as Quality data so a bit of ugly
    # hacking needed...
    return matching_line(cell, "Quality=").split("Signal level=")[1]


def get_encryption(cell):
    enc = ""
    if matching_line(cell, "Encryption key:") == "off":
        enc = "Open"
    else:
        for line in cell:
            matching = match(line, "IE:")
            if matching != None:
                wpa = match(matching, "WPA Version ")
                if wpa != None:
                    enc = "WPA v." + wpa
        if enc == "":
            enc = "WEP"
    return enc


def get_address(cell):
    return matching_line(cell, "Address: ")


# Here's a dictionary of rules that will be applied to the description of each
# cell. The key will be the name of the column in the table. The value is a
# function defined above.

rules = {"Name": get_name,
         "Quality": get_quality,
         "Channel": get_channel,
         "Encryption": get_encryption,
         "Address": get_address,
         "Signal": get_signal_level
         }


# Here you can choose the way of sorting the table. sortby should be a key of
# the dictionary rules.

def sort_cells(cells):
    sortby = "Quality"
    reverse = True


# You can choose which columns to display here, and most importantly in what order. Of
# course, they must exist as keys in the dict rules.

columns = ["Name", "Address", "Quality", "Signal", "Channel", "Encryption"]


# Below here goes the boring stuff. You shouldn't have to edit anything below
# this point

def matching_line(lines, keyword):
    """Returns the first matching line in a list of lines. See match()"""
    for line in lines:
        matching = match(line, keyword)
        if matching != None:
            return matching
    return None


def match(line, keyword):
    """If the first part of line (modulo blanks) matches keyword,
    returns the end of that line. Otherwise returns None"""
    line = line.lstrip()
    length = len(keyword)
    if line[:length] == keyword:
        return line[length:]
    else:
        return None


def parse_cell(cell):
    """Applies the rules to the bunch of text describing a cell and returns the
    corresponding dictionary"""
    parsed_cell = {}
    for key in rules:
        rule = rules[key]
        parsed_cell.update({key: rule(cell)})
    return parsed_cell


def print_table(x,y,table):
    widths = map(max, map(lambda l: map(len, l), zip(*table)))  # functional magic

    justified_table = []
    print (len(table))
    if(len(table)<=1):
        print("only one detected")
        return -1
    file = open("logfile.txt", "a")

    file.write("%.3f,%.3f"%(x,y))
    for line in table:
        file.write(",%s,%s"%(line[1],str(line[3]).replace(" dBm","")))
        print(line[1]," ",str(line[3]).replace(" dBm",""))
    file.write("\n")
    file.close()

def print_cells(x,y,cells):
    #table = [columns]
    table = []
    for cell in cells:
        cell_properties = []
        for column in columns:
            cell_properties.append(cell[column])
        table.append(cell_properties)
    return print_table(x,y,table)


def saveMACandRSS(x,y):
    """Pretty prints the output of iwlist scan into a table"""

    for j in range (5):
        cells = [[]]
        parsed_cells = []

        proc = subprocess.Popen(["iwlist", interface, "scan"], stdout=subprocess.PIPE, universal_newlines=True)
        out, err = proc.communicate()

        for line in out.split("\n"):
            cell_line = match(line, "Cell ")
            if cell_line != None:
                cells.append([])
                line = cell_line[-27:]
            cells[-1].append(line.rstrip())

        cells = cells[1:]

        for cell in cells:
            parsed_cells.append(parse_cell(cell))

        sort_cells(parsed_cells)

        if(print_cells(x,y,parsed_cells)==-1):
            return
        time.sleep(2)


xs = []
ys = []
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        print("ok")

    def __call__(self, event):
        print('click', event.xdata,event.ydata )
        if event.inaxes != self.line.axes: return
        xs.append(event.xdata)
        ys.append(event.ydata)
        saveMACandRSS(event.xdata, event.ydata)
        ax.scatter(xs,ys)
        ax.figure.canvas.draw()


image = mpimg.imread("hartacameracamin.png")
fig = plt.figure()

plt.grid(True)

plt.imshow(image)
ax = fig.add_subplot(111)
ax=plt.gca()                            # get the axis
ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
#ax.xaxis.tick_top()                     # and move the X-Axis
#ax.yaxis.tick_left()

ax.set_title('click to build line segments')
line = ax.scatter([],[])  # empty line
linebuilder = LineBuilder(line)

plt.show()

# !/usr/bin/env python
#
# iwlistparse.py
# Hugo Chargois - 17 jan. 2010 - v.0.1
# Parses the output of iwlist scan into a table

