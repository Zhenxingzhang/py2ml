import numpy as np
import sys


ship_B = ["B", "B", "B", "B"]
ship_C = ["C", "C", "C"]
ship_D = ["D", "D"]
ship_F = ["F"]

class Ship:
    def __init__(self, flag, length, px, py, direction):
        self.flag = flag
        self.length = length
        self.positionX = px
        self.positionY = py
        self.direction = direction

playerA_ships = [Ship("B", 4, 2, 2, "right"), Ship("C", 3, 4, 2, "down"), Ship("C", 3, 4, 5, "right"), Ship("D", 2, 6, 4, "right")]
playerB_ships = []


def initial_filed(field_size):
    field = np.empty((field_size, field_size), dtype='S2')
    field[:]= "-"
    field[0, :] = ["  ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    field[:, 0] = ["  ", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", "11", "12", "13", "14", "15", "16"]
    return field

def place_ship(field, ship):
    for l in range(ship.length):
        if ship.direction == 'right':
            px = ship.positionX
            py = ship.positionY+l
            # field[ship.positionX][ship.positionY+l]=ship.flag
            # Do the thing
        elif ship.direction == 'left':
            px = ship.positionX
            py = ship.positionY-l
            # field[ship.positionX][ship.positionY-l]=ship.flag
            # Do the other thing
        elif ship.direction == 'up':
            px = ship.positionX-l
            py = ship.positionY
            # field[ship.positionX-l][ship.positionY]=ship.flag

        elif ship.direction == 'down':
            px = ship.positionX+l
            py = ship.positionY

        # validate the position in field
        if field[px][py] != '-' or px < 1 or px > field_size or py < 1 or py > field_size:
            print "Problem: Two ships occupied the same location in field: %2d, %2d"%(px,py)
            print_field(field)
            return False

        field[px][py]=ship.flag

def remove_ship(field, ship):
    for l in range(ship.length):
        if ship.direction == 'right':
            px = ship.positionX
            py = ship.positionY+l
            # field[ship.positionX][ship.positionY+l]=ship.flag
            # Do the thing
        elif ship.direction == 'left':
            px = ship.positionX
            py = ship.positionY-l
            # field[ship.positionX][ship.positionY-l]=ship.flag
            # Do the other thing
        elif ship.direction == 'up':
            px = ship.positionX-l
            py = ship.positionY
            # field[ship.positionX-l][ship.positionY]=ship.flag

        elif ship.direction == 'down':
            px = ship.positionX+l
            py = ship.positionY

        # # validate the position in field
        # if field[px][py] != '-' or px < 1 or px > field_size or py < 1 or py > field_size:
        #     print "Problem: Two ships occupied the same location in field: %2d, %2d"%(px,py)
        #     print_field(field)
        #     return False

        field[px][py]='-'

def initial_ships(field, ships):
    for ship in ships:
        place_ship(field,ship)

def refresh_field(field):
    initial_filed(field)
    initial_ships(field, playerA_ships)
    initial_ships(field, playerB_ships)

def move_ship(field, ship, direction):
    # if(ship.direction == "right" or ship.direction == "left"):
    # firstly, check the direction with ship direction
    if (direction == 'right' or direction == 'left') and (ship.direction == 'up' or ship.direction == 'down'):
        print "Problem: the ship can not move horizontally"
        return False

    if (direction == 'up' or direction == 'down') and (ship.direction == 'left' or ship.direction == 'right'):
        print "Problem: the ship can not move vertically"

    if direction == 'right':
        px = ship.positionX
        py = ship.positionY+1
        tail = py + ship.length
        # field[ship.positionX][ship.positionY+l]=ship.flag
        # Do the thing
    elif direction == 'left':
        px = ship.positionX
        py = ship.positionY-1
        tail = py + ship.length
        # field[ship.positionX][ship.positionY-l]=ship.flag
        # Do the other thing
    elif direction == 'up':
        px = ship.positionX-1
        py = ship.positionY
        tail = px + ship.length
        # field[ship.positionX-l][ship.positionY]=ship.flag

    elif direction == 'down':
        px = ship.positionX+1
        py = ship.positionY
        tail = px + ship.length

    # validate the position in field
    if px < 1 or px > field_size or py < 1 or py > field_size or tail > field_size or tail < 0:
        print "Problem: The ship will be outside of field, so the movement is denied."
        return False

    if field[px][py] != '-':
        print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(px,py)
        print_field(field)
        return False

    remove_ship(field, ship)
    ship.positionX = px
    ship.positionY = py
    place_ship(field, ship)

def fire_shot(field, positionX, positionY):
    if positionX < 1 or positionX > field_size or positionY < 1 or positionY > field_size :
        print "Problem: The location (%2d, %2d) is not in the field, could not fire the shot: "%(positionX, positionY)
        return False
    
    print ""

def print_field(field):
    for row in field:
        print " ".join(row)

field_size = 17
field = initial_filed(field_size)

initial_ships(field, playerA_ships)
print_field(field)

move_ship(field, playerA_ships[0], "left")
# refresh_field(field)
print_field(field)
#
move_ship(field, playerA_ships[1], "up")
# refresh_field(field)
print_field(field)

move_ship(field, playerA_ships[1], "up")
# refresh_field(field)
print_field(field)