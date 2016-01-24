import numpy as np

ship_B = ["B", "B", "B", "B"]
ship_C = ["C", "C", "C"]
ship_D = ["D", "D"]
ship_F = ["F"]

ship_b = ["b", "b", "b", "b"]
ship_c = ["c", "c", "c"]
ship_d = ["d", "d"]
ship_f = ["f"]

class Ship:
    def __init__(self, flag, length, px, py, direction):
        self.flag = flag
        self.length = length
        self.positionX = px
        self.positionY = py
        self.direction = direction

playerA_ships = [Ship(ship_B, 4, 2, 2, "horizontal"), Ship(ship_C, 3, 4, 2, "vertical"), Ship(ship_C, 3, 4, 5, "horizontal"), Ship(ship_D, 2, 6, 4, "horizontal"), Ship(ship_D, 2, 1, 7, "vertical"), Ship(ship_D, 2, 8, 7, "horizontal")]
playerB_ships = [Ship(ship_b, 4, 7, 12, "vertical"), Ship(ship_c, 3, 2, 13, "horizontal"), Ship(ship_c, 3, 12, 12, "horizontal"), Ship(ship_d, 2, 16, 9, "horizontal"), Ship(ship_f, 1, 4, 11, "vertical"), Ship(ship_f, 1, 5, 15, "horizontal")]


def initial_filed(field_size):
    field = np.empty((field_size, field_size), dtype='S2')
    field[:]= "-"
    field[0, :] = ["  ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    field[:, 0] = ["  ", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", "11", "12", "13", "14", "15", "16"]
    return field

def place_ship(field, ship):
    for l in range(ship.length):
        if ship.direction == 'horizontal':
            px = ship.positionX
            py = ship.positionY+l

        elif ship.direction == 'vertical':
            px = ship.positionX+l
            py = ship.positionY

        # validate the position in field
        if field[px][py] != '-' or px < 1 or px > field_size or py < 1 or py > field_size:
            print "Problem: Two ships occupied the same location in field: %2d, %2d"%(px,py)
            return False

        field[px][py]=ship.flag[l]

def remove_ship(field, ship):
    for l in range(ship.length):
        if ship.direction == 'horizontal':
            px = ship.positionX
            py = ship.positionY+l

        elif ship.direction == 'vertical':
            px = ship.positionX+l
            py = ship.positionY

        # # validate the position in field
        # if field[px][py] != '-' or px < 1 or px > field_size or py < 1 or py > field_size:
        #     print "Problem: Two ships occupied the same location in field: %2d, %2d"%(px,py)
        #     print_field(field)
        #     return False
        ship.flag[l]=field[px][py]
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
    if (direction == 'right' or direction == 'left') and (ship.direction == 'vertical'):
        print "Problem: the ship can not move horizontally"
        return False

    if (direction == 'up' or direction == 'down') and (ship.direction == 'horizontal'):
        print "Problem: the ship can not move vertically"

    if direction == 'right':
        xHead = ship.positionX
        yHead = ship.positionY+1
        xTail = ship.positionX
        yTail = yHead + ship.length-1

        if field[xTail][yTail] != '-':
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xTail,yTail)
            return False

    elif direction == 'left':
        xHead = ship.positionX
        yHead = ship.positionY-1
        xTail = ship.positionX
        yTail = yHead + ship.length-1

        if field[xHead][yHead] != '-':
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xHead,yHead)
            return False

    elif direction == 'up':
        xHead = ship.positionX-1
        yHead = ship.positionY
        xTail = xHead + ship.length-1
        yTail = yHead

        if field[xHead][yHead] != '-':
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xHead,yHead)
            return False


    elif direction == 'down':
        xHead = ship.positionX+1
        yHead = ship.positionY
        xTail = xHead + ship.length-1
        yTail = yHead

        if field[xTail][yTail] != '-':
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xTail,yTail)
            return False

    # validate the position in field
    if xHead < 1 or xTail > field_size or yHead < 1 or yTail > field_size:
        print "Problem: The ship will be outside of field, so the movement is denied."
        return False


    # if field[xHead][yHead] != '-':
    #     print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xHead,yHead)
    #     # place_ship(field, ship)
    #     # print_field(field)
    #     return False
    remove_ship(field, ship)
    ship.positionX = xHead
    ship.positionY = yHead
    place_ship(field, ship)

def fire_shot(field, positionX, positionY):
    if positionX < 1 or positionX > field_size or positionY < 1 or positionY > field_size :
        print "Problem: The location (%2d, %2d) is not in the field, could not fire the shot: "%(positionX, positionY)
        return False
    if field[positionX][positionY] == '-':
        print "Missed"
        field[positionX][positionY] = "+"
    elif field[positionX][positionY] == '+':
        print "Location Occupied!"
    else:
        print "Hit"
        field[positionX][positionY] = "$"


def print_field(field):
    print " ".join(field[0, :])
    for row in range(1, field_size):
        display_row = np.array(field[row, :])
        # for idx in range(1,field_size):
        #     if field[row, idx].islower():
        #         display_row[idx] = '-'
        print " ".join(display_row)

if __name__ == "__main__":
    field_size = 17
    field = initial_filed(field_size)

    initial_ships(field, playerA_ships)
    initial_ships(field, playerB_ships)
    # print_field(field)
    print "Game Started: "
    player_turn = 1
    while True:
        if player_turn % 2 != 0:
            print "Player A"
            print_field(field)
        else:
            print "Player B"
            print_field(field)
        user_command = raw_input("Move Ship(M) or Fire Shot(F) :")
        if user_command == "M":
            print "Move Ship: "
            player_turn += 1
        elif user_command == "F":
            print "Fire Shot:"
            guess_row = int(raw_input("Row:"))
            guess_col = int(raw_input("Col:"))

            fire_shot(field, guess_row, guess_col)
            player_turn += 1
        else:
            print "Wrong command, try again."

# move_ship(field, playerA_ships[0], "left")
# # refresh_field(field)
# print_field(field)
#
# move_ship(field, playerA_ships[1], "up")
# # refresh_field(field)
# print_field(field)
#
# move_ship(field, playerA_ships[1], "up")
# # refresh_field(field)
# print_field(field)
#
# fire_shot(field, 2, 1)
# print_field(field)
#
# move_ship(field, playerA_ships[0], "left")
# # refresh_field(field)
# print_field(field)
#
# fire_shot(field, 2, 1)
# print_field(field)
#
# move_ship(field, playerA_ships[0], "right")
# print_field(field)
#
# fire_shot(field, 7, 11)
# print_field(field)
#
# fire_shot(field, 7, 12)
# fire_shot(field, 8, 12)
# fire_shot(field, 9, 12)
# fire_shot(field, 10, 12)
# fire_shot(field, 11, 12)
#
# move_ship(field, playerB_ships[0], "down")
#
# print_field(field)
