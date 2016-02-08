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
    """
    Ship class
    Ship_direction : vertical or horizontal
    """
    def __init__(self, flag, length, px, py, ship_direction):
        self.flag = flag
        self.length = length
        self.positionX = px
        self.positionY = py
        self.direction = ship_direction


def initial_filed(field_size):
    """
    Initialize the battle field.
    :param field_size:
    :return:
    """
    field = np.empty((field_size, field_size), dtype='S2')
    field[:]= "-"
    field[0, :] = ["  ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    field[:, 0] = ["  ", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", "11", "12", "13", "14", "15", "16"]
    return field


def place_ship(field, ship):
    """
    place a ship on field
    :param field:
    :param ship:
    :return:
    """
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


def initial_ships(field, ships):
    """
    place a list of ships in the field
    :param field:
    :param ships:
    :return:
    """
    for ship in ships:
        place_ship(field,ship)


def remove_ship(field, ship):
    """
    remove a ship from the field
    :param field:
    :param ship:
    :return:
    """
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
        ship.flag[l] = field[px][py]
        field[px][py] = '-'


def move_ship(field, ship, direction):
    """
    move the ship according to the giving direction:
    for vertical ships, they can only be moved up and down direction.
    for horizontal ships, they can only be moved right and left direction.

    :param field:
    :param ship:
    :param direction:
    :return:
    """

    direction = direction.lower()

    if (direction == 'right' or direction == 'left') and (ship.direction == 'vertical'):
        print "Problem: the ship can not move horizontally"
        return False

    if (direction == 'up' or direction == 'down') and (ship.direction == 'horizontal'):
        print "Problem: the ship can not move vertically"

    # When moving horizontal ship to right direction, validate the position for the ship's tail.
    if direction == 'right':
        xHead = ship.positionX
        yHead = ship.positionY+1
        xTail = ship.positionX
        yTail = yHead + ship.length-1

        if yTail >= field_size or field[xTail][yTail] != '-':
            print "Problem: The location (%2d, %2d) is occupied or ship will be outside the field, " \
                  "ship can not move to this location in field: " % (xTail, yTail)
            return False

    # When moving horizontal ship to left direction, validate the position for the ship's head.
    elif direction == 'left':
        xHead = ship.positionX
        yHead = ship.positionY-1
        xTail = ship.positionX
        yTail = yHead + ship.length-1

        if field[xHead][yHead] != '-' or yHead < 1:
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xHead,yHead)
            return False

    # When moving vertical ship to up direction, validate the position for the ship's head.
    elif direction == 'up':
        xHead = ship.positionX-1
        yHead = ship.positionY
        xTail = xHead + ship.length-1
        yTail = yHead

        if field[xHead][yHead] != '-' or xHead < 1:
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xHead,yHead)
            return False

    # When moving vertical ship to down direction, validate the position for the ship's tail.
    elif direction == 'down':
        xHead = ship.positionX+1
        yHead = ship.positionY
        xTail = xHead + ship.length-1
        yTail = yHead

        if xTail >= field_size or field[xTail][yTail] != '-':
            print "Problem: The location (%2d, %2d) is occupied, ship can not move to this location in field: "%(xTail,yTail)
            return False

    else:
        print "Wrong Direction. Direction only include (right,left and up,down)"
        return False

    remove_ship(field, ship)
    ship.positionX = xHead
    ship.positionY = yHead
    place_ship(field, ship)


def fire_shot(field, positionX, positionY, sink_count):
    """
    Fire a shot at the location in the field given by (positionX, positionY).
    The position should not be outside the field.
    If missed, the position will be marked as "+", otherwise it will be marked as "$"
    :param field:
    :param positionX:
    :param positionY:
    :return:
    """
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
        sink_count = sink_count +1
    return sink_count

def print_field(field, player_turn = True):
    """
    Output the field to the players.
    The player could only see their own ships and also the fired positions.
    :param field:
    :param player_turn:
    :return:
    """
    print " ".join(field[0, :])
    for row in range(1, field_size):
        display_row = np.array(field[row, :])
        for idx in range(1,field_size):
            if field[row, idx] not in ["$", "+"] and field[row, idx].islower() == player_turn:
                display_row[idx] = '-'
        print " ".join(display_row)

if __name__ == "__main__":
    """
    The main function to control the game flow.
    1. initial the field.
    2. place the ships for both players.
    3. Game with be started with player A who have two options: F--fire a shot, M--Move the ship's position
    4. After player A finished, the game will be handle over to player B.
    5. Game will be finished whenever all the ships from any player's been shot.
    """
    field_size = 17
    field = initial_filed(field_size)

    # playerA_ships = [Ship(ship_B, 4, 2, 2, "horizontal"), Ship(ship_C, 3, 4, 2, "vertical"), Ship(ship_C, 3, 4, 5, "horizontal"), Ship(ship_D, 2, 6, 4, "horizontal"), Ship(ship_D, 2, 1, 7, "vertical"), Ship(ship_D, 2, 8, 7, "horizontal")]
    # playerB_ships = [Ship(ship_b, 4, 7, 12, "vertical"), Ship(ship_c, 3, 2, 13, "horizontal"), Ship(ship_c, 3, 12, 12, "horizontal"), Ship(ship_d, 2, 16, 9, "horizontal"), Ship(ship_f, 1, 4, 11, "vertical"), Ship(ship_f, 1, 5, 15, "horizontal")]
    playerA_ships = [Ship(ship_B, 4, 2, 2, "horizontal"), Ship(ship_C, 3, 4, 2, "vertical"), Ship(ship_C, 3, 4, 5, "horizontal"), Ship(ship_D, 2, 6, 4, "horizontal"), Ship(ship_D, 2, 1, 7, "vertical"), Ship(ship_D, 2, 8, 7, "horizontal")]
    playerB_ships = [Ship(ship_d, 2, 16, 9, "horizontal")]

    playerA_sinks = 0
    playerB_sinks = 0

    initial_ships(field, playerA_ships)
    initial_ships(field, playerB_ships)
    # print_field(field)
    print "Game Started: "
    player_turn = 0
    while True:
        if player_turn % 2 == 0:
            print "Player A"
            ships = playerA_ships
            sink_count= playerB_sinks
            print_field(field, True)
        else:
            print "Player B"
            ships = playerB_ships
            sink_count= playerA_sinks
            print_field(field, False)

        user_command = raw_input("Move Ship(M) or Fire Shot(F) :")

        if user_command == "M":
            ship_id = int(raw_input("Ship ID:"))
            direction = raw_input("direction (right or left, up or down):")
            print ship_id, direction
            move_ship(field, ships[ship_id-1], direction)
            player_turn += 1

        elif user_command == "F":
            print "Fire Shot:"
            guess_row = int(raw_input("Row:"))
            guess_col = int(raw_input("Col:"))

            sink_count = fire_shot(field, guess_row, guess_col, sink_count)
            player_turn += 1
        else:
            print "Wrong command, try again."

        print playerA_sinks, playerA_sinks
        if playerA_sinks >= sum(ship.length for ship in playerA_ships):
            print "Game Over, player A won!"
            exit()
        elif playerB_sinks >= sum(ship.length for ship in playerB_ships):
            print "Game Over, player B won!"
            exit()
