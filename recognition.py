import numpy as np
import pygame
import sys
import tensorflow as tf
import time

# Check command-line arguments

model = tf.keras.models.load_model('brad.keras')
let={10:'A',
     11:'B',
     12:'C',
     13:'D',
     14:'E',
     15:'F',
     16:'G',
     17:'H',
     18:'I',
     19:'J',
     20:'K',
     21:'L',
     22:'M',
     23:'N',
     24:'O',
     25:'P',
     26:'Q',
     27:'R',
     28:'S',
     29:'T',
     30:'U',
     31:'V',
     32:'W',
     33:'X',
     34:'Y',
     35:'Z',
     }
# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 20, 16

OFFSET = 20
CELL_SIZE = 15

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

while True:

    # Check if game quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(handwriting).reshape(1, 20,16, 1)]
        ).argmax()

    # Show classification if one exists
    if classification is not None:
        
        if int(classification)>=10:
            plassification=let[classification]
        else:
            plassification=classification
        classificationText = largeFont.render(str(plassification), True, WHITE)    
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
