import numpy as np
import pygame
import sys
import tensorflow as tf
import cv2

# ======================
# Load Model
# ======================
model = tf.keras.models.load_model('best_emnist.keras')
print(model.input_shape, model.output_shape)

# ======================
# Label Mapping
# ======================
let = {}
# Digits
for i in range(10):
    let[i] = str(i)
# Uppercase A–Z
for i, ch in enumerate(range(ord('A'), ord('Z') + 1), start=10):
    let[i] = chr(ch)
# Lowercase a–z
for i, ch in enumerate(range(ord('a'), ord('z') + 1), start=36):
    let[i] = chr(ch)

# ======================
# Pygame Setup
# ======================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
size = width, height = 600, 500
screen = pygame.display.set_mode(size)

OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 60)

ROWS, COLS = 20, 16
OFFSET = 20
CELL_SIZE = 20

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None


# ======================
# Preprocessing Function
# ======================
def preprocess_canvas(handwriting):
    img = np.array(handwriting, dtype=np.float32)

    # Invert (pygame draws black on white, EMNIST is white on black)
    img = 1.0 - img

    # Resize proportionally with padding to 28x28
    h, w = img.shape
    scale = min(28 / h, 28 / w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))

    pad_h = (28 - resized.shape[0]) // 2
    pad_w = (28 - resized.shape[1]) // 2
    img_padded = np.pad(
        resized,
        ((pad_h, 28 - resized.shape[0] - pad_h),
         (pad_w, 28 - resized.shape[1] - pad_w)),
        mode="constant", constant_values=0
    )

    # Center mass (like MNIST preprocessing)
    cy, cx = np.argwhere(img_padded > 0).mean(axis=0) if np.any(img_padded > 0) else (14, 14)
    shiftx = int(np.round(14 - cx))
    shifty = int(np.round(14 - cy))
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = cv2.warpAffine(img_padded, M, (28, 28))

    # Final preprocessing
    img_final = img_centered[..., np.newaxis]
    img_final = img_final / 1.0  # already in [0,1]

    return np.expand_dims(img_final, axis=0)  # shape (1,28,28,1)


# ======================
# Main Loop
# ======================
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # Mouse input
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw grid
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            if handwriting[i][j]:
                channel = 255 - int(handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Buttons
    resetButton = pygame.Rect(30, OFFSET + ROWS * CELL_SIZE + 30, 100, 40)
    classifyButton = pygame.Rect(150, OFFSET + ROWS * CELL_SIZE + 30, 120, 40)

    pygame.draw.rect(screen, WHITE, resetButton)
    pygame.draw.rect(screen, WHITE, classifyButton)

    resetText = smallFont.render("Reset", True, BLACK)
    classifyText = smallFont.render("Classify", True, BLACK)

    screen.blit(resetText, resetText.get_rect(center=resetButton.center))
    screen.blit(classifyText, classifyText.get_rect(center=classifyButton.center))

    # Actions
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    if mouse and classifyButton.collidepoint(mouse):
        img_ready = preprocess_canvas(handwriting)
        prediction = model.predict(img_ready, verbose=0)
        classification = int(np.argmax(prediction))

    # Show classification
    if classification is not None:
        plassification = let.get(int(classification), "?")
        classificationText = largeFont.render(str(plassification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            120
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
