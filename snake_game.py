import pygame
import random
import sys


pygame.init()
screen1 = pygame.display.set_mode((720, 480))
clock = pygame.time.Clock()

CELL_SIZE = 20 # 1 Cell is 20 pixels
SNAKE_COLOR = (34, 139, 34) # RGB values for Forest Green color
APPLE_COLOR = (255, 8, 0)

GRID_WIDTH = screen1.get_width() // CELL_SIZE # Gets the number of cells in width 
GRID_HEIGHT = screen1.get_height() // CELL_SIZE # Gets the number of cells in height 

class Snake:

    def __init__(self):
        self.body = [(15, 10), (14, 10), (13, 10)]
        self.direction = (1, 0) # Initially moving to the right

    def move(self):
        head_x, head_y = self.body[0] # Sets head to first tuple
        direction_x, direction_y = self.direction
        
        # Update the position of the head
        new_head = (head_x + direction_x, head_y + direction_y)

        # Update the position of the whole snake
        self.body = [new_head] + self.body[:-1]

    def grow(self):
        
        # Add previous postion of tail to the body 
        self.body.append(self.body[-1])

    def collide_self(self, pt = None):
        # Only the head can collide with its body
        for index, body_position in enumerate(self.body):
            if index == 0:
                continue
            # Check if head collides with body
            if self.body[0] == body_position:
                return True
            
        return False


    def collide_wall(self, pt = None):
        head_x = self.body[0][0]
        head_y = self.body[0][1]

        if head_x < 0 or head_y < 0 or head_x >= GRID_WIDTH or head_y >= GRID_HEIGHT:
            return True
        else:
            return False
        
    def draw(self):
        for segment in self.body:
            pygame.draw.rect(screen1, SNAKE_COLOR, pygame.Rect(segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Apple:

    def __init__(self):

        # 2D Tuple : random (x, y) coordinates
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

    def random_pos(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

    def draw(self):
        pygame.draw.rect(screen1, APPLE_COLOR, (self.position[0]*CELL_SIZE, self.position[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Score:

    def __init__(self):
        self.current_score = 0
        self.best_score = 0

    def increase(self):
        self.current_score += 1

    def reset(self):
        self.current_score = 0

    def max_score(self):
        if self.current_score > self.best_score:
            self.best_score = self.current_score

    def display(self):
        score_display = score_obj.render(f'Score: {score.current_score}', True, (255, 255, 255))
        screen1.blit(score_display, (10, 10))

        best_score_display = score_obj.render(f'Best Score: {score.best_score}', True, (255, 255, 255))
        screen1.blit(best_score_display, (550, 10))
            
restart_obj = pygame.font.Font(None, size = 35)

score_obj = pygame.font.Font(None, size=35)
score_obj.set_underline(True)

# Create an instance of Snake, Apple, Score classes
snake = Snake()
apple = Apple()
score = Score()

reward_acc = 1.0

# FOR AI
def play_step(snake, apple, score, final_move):
    global reward_acc
    reward_acc -= 0.01
    reward = reward_acc
    snake.direction = tuple(final_move)
    snake.move()
    game_over = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Snake eats the apple, grows + new apple spawns
    if snake.body[0][0] == apple.position[0] and snake.body[0][1] == apple.position[1]:
        apple.random_pos()
        snake.grow()
        score.increase()
        reward = 30
        reward_acc = 1.0

    if snake.collide_wall() or snake.collide_self():
        reward = -10
        reward_acc = 1.0
        game_over = True
        return reward, game_over, score.current_score
    
    # Update the UI
    clock.tick(20)
    screen1.fill((0, 0, 0))
    snake.draw()
    apple.draw()
    # score.display()
    pygame.display.flip()
    return reward, game_over, score.current_score



def reset_game():
    global snake, apple, score
    snake.body = [(15, 10), (14, 10), (13, 10)]
    snake.direction = (1, 0)
    apple.random_pos()
    score.reset()

def main():
    pygame.init()
    global score, apple, snake

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != (0, 1):
                    snake.direction = (0, -1)
                if event.key == pygame.K_DOWN and snake.direction != (0, -1):
                    snake.direction = (0, 1)
                if event.key == pygame.K_RIGHT and snake.direction != (-1, 0):
                    snake.direction = (1, 0)
                if event.key == pygame.K_LEFT and snake.direction != (1, 0):
                    snake.direction = (-1, 0)

        # Get the snake to move
        snake.move()

        # Snake eats the apple, grows + new apple spawns
        if snake.body[0][0] == apple.position[0] and snake.body[0][1] == apple.position[1]:
            apple.random_pos()
            snake.grow()
            score.increase()

        if snake.collide_wall() or snake.collide_self():
            running = False

        # Set the best score
        if score.current_score > score.best_score:
            score.best_score = score.current_score


        screen1.fill((0, 0, 0))
        snake.draw()
        apple.draw()
        score.display()
        pygame.display.flip()

        clock.tick(10)
    
    restart_message = ["Would you like to restart ?", "Press Y for Yes.",  "Press N for No"]
    y_pos = 200
    for i in restart_message:
        restart_txt = restart_obj.render(i, True, (255, 255, 255))
        screen1.blit(restart_txt, (250, y_pos))
        y_pos += restart_txt.get_height()
        pygame.display.flip()

    restart_response = True
    while restart_response:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    sys.exit()
                if event.key == pygame.K_y:
                    restart_response = False
                    reset_game()
                    return True
                
if __name__ == "__main__":
    while True:
        if main() is False:
            break
