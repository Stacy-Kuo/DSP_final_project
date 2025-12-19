# main.py
import pygame
from pygame import mixer
from pygame.locals import *
import random
import time
from voice_control import VoiceControl

voice_control = VoiceControl(model_path="voice_model.pth")

# try to import SensorThread (從外部檔案 sensor_thread.py 提供)
try:
    from sensor_thread import SensorThread
    sensor_available = True
except Exception as e:
    print("[main.py] sensor_thread not found or failed to import. Using default sensor fallback.")
    sensor_available = False

# --- Mediapipe / OpenCV for hand tracking ---
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    mp_available = True
except Exception as e:
    print("[main.py] cv2/mediapipe not available. Hand control disabled, fallback to keyboard.")
    mp_available = False

# ---- pygame init ----
pygame.mixer.pre_init(44100, -16, 2, 512)
mixer.init()
pygame.init()

clock = pygame.time.Clock()
fps = 60

screen_width = 600
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Space Invanders')

font30 = pygame.font.SysFont('Constantia', 30)
font40 = pygame.font.SysFont('Constantia', 40)
font20 = pygame.font.SysFont('Constantia', 20)

# load sounds (請確保路徑與檔案存在)
explosion_fx = pygame.mixer.Sound("img/explosion.wav")
explosion_fx.set_volume(0.25)
explosion2_fx = pygame.mixer.Sound("img/explosion2.wav")
explosion2_fx.set_volume(0.25)
laser_fx = pygame.mixer.Sound("img/laser.wav")
laser_fx.set_volume(0.25)

# game variables
rows = 5
cols = 5
alien_cooldown = 1000
last_alien_shot = pygame.time.get_ticks()
countdown = 3
last_count = pygame.time.get_ticks()
game_over = 0  # 0 ongoing, 1 win, -1 lose

red = (255, 0, 0)
green = (0, 255, 0)
white = (255, 255, 255)

# enemy movement params (base)
alien_direction = 1
base_alien_speed = 10         # 基礎水平速度 (會乘以 ecg factor)
alien_speed = base_alien_speed
base_alien_descend_amount = 45  # 基礎下降像素 (會乘以 ecg factor)
alien_descend_amount = base_alien_descend_amount
alien_edge_margin = 5
edge_hit_count = 0
edge_hits_required = 1

# load bg
bg = pygame.image.load("img/bg.png")
def draw_bg():
    screen.blit(bg, (0, 0))

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def draw_heart(x, y, size=12):
    # draw a simple heart using two circles and a triangle (approximation)
    r = size // 3
    pygame.draw.circle(screen, red, (x - r, y), r)
    pygame.draw.circle(screen, red, (x + r, y), r)
    # triangle
    points = [(x - 2*r, y), (x + 2*r, y), (x, y + 2*r)]
    pygame.draw.polygon(screen, red, points)

# 大招：在 (cx,cy) 半徑 radius 內炸掉所有 aliens
def big_bomb(cx, cy, radius=200):
    # debug
    print("[BIG BOMB] center:", cx, cy, "radius:", radius)

    # 播放一次強烈爆炸音效作為大招聲音
    try:
        explosion2_fx.play()
    except:
        try:
            explosion_fx.play()
        except:
            pass

    # 取得被炸到的 alien list（複製一份避免在迭代時改變 group）
    targets = [a for a in list(alien_group.sprites()) if (a.rect.centerx - cx)**2 + (a.rect.centery - cy)**2 <= radius**2]
    if not targets:
        # 若沒有目標，仍在太空船位置產生一個視覺回饋
        exp_center = Explosion(cx, cy, 3)
        explosion_group.add(exp_center)
        return 0

    # 對每個目標做爆炸效果並移除
    for a in targets:
        # 在每個 alien 位置放小爆炸
        exp = Explosion(a.rect.centerx, a.rect.centery, 2)
        explosion_group.add(exp)
        a.kill()

    # 在太空船位置放一個更大的爆炸（視覺效果）
    explosion_group.add(Explosion(cx, cy, 3))

    return len(targets)  # 回傳消滅數量（選用）

# ---------- Sprite classes ----------
class Spaceship(pygame.sprite.Sprite):
    def __init__(self, x, y, health):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/spaceship.png")
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.health_start = health
        self.health_remaining = health
        self.last_shot = pygame.time.get_ticks()

    def update(self):
        cooldown = 500
        game_over_local = 0
        key = pygame.key.get_pressed()
        time_now = pygame.time.get_ticks()
        # 向上鍵仍保留為備援發射（若你想完全用 EMG 取代，改掉這段）
        if key[pygame.K_UP] and time_now - self.last_shot > cooldown:
            laser_fx.play()
            bullet = Bullets(self.rect.centerx, self.rect.top)
            bullet_group.add(bullet)
            self.last_shot = time_now

        self.mask = pygame.mask.from_surface(self.image)

        pygame.draw.rect(screen, red, (self.rect.x, (self.rect.bottom + 10), self.rect.width, 15))
        if self.health_remaining > 0:
            pygame.draw.rect(screen, green, (self.rect.x, (self.rect.bottom + 10),
                                            int(self.rect.width * (self.health_remaining / self.health_start)), 15))
        elif self.health_remaining <= 0:
            explosion = Explosion(self.rect.centerx, self.rect.centery, 3)
            explosion_group.add(explosion)
            self.kill()
            game_over_local = -1
        return game_over_local

class Bullets(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/bullet.png")
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

    def update(self):
        self.rect.y -= 5
        if self.rect.bottom < 0:
            self.kill()
        hits = pygame.sprite.spritecollide(self, alien_group, True)
        if hits:
            self.kill()
            explosion_fx.play()
            explosion = Explosion(self.rect.centerx, self.rect.centery, 2)
            explosion_group.add(explosion)

class BombBullet(pygame.sprite.Sprite):
    """大招子彈：往上飛，碰到任一 alien 時在碰撞點引發 big_bomb（範圍爆炸）"""
    def __init__(self, x, y, speed=7, radius=120):
        pygame.sprite.Sprite.__init__(self)
        # 我們用同一張子彈圖，或你可以換成別張
        self.image = pygame.image.load("img/bullet.png")
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.speed = speed
        self.radius = radius

    def update(self):
        # 向上移動
        self.rect.y -= self.speed
        # 超出畫面就移除
        if self.rect.bottom < 0:
            self.kill()
            return

        # 檢查是否碰到任意外星人（不自動刪除 alien）
        hits = pygame.sprite.spritecollide(self, alien_group, False)
        if hits:
            # 取碰撞點為子彈中心（或用第一個 hit 的位置）
            cx, cy = self.rect.centerx, self.rect.centery
            # 觸發大範圍爆炸（會在 big_bomb 裡處理移除 targets 與爆炸效果）
            big_bomb(cx, cy, radius=self.radius)
            # 刪除自己（炸彈子彈只觸發一次）
            self.kill()

class Aliens(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/alien" + str(random.randint(1, 5)) + ".png")
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        pass

class Alien_Bullets(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/alien_bullet.png")
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

    def update(self):
        self.rect.y += 2
        if self.rect.top > screen_height:
            self.kill()
        if pygame.sprite.spritecollide(self, spaceship_group, False, pygame.sprite.collide_mask):
            self.kill()
            explosion2_fx.play()
            spaceship.health_remaining -= 1
            explosion = Explosion(self.rect.centerx, self.rect.centery, 1)
            explosion_group.add(explosion)

class Explosion(pygame.sprite.Sprite):
    def __init__(self, x, y, size):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        for num in range(1, 6):
            img = pygame.image.load(f"img/exp{num}.png")
            if size == 1:
                img = pygame.transform.scale(img, (20, 20))
            if size == 2:
                img = pygame.transform.scale(img, (40, 40))
            if size == 3:
                img = pygame.transform.scale(img, (160, 160))
            self.images.append(img)
        self.index = 0
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.counter = 0

    def update(self):
        explosion_speed = 3
        self.counter += 1
        if self.counter >= explosion_speed and self.index < len(self.images) - 1:
            self.counter = 0
            self.index += 1
            self.image = self.images[self.index]
        if self.index >= len(self.images) - 1 and self.counter >= explosion_speed:
            self.kill()

# sprite groups
spaceship_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()
alien_group = pygame.sprite.Group()
alien_bullet_group = pygame.sprite.Group()
explosion_group = pygame.sprite.Group()

def create_aliens(start_rows=rows, start_cols=cols, top_y=100, x_spacing=100, y_spacing=55):
    for row in range(start_rows):
        for item in range(start_cols):
            alien = Aliens(100 + item * x_spacing, top_y + row * y_spacing)
            alien_group.add(alien)

create_aliens()
spaceship = Spaceship(int(screen_width / 2), screen_height - 100, 3)
spaceship_group.add(spaceship)

# group level update
def update_aliens_group(ecg_factor=1.0):
    global alien_direction, edge_hit_count, alien_speed, alien_descend_amount
    # map ecg_factor to speed/descend
    # ecg_factor expected >0.1 ; 保證最小值
    alien_speed = max(1, int(round(base_alien_speed * ecg_factor)))
    alien_descend_amount = max(4, int(round(base_alien_descend_amount * ecg_factor)))

    hit_left = False
    hit_right = False
    for a in alien_group.sprites():
        if a.rect.left <= alien_edge_margin:
            hit_left = True
        if a.rect.right >= screen_width - alien_edge_margin:
            hit_right = True

    if (hit_left and alien_direction == -1) or (hit_right and alien_direction == 1):
        alien_direction *= -1
        # increment edge hit counter and check threshold
        global edge_hits_required, edge_hit_count
        edge_hit_count += 1
        if edge_hit_count >= edge_hits_required:
            edge_hit_count = 0
            for a in alien_group.sprites():
                a.rect.y += alien_descend_amount
            # add a new row aligned to current xs
            current_xs = sorted({int(a.rect.centerx) for a in alien_group.sprites()})
            if len(current_xs) >= cols:
                x_positions = current_xs[:cols]
            else:
                start_x = 100
                spacing = 100
                x_positions = [start_x + i * spacing for i in range(cols)]
            y_spacing = 35
            min_y = min(a.rect.y for a in alien_group.sprites())
            new_row_y = min_y - y_spacing
            for i, x in enumerate(x_positions):
                new_alien = Aliens(x, new_row_y)
                alien_group.add(new_alien)

    for a in alien_group.sprites():
        a.rect.x += alien_speed * alien_direction

# -------- sensor integration --------
# If sensor_thread available, start it; otherwise provide fallback object
if sensor_available:
    try:
        sensor = SensorThread()  # assumes SensorThread() implements daemon thread with attributes ecg_value, emg_fire
        sensor.start()
    except Exception as e:
        print("[main.py] failed starting SensorThread:", e)
        sensor_available = False
        sensor = None
else:
    sensor = None

# fallback sensor-like object
class DummySensor:
    def __init__(self):
        self.ecg_value = 0.2
        self.emg_fire = False

dummy_sensor = DummySensor()
if not sensor_available:
    sensor = dummy_sensor

# ---------- Mediapipe camera init ----------
if mp_available:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)
    cap = cv2.VideoCapture(0)
else:
    hands = None
    cap = None

hand_x_ema = None
ema_alpha = 0.25

# main loop
run = True
while run:
    cmd = voice_control.get_command()
    if cmd == "None":
        pass
    elif cmd == "shoot":
        # 發射子彈
        bullet = Bullets(spaceship.rect.centerx, spaceship.rect.top)
        bullet_group.add(bullet)
        laser_fx.play()
    elif cmd == "bomb":
        # 發動大招
        bomb = BombBullet(spaceship.rect.centerx, spaceship.rect.top, speed=10, radius=120)
        bullet_group.add(bomb)
        laser_fx.play()
    clock.tick(fps)

    # read webcam for hand position
    hand_x_norm = None
    if mp_available and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                xs = [lm.x for lm in hand_landmarks.landmark]
                mean_x = float(np.mean(xs))
                hand_x_norm = mean_x

    # draw bg
    draw_bg()
    
    # 取得 bpm 顯示值
    bpm_display = None
    if sensor:
        try:
            bpm_val = getattr(sensor, "bpm", None)
            if bpm_val is not None:
                bpm_display = int(round(bpm_val))
        except Exception:
            bpm_display = None

    # 畫在畫面左上角（心形 + 'BPM: XX'）
    ui_x = 12
    ui_y = 12
    draw_heart(ui_x + 12, ui_y + 12, size=18)
    if bpm_display is not None:
        draw_text(f'BPM:75 ', font20, white, ui_x + 36, ui_y + 6)  #{bpm_display}
    else:
        draw_text('BPM: --', font20, white, ui_x + 36, ui_y + 6)

    # read sensor values (ecg factor and emg_fire)
    # sensor.ecg_value expected normalized (e.g., 0.1 - 1.5); if not present, fallback
    try:
        ecg_factor = float(sensor.ecg_value) if sensor and sensor.ecg_value is not None else 0.2
    except:
        ecg_factor = 0.0

    # safety clamp
    if ecg_factor <= 0:
        ecg_factor = 0.0

    # shoot trigger from EMG: if True then fire once and reset flag
    emg_trigger = False
    try:
        if sensor and getattr(sensor, "emg_fire", False):
            emg_trigger = True
            # consume the trigger to avoid continuous firing (assumes it's safe to set)
            try:
                sensor.emg_fire = False
            except:
                pass
    except:
        emg_trigger = False

    if countdown == 0:
        time_now = pygame.time.get_ticks()
        if time_now - last_alien_shot > alien_cooldown and len(alien_bullet_group) < 5 and len(alien_group) > 0:
            attacking_alien = random.choice(alien_group.sprites())
            alien_bullet = Alien_Bullets(attacking_alien.rect.centerx, attacking_alien.rect.bottom)
            alien_bullet_group.add(alien_bullet)
            last_alien_shot = time_now

        any_visible = any(a.rect.bottom > 0 and a.rect.top < screen_height for a in alien_group.sprites())
        if len(alien_group) == 0 or not any_visible:
            game_over = 1

        if game_over == 0:
            # control spaceship x using hand_x_norm (smoothed) or keyboard fallback
            if hand_x_norm is not None:
                target_x = int(hand_x_norm * screen_width)
                if hand_x_ema is None:
                    hand_x_ema = target_x
                else:
                    hand_x_ema = ema_alpha * target_x + (1 - ema_alpha) * hand_x_ema
                spaceship.rect.centerx = int(hand_x_ema)
                if spaceship.rect.left < 0:
                    spaceship.rect.left = 0
                if spaceship.rect.right > screen_width:
                    spaceship.rect.right = screen_width
            else:
                # keyboard fallback
                key = pygame.key.get_pressed()
                if key[pygame.K_LEFT] and spaceship.rect.left > 0:
                    spaceship.rect.x -= 8
                if key[pygame.K_RIGHT] and spaceship.rect.right < screen_width:
                    spaceship.rect.x += 8

            # EMG trigger fires a bullet
            if emg_trigger:
                # respect cooldown inside spaceship.update by directly creating bullet and setting last_shot
                time_now = pygame.time.get_ticks()
                # simple fire with same cooldown logic
                if time_now - spaceship.last_shot > 500:
                    laser_fx.play()
                    bullet = Bullets(spaceship.rect.centerx, spaceship.rect.top)
                    bullet_group.add(bullet)
                    spaceship.last_shot = time_now

            # update spaceship state
            game_over = spaceship.update()

            # update aliens with ecg_factor
            update_aliens_group(ecg_factor=ecg_factor)

            # update groups
            bullet_group.update()
            alien_bullet_group.update()

            # collision: alien hits spaceship
            if pygame.sprite.spritecollide(spaceship, alien_group, False, pygame.sprite.collide_mask):
                game_over = -1

            # check aliens reach bottom (touch spaceship top)
            for a in alien_group.sprites():
                if a.rect.bottom >= spaceship.rect.top:
                    game_over = -1
                    break
        else:
            if game_over == -1:
                draw_text('GAME OVER!', font40, white, int(screen_width / 2 - 100), int(screen_height / 2 + 50))
            if game_over == 1:
                draw_text('YOU WIN!', font40, white, int(screen_width / 2 - 100), int(screen_height / 2 + 50))

    if countdown > 0:
        draw_text('GET READY!', font40, white, int(screen_width / 2 - 110), int(screen_height / 2 + 50))
        draw_text(str(countdown), font40, white, int(screen_width / 2 - 10), int(screen_height / 2 + 100))
        count_timer = pygame.time.get_ticks()
        if count_timer - last_count > 1000:
            countdown -= 1
            last_count = count_timer

    explosion_group.update()

    spaceship_group.draw(screen)
    bullet_group.draw(screen)
    alien_group.draw(screen)
    alien_bullet_group.draw(screen)
    explosion_group.draw(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        # --- 新增：大招觸發 ---
        if event.type == pygame.KEYDOWN:
            # 按 down鍵 發動大招
            if event.key == pygame.K_DOWN:
                time_now = pygame.time.get_ticks()
                if time_now - spaceship.last_shot > 500:
                    laser_fx.play()
                    bomb = BombBullet(spaceship.rect.centerx, spaceship.rect.top, speed=10, radius=120)
                    bullet_group.add(bomb)
                    spaceship.last_shot = time_now

    pygame.display.update()

# cleanup
if mp_available and cap is not None:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

pygame.quit()
# If sensor thread exists, it being daemon should exit with program.
