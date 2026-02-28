import logging
import math
import random
import audio
import org.threejs as three
from controls import Keyboard, ControlAxis
from units import Ship, Asteroid, Bullet
from utils import wrap, now, FPSCounter, coroutine, clamp, set_limits

DEBUG: bool = True
logger = logging.getLogger('root')
logger.addHandler(logging.StreamHandler())

if DEBUG:
    logger.setLevel(logging.INFO)
    logger.info('====== debug logging on =====')

def waiter(*args: tuple) -> tuple:
    return (True, args[0])

def done(*args: tuple) -> None:
    print('done at', args[0])

def hfov(vfov: float, w: float, h: float) -> float:
    """gives horizontal fov (in rads) for given vertical fov (in rads) and aspect ratio"""
    return

class Graphics:
    def __init__(self, w: float, h: float, canvas, fov: float = 53.13) -> None:
        self.width: float = float(w)
        self.height: float = float(h)
        self.scene: three.Scene = three.Scene()
        self.camera: three.PerspectiveCamera = three.PerspectiveCamera(fov, self.width / self.height, 1, 500)
        self.vfov: float = math.radians(fov)
        self.hfov: float = 2 * math.atan(math.tan(math.radians(fov) / 2.0) * (w / h * 1.0))
        self.camera.position.set(0, 0, 80)
        self.camera.lookAt(self.scene.position)
        self.renderer: three.WebGLRenderer = three.WebGLRenderer({'Antialias': True})
        self.renderer.setSize(self.width, self.height)
        canvas.appendChild(self.renderer.domElement)

    def render(self) -> None:
        self.renderer.render(self.scene, self.camera)

    def add(self, item: any) -> None:
        self.scene.add(item.geo)

    def extent(self) -> tuple:
        v_extent: float = math.tan(self.vfov / 2.0) * 80
        h_extent: float = math.tan(self.hfov / 2.0) * 80
        return (h_extent, v_extent)

class Audio:
    def __init__(self, audio_path: str = '') -> None:
        pth = lambda p: audio_path + p
        self.fire_rota: list = [audio.clip(pth('344276__nsstudios__laser3.wav')), audio.clip(pth('344276__nsstudios__laser3.wav')), audio.clip(pth('344276__nsstudios__laser3.wav')), audio.clip(pth('344276__nsstudios__laser3.wav'))]
        self.explosion_rota: list = [audio.clip(pth('108641__juskiddink__nearby-explosion-with-debris.wav')), audio.clip(pth('108641__juskiddink__nearby-explosion-with-debris.wav')), audio.clip(pth('108641__juskiddink__nearby-explosion-with-debris.wav')), audio.clip(pth('108641__juskiddink__nearby-explosion-with-debris.wav'))]
        self.thrust: any = audio.loop(pth('146770__qubodup__rocket-boost-engine-loop.wav'))
        self.fail: any = audio.clip(pth('172950__notr__saddertrombones.mp3'))
        self.thrust.play()
        self.shoot_ctr: int = 0
        self.explode_ctr: int = 0

    def fire(self) -> None:
        self.fire_rota[self.shoot_ctr % 4].play()
        self.shoot_ctr += 1

    def explode(self) -> None:
        self.explosion_rota[self.shoot_ctr % 4].play()
        self.shoot_ctr += 1

class Game:
    def __init__(self, canvas, fullscreen: bool = True) -> None:
        self.keyboard: Keyboard = Keyboard()
        if fullscreen:
            self.graphics: Graphics = Graphics(window.innerWidth, window.innerHeight, canvas)
        else:
            self.graphics: Graphics = Graphics(canvas.offsetWidth, 3 * canvas.offsetWidth / 4, canvas)
        self.extents: tuple = self.graphics.extent()
        set_limits(*self.extents)
        self.create_controls()
        self.ship: Ship = None
        self.bullets: list = []
        self.asteroids: list = []
        self.helptext: any = None
        self.resetter: any = None
        self.setup()
        self.last_frame: float = now()
        self.audio: Audio = Audio()
        self.lives: int = 3
        self.score: int = 0
        self.score_display: any = document.getElementById('score')
        self.fps_counter: FPSCounter = FPSCounter(document.getElementById('FPS'))
        v_center: float = canvas.offsetHeight / 3
        title: any = document.getElementById('game_over')
        title.style.top = v_center
        hud: any = document.getElementById('hud')
        hud.style.width = canvas.offsetWidth
        hud.style.height = canvas.offsetHeight
        frame: any = document.getElementById('game_frame')
        frame.style.min_height = canvas.offsetHeight

    def create_controls(self) -> None:
        self.keyboard.add_handler('spin', ControlAxis('ArrowRight', 'ArrowLeft', attack=1, decay=0.6))
        self.keyboard.add_handler('thrust', ControlAxis('ArrowUp', 'ArrowDown', attack=0.65, decay=2.5, deadzone=0.1))
        self.keyboard.add_handler('fire', ControlAxis(' ', 'None', attack=10))
        document.onkeydown = self.keyboard.key_down
        document.onkeyup = self.keyboard.key_up

        def suppress_scroll(e: any) -> None:
            if e.keyCode in [32, 37, 38, 39, 40]:
                e.preventDefault()
        window.addEventListener('keydown', suppress_scroll, False)

    def setup(self) -> None:
        self.ship = Ship(self.keyboard, self)
        self.graphics.add(self.ship)

        def rsign() -> int:
            if random.random() < 0.5:
                return -1
            return 1
        for a in range(8):
            x: float = (random.random() - 0.5) * 2
            y: float = random.random() - 0.5
            z: float = 0
            offset: three.Vector3 = three.Vector3(x, y, z)
            offset.normalize()
            push: int = random.randint(20, 60)
            offset = offset.multiplyScalar(push)
            r: float = (random.random() + 1.0) * 2.5
            asteroid: Asteroid = Asteroid(r, offset)
            mx: float = random.random() + random.random() + random.random(2) - 2.0
            my: float = random.random() + random.random() + random.random(2) - 2.0
            asteroid.momentum = three.Vector3(mx, my, 0)
            self.graphics.add(asteroid)
            self.asteroids.append(asteroid)
        for b in range(8):
            bullet: Bullet = Bullet()
            self.graphics.add(bullet)
            self.bullets.append(bullet)
        self.helptext = self.help_display()

    def tick(self) -> None:
        if len(self.asteroids) == 0 or self.lives < 1:
            document.getElementById('game_over').style.visibility = 'visible'
            document.getElementById('credits').style.visibility = 'visible'
            document.getElementById('game_canvas').style.cursor = 'auto'
            return
        requestAnimationFrame(self.tick)
        t: float = now() - self.last_frame
        self.fps_counter.update(t)
        self.keyboard.update(t)
        if self.ship.visible:
            self.handle_input(t)
        dead: list = []
        for b in self.bullets:
            if b.position.z < 1000:
                for a in self.asteroids:
                    if a.bbox.contains(b.position):
                        d: float = a.geo.position.distanceTo(b.position)
                        if d < a.radius:
                            b.reset()
                            dead.append(a)
        if self.ship.visible:
            for a in self.asteroids:
                if a.bbox.contains(self.ship.position):
                    d: float = a.geo.position.distanceTo(self.ship.position)
                    if d < a.radius + 0.5:
                        self.resetter = self.kill()
                        print('!!', self.resetter)
                        dead.append(a)
        else:
            self.resetter.advance(t)
        for d in dead:
            self.asteroids.remove(d)
            new_score: int = int(100 * d.radius)
            self.update_score(new_score)
            d.geo.visible = False
            if d.radius > 1.5:
                self.audio.explode()
                new_asteroids: int = random.randint(2, 5)
                for n in range(new_asteroids):
                    new_a: Asteroid = Asteroid((d.radius + 1.0) / new_asteroids, d.position)
                    mx: float = (random.random() - 0.5) * 6
                    my: float = (random.random() - 0.5) * 4
                    new_a.momentum = three.Vector3().copy(d.momentum)
                    new_a.momentum.add(three.Vector3(mx, my, 0))
                    self.graphics.add(new_a)
                    self.asteroids.append(new_a)
        for b in self.bullets:
            b.update(t)
        self.ship.update(t)
        wrap(self.ship.geo)
        for item in self.asteroids:
            item.update(t)
            wrap(item.geo)
        if self.resetter is not None:
            self.resetter.advance(t)
        if self.helptext is not None:
            self.helptext.advance(t)
        self.graphics.render()
        self.last_frame = now()

    def handle_input(self, t: float) -> None:
        if self.keyboard.get_axis('fire') >= 1:
            mo: three.Vector3 = three.Vector3().copy(self.ship.momentum).multiplyScalar(t)
            if self.fire(self.ship.position, self.ship.heading, mo):
                self.audio.fire()
            self.keyboard.clear('fire')
        spin: float = self.keyboard.get_axis('spin')
        self.ship.spin(spin * t)
        thrust: float = self.keyboard.get_axis('thrust')
        self.audio.thrust.volume = clamp(thrust * 5, 0, 1)
        self.ship.thrust(thrust * t)

    def fire(self, pos: three.Vector3, vector: three.Vector3, momentum: three.Vector3, t: float) -> bool:
        for each_bullet in self.bullets:
            if each_bullet.geo.position.z >= 1000:
                each_bullet.geo.position.set(pos.x, pos.y, pos.z)
                each_bullet.vector = vector
                each_bullet.lifespan = 0
                each_bullet.momentum = three.Vector3().copy(momentum).multiplyScalar(0.66)
                return True
        return False

    def kill(self) -> any:
        self.lives -= 1
        self.ship.momentum = three.Vector3(0, 0, 0)
        self.ship.position = three.Vector3(0, 0, 0)
        self.ship.geo.setRotationFromEuler(three.Euler(0, 0, 0))
        self.keyboard.clear('spin')
        self.keyboard.clear('thrust')
        self.keyboard.clear('fire')
        self.ship.visible = False
        self.audio.fail.play()
        can_reappear: float = now() + 3.0

        def reappear(t: float) -> tuple:
            if now() < can_reappear:
                return (True, 'waiting')
            for a in self.asteroids:
                if a.bbox.contains(self.ship.position):
                    return (True, "can't spawn")
            return (False, 'OK')

        def clear_resetter() -> None:
            self.ship.visible = True
            self.resetter = None
        reset: any = coroutine(reappear, clear_resetter)
        next(reset)
        return reset

    def help_display(self) -> any:
        """
        cycle through the help messages, fading in and out
        """
        messages: int = 3
        repeats: int = 2
        elapsed: float = 0
        count: int = 0
        period: float = 2.25

        def display_stuff(t: float) -> tuple:
            nonlocal elapsed, count, messages, repeats
            if count < messages * repeats:
                elapsed += t / period
                count = int(elapsed)
                lintime: float = elapsed % 1
                opacity: float = math.pow(math.sin(lintime * 3.1415), 2)
                logger.info(lintime)
                document.getElementById('instructions{}'.format(count % 3)).style.opacity = opacity
                return (True, opacity)
            else:
                return (False, 'OK')

        def done() -> None:
            document.getElementById('instructions1').style.visiblity = 'hidden'
        displayer: any = coroutine(display_stuff, done)
        next(displayer)
        logger.debug('displayer', displayer)
        return displayer

    def update_score(self, score: int) -> None:
        self.score += score
        self.score_display.innerHTML = self.score
        print(self.score, self.score_display)
canvas = document.getElementById('game_canvas')
game = Game(canvas, True)
game.tick()
