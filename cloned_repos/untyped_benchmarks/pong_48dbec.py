__pragma__('skip')
document = window = Math = Date = 0
__pragma__('noskip')
__pragma__('noalias', 'clear')
from com.fabricjs import fabric
orthoWidth = 1000
orthoHeight = 750
fieldHeight = 650
enter, esc, space = (13, 27, 32)
window.onkeydown = lambda event: event.keyCode != space

class Attribute:

    def __init__(self, game):
        self.game = game
        self.game.attributes.append(self)
        self.install()
        self.reset()

    def reset(self):
        self.commit()

    def predict(self):
        pass

    def interact(self):
        pass

    def commit(self):
        pass

class Sprite(Attribute):

    def __init__(self, game, width, height):
        self.width = width
        self.height = height
        Attribute.__init__(self, game)

    def install(self):
        self.image = __new__(fabric.Rect({'width': self.game.scaleX(self.width), 'height': self.game.scaleY(self.height), 'originX': 'center', 'originY': 'center', 'fill': 'white'}))
    __pragma__('kwargs')

    def reset(self, vX=0, vY=0, x=0, y=0):
        self.vX = vX
        self.vY = vY
        self.x = x
        self.y = y
        Attribute.reset(self)
    __pragma__('nokwargs')

    def predict(self):
        self.x += self.vX * self.game.deltaT
        self.y += self.vY * self.game.deltaT

    def commit(self):
        self.image.left = self.game.orthoX(self.x)
        self.image.top = self.game.orthoY(self.y)

    def draw(self):
        self.game.canvas.add(self.image)

class Paddle(Sprite):
    margin = 30
    width = 10
    height = 100
    speed = 400

    def __init__(self, game, index):
        self.index = index
        Sprite.__init__(self, game, self.width, self.height)

    def reset(self):
        Sprite.reset(self, x=orthoWidth // 2 - self.margin if self.index else -orthoWidth // 2 + self.margin, y=0)

    def predict(self):
        self.vY = 0
        if self.index:
            if self.game.keyCode == ord('K'):
                self.vY = self.speed
            elif self.game.keyCode == ord('M'):
                self.vY = -self.speed
        elif self.game.keyCode == ord('A'):
            self.vY = self.speed
        elif self.game.keyCode == ord('Z'):
            self.vY = -self.speed
        Sprite.predict(self)

    def interact(self):
        self.y = Math.max(self.height // 2 - fieldHeight // 2, Math.min(self.y, fieldHeight // 2 - self.height // 2))
        if self.y - self.height // 2 < self.game.ball.y < self.y + self.height // 2 and (self.index == 0 and self.game.ball.x < self.x or (self.index == 1 and self.game.ball.x > self.x)):
            self.game.ball.x = self.x
            self.game.ball.vX = -self.game.ball.vX
            self.game.ball.speedUp(self)

class Ball(Sprite):
    side = 8
    speed = 300

    def __init__(self, game):
        Sprite.__init__(self, game, self.side, self.side)

    def reset(self):
        angle = self.game.serviceIndex * Math.PI + (1 if Math.random() > 0.5 else -1) * Math.random() * Math.atan(fieldHeight / orthoWidth)
        Sprite.reset(self, vX=self.speed * Math.cos(angle), vY=self.speed * Math.sin(angle))

    def predict(self):
        Sprite.predict(self)
        if self.x < -orthoWidth // 2:
            self.game.scored(1)
        elif self.x > orthoWidth // 2:
            self.game.scored(0)
        if self.y > fieldHeight // 2:
            self.y = fieldHeight // 2
            self.vY = -self.vY
        elif self.y < -fieldHeight // 2:
            self.y = -fieldHeight // 2
            self.vY = -self.vY

    def speedUp(self, bat):
        factor = 1 + 0.15 * (1 - Math.abs(self.y - bat.y) / (bat.height // 2)) ** 2
        if Math.abs(self.vX) < 3 * self.speed:
            self.vX *= factor
            self.vY *= factor

class Scoreboard(Attribute):
    nameShift = 75
    hintShift = 25

    def install(self):
        self.playerLabels = [__new__(fabric.Text('Player {}'.format(name), {'fill': 'white', 'fontFamily': 'arial', 'fontSize': '{}'.format(self.game.canvas.width / 30), 'left': self.game.orthoX(position * orthoWidth), 'top': self.game.orthoY(fieldHeight // 2 + self.nameShift)})) for name, position in (('AZ keys:', -7 / 16), ('KM keys:', 1 / 16))]
        self.hintLabel = __new__(fabric.Text('[spacebar] starts game, [enter] resets score', {'fill': 'white', 'fontFamily': 'arial', 'fontSize': '{}'.format(self.game.canvas.width / 70), 'left': self.game.orthoX(-7 / 16 * orthoWidth), 'top': self.game.orthoY(fieldHeight // 2 + self.hintShift)}))
        self.image = __new__(fabric.Line([self.game.orthoX(-orthoWidth // 2), self.game.orthoY(fieldHeight // 2), self.game.orthoX(orthoWidth // 2), self.game.orthoY(fieldHeight // 2)], {'stroke': 'white'}))

    def increment(self, playerIndex):
        self.scores[playerIndex] += 1

    def reset(self):
        self.scores = [0, 0]
        Attribute.reset(self)

    def commit(self):
        self.scoreLabels = [__new__(fabric.Text('{}'.format(score), {'fill': 'white', 'fontFamily': 'arial', 'fontSize': '{}'.format(self.game.canvas.width / 30), 'left': self.game.orthoX(position * orthoWidth), 'top': self.game.orthoY(fieldHeight // 2 + self.nameShift)})) for score, position in zip(self.scores, (-2 / 16, 6 / 16))]

    def draw(self):
        for playerLabel, scoreLabel in zip(self.playerLabels, self.scoreLabels):
            self.game.canvas.add(playerLabel)
            self.game.canvas.add(scoreLabel)
            self.game.canvas.add(self.hintLabel)
        self.game.canvas.add(self.image)

class Game:

    def __init__(self):
        self.serviceIndex = 1 if Math.random() > 0.5 else 0
        self.pause = True
        self.keyCode = None
        self.textFrame = document.getElementById('text_frame')
        self.canvasFrame = document.getElementById('canvas_frame')
        self.buttonsFrame = document.getElementById('buttons_frame')
        self.canvas = __new__(fabric.Canvas('canvas', {'backgroundColor': 'black', 'originX': 'center', 'originY': 'center'}))
        self.canvas.onWindowDraw = self.draw
        self.canvas.lineWidth = 2
        self.canvas.clear()
        self.attributes = []
        self.paddles = [Paddle(self, index) for index in range(2)]
        self.ball = Ball(self)
        self.scoreboard = Scoreboard(self)
        window.setInterval(self.update, 10)
        window.setInterval(self.draw, 20)
        window.addEventListener('keydown', self.keydown)
        window.addEventListener('keyup', self.keyup)
        self.buttons = []
        for key in ('A', 'Z', 'K', 'M', 'space', 'enter'):
            button = document.getElementById(key)
            button.addEventListener('mousedown', (lambda aKey: lambda: self.mouseOrTouch(aKey, True))(key))
            button.addEventListener('touchstart', (lambda aKey: lambda: self.mouseOrTouch(aKey, True))(key))
            button.addEventListener('mouseup', (lambda aKey: lambda: self.mouseOrTouch(aKey, False))(key))
            button.addEventListener('touchend', (lambda aKey: lambda: self.mouseOrTouch(aKey, False))(key))
            button.style.cursor = 'pointer'
            button.style.userSelect = 'none'
            self.buttons.append(button)
        self.time = +__new__(Date)
        window.onresize = self.resize
        self.resize()

    def install(self):
        for attribute in self.attributes:
            attribute.install()

    def mouseOrTouch(self, key, down):
        if down:
            if key == 'space':
                self.keyCode = space
            elif key == 'enter':
                self.keyCode = enter
            else:
                self.keyCode = ord(key)
        else:
            self.keyCode = None

    def update(self):
        oldTime = self.time
        self.time = +__new__(Date)
        self.deltaT = (self.time - oldTime) / 1000.0
        if self.pause:
            if self.keyCode == space:
                self.pause = False
            elif self.keyCode == enter:
                self.scoreboard.reset()
        else:
            for attribute in self.attributes:
                attribute.predict()
            for attribute in self.attributes:
                attribute.interact()
            for attribute in self.attributes:
                attribute.commit()

    def scored(self, playerIndex):
        self.scoreboard.increment(playerIndex)
        self.serviceIndex = 1 - playerIndex
        for paddle in self.paddles:
            paddle.reset()
        self.ball.reset()
        self.pause = True

    def commit(self):
        for attribute in self.attributes:
            attribute.commit()

    def draw(self):
        self.canvas.clear()
        for attribute in self.attributes:
            attribute.draw()

    def resize(self):
        self.pageWidth = window.innerWidth
        self.pageHeight = window.innerHeight
        self.textTop = 0
        if self.pageHeight > 1.2 * self.pageWidth:
            self.canvasWidth = self.pageWidth
            self.canvasTop = self.textTop + 300
        else:
            self.canvasWidth = 0.6 * self.pageWidth
            self.canvasTop = self.textTop + 200
        self.canvasLeft = 0.5 * (self.pageWidth - self.canvasWidth)
        self.canvasHeight = 0.6 * self.canvasWidth
        self.buttonsTop = self.canvasTop + self.canvasHeight + 50
        self.buttonsWidth = 500
        self.textFrame.style.top = self.textTop
        self.textFrame.style.left = self.canvasLeft + 0.05 * self.canvasWidth
        self.textFrame.style.width = 0.9 * self.canvasWidth
        self.canvasFrame.style.top = self.canvasTop
        self.canvasFrame.style.left = self.canvasLeft
        self.canvas.setDimensions({'width': self.canvasWidth, 'height': self.canvasHeight})
        self.buttonsFrame.style.top = self.buttonsTop
        self.buttonsFrame.style.left = 0.5 * (self.pageWidth - self.buttonsWidth)
        self.buttonsFrame.style.width = self.canvasWidth
        self.install()
        self.commit()
        self.draw()

    def scaleX(self, x):
        return x * (self.canvas.width / orthoWidth)

    def scaleY(self, y):
        return y * (self.canvas.height / orthoHeight)

    def orthoX(self, x):
        return self.scaleX(x + orthoWidth // 2)

    def orthoY(self, y):
        return self.scaleY(orthoHeight - fieldHeight // 2 - y)

    def keydown(self, event):
        self.keyCode = event.keyCode

    def keyup(self, event):
        self.keyCode = None
game = Game()