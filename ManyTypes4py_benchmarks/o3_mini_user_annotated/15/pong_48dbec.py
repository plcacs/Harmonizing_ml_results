from typing import Any, List, Optional, Callable

__pragma__ ('skip')
document: Any = window: Any = Math: Any = Date: Any = 0  # Prevent complaints by optional static checker
__pragma__ ('noskip')

__pragma__ ('noalias', 'clear')

from com.fabricjs import fabric

orthoWidth: int = 1000
orthoHeight: int = 750
fieldHeight: int = 650

enter: int = 13
esc: int = 27
space: int = 32

window.onkeydown = lambda event: event.keyCode != space  # Prevent scrolldown on spacebar press

class Attribute:
    def __init__(self, game: "Game") -> None:
        self.game: Game = game                    # Attribute knows game it's part of
        self.game.attributes.append(self)         # Game knows all its attributes
        self.install()                            # Put in place graphical representation of attribute
        self.reset()                              # Reset attribute to start position
                
    def reset(self) -> None:
        # Restore starting positions or score, then commit to fabric
        self.commit()  # Nothing to restore for the Attribute base class
                
    def predict(self) -> None:
        pass
                
    def interact(self) -> None:
        pass
        
    def commit(self) -> None:
        pass

class Sprite(Attribute):  # Here, a sprite is an attribute that can move
    def __init__(self, game: "Game", width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
        super().__init__(game)
        
    def install(self) -> None:
        # The sprite holds an image that fabric can display
        self.image: Any = __new__(fabric.Rect({
            'width': self.game.scaleX(self.width), 
            'height': self.game.scaleY(self.height),
            'originX': 'center', 
            'originY': 'center', 
            'fill': 'white'
        }))
        
    __pragma__ ('kwargs')
    def reset(self, vX: int = 0, vY: int = 0, x: int = 0, y: int = 0) -> None:
        self.vX: int = vX       # Speed
        self.vY: int = vY
        
        self.x: float = x      # Predicted position, can be commit, no bouncing initially
        self.y: float = y
        
        super().reset()
    __pragma__ ('nokwargs')
        
    def predict(self) -> None:
        # Predict position, do not yet commit, bouncing may alter it
        self.x += self.vX * self.game.deltaT
        self.y += self.vY * self.game.deltaT

    def commit(self) -> None:
        # Update fabric image for asynch draw
        self.image.left = self.game.orthoX(self.x)
        self.image.top = self.game.orthoY(self.y)
        
    def draw(self) -> None:
        self.game.canvas.add(self.image)
         
class Paddle(Sprite):
    margin: int = 30  # Distance of paddles from walls
    width: int = 10
    height: int = 100
    speed: int = 400  # / s
    
    def __init__(self, game: "Game", index: int) -> None:
        self.index: int = index  # Paddle knows its player index, 0 == left, 1 == right
        super().__init__(game, self.width, self.height)
        
    def reset(self) -> None:
        # Put paddle in rest position, dependent on player index
        if self.index:
            x_val: int = orthoWidth // 2 - self.margin
        else:
            x_val: int = -orthoWidth // 2 + self.margin
        Sprite.reset(self, x=x_val, y=0)
        
    def predict(self) -> None:
        # Let paddle react on keys
        self.vY = 0
        
        if self.index:  # Right player
            if self.game.keyCode == ord('K'):  # Letter K pressed
                self.vY = self.speed
            elif self.game.keyCode == ord('M'):
                self.vY = -self.speed
        else:  # Left player
            if self.game.keyCode == ord('A'):
                self.vY = self.speed
            elif self.game.keyCode == ord('Z'):
                self.vY = -self.speed
                
        Sprite.predict(self)  # Do not yet commit, paddle may bounce with walls

    def interact(self) -> None:
        # Paddles and ball assumed infinitely thin
        # Paddle touches wall
        self.y = Math.max(self.height // 2 - fieldHeight // 2, Math.min(self.y, fieldHeight // 2 - self.height // 2))
        
        # Paddle hits ball
        if (
            (self.y - self.height // 2) < self.game.ball.y < (self.y + self.height // 2)
            and (
                (self.index == 0 and self.game.ball.x < self.x)  # On or behind left paddle
                or
                (self.index == 1 and self.game.ball.x > self.x)  # On or behind right paddle
            )
        ):
            self.game.ball.x = self.x  # Ball may have gone too far already
            self.game.ball.vX = -self.game.ball.vX  # Bounce on paddle
            self.game.ball.speedUp(self)
        
class Ball(Sprite):
    side: int = 8
    speed: int = 300  # / s
    
    def __init__(self, game: "Game") -> None:
        super().__init__(game, self.side, self.side)
 
    def reset(self) -> None:
        # Launch according to service direction with random angle offset from horizontal
        angle: float = (
            self.game.serviceIndex * Math.PI    # Service direction
            +
            (1 if Math.random() > 0.5 else -1) * Math.random() * Math.atan(fieldHeight / orthoWidth)
        )
        
        Sprite.reset(
            self,
            vX = self.speed * Math.cos(angle),
            vY = self.speed * Math.sin(angle)
        )
        
    def predict(self) -> None:
        Sprite.predict(self)  # Integrate velocity to position
        
        if self.x < -orthoWidth // 2:   # If out on left side
            self.game.scored(1)        #   Right player scored
        elif self.x > orthoWidth // 2:
            self.game.scored(0)
            
        if self.y > fieldHeight // 2:   # If it hits top wall
            self.y = fieldHeight // 2   #   It may have gone too far already
            self.vY = -self.vY          #   Bounce
        elif self.y < -fieldHeight // 2:
            self.y = -fieldHeight // 2
            self.vY = -self.vY

    def speedUp(self, bat: Paddle) -> None:
        factor: float = 1 + 0.15 * (1 - Math.abs(self.y - bat.y) / (bat.height // 2)) ** 2  # Speed will increase more if paddle hit near centre
        
        if Math.abs(self.vX) < 3 * self.speed:
            self.vX *= factor
            self.vY *= factor           

class Scoreboard(Attribute):
    nameShift: int = 75
    hintShift: int = 25
            
    def install(self) -> None:
        # Graphical representation of scoreboard are four labels and a separator line
        self.playerLabels: List[Any] = [__new__(fabric.Text('Player {}'.format(name), {
                'fill': 'white', 
                'fontFamily': 'arial', 
                'fontSize': '{}'.format(self.game.canvas.width / 30),
                'left': self.game.orthoX(position * orthoWidth), 
                'top': self.game.orthoY(fieldHeight // 2 + self.nameShift)
        })) for name, position in (('AZ keys:', -7/16), ('KM keys:', 1/16))]
        
        self.hintLabel: Any = __new__(fabric.Text('[spacebar] starts game, [enter] resets score', {
                'fill': 'white', 
                'fontFamily': 'arial', 
                'fontSize': '{}'.format(self.game.canvas.width / 70),
                'left': self.game.orthoX(-7/16 * orthoWidth), 
                'top': self.game.orthoY(fieldHeight // 2 + self.hintShift)
        }))
        
        self.image: Any = __new__(fabric.Line([
                self.game.orthoX(-orthoWidth // 2), self.game.orthoY(fieldHeight // 2),
                self.game.orthoX(orthoWidth // 2), self.game.orthoY(fieldHeight // 2)
            ],
            {'stroke': 'white'}
        ))
                
    def increment(self, playerIndex: int) -> None:
        self.scores[playerIndex] += 1
        
    def reset(self) -> None:
        self.scores: List[int] = [0, 0]
        super().reset()  # Only does a commit here
        
    def commit(self) -> None:
        # Committing labels is adapting their texts
        self.scoreLabels: List[Any] = [__new__(fabric.Text('{}'.format(score), {
                'fill': 'white', 
                'fontFamily': 'arial', 
                'fontSize': '{}'.format(self.game.canvas.width / 30),
                'left': self.game.orthoX(position * orthoWidth), 
                'top': self.game.orthoY(fieldHeight // 2 + self.nameShift)
        })) for score, position in zip(self.scores, (-2/16, 6/16))]

    def draw(self) -> None:
        for playerLabel, scoreLabel in zip(self.playerLabels, self.scoreLabels):
            self.game.canvas.add(playerLabel)
            self.game.canvas.add(scoreLabel)
            self.game.canvas.add(self.hintLabel)
        self.game.canvas.add(self.image)
        
class Game:
    def __init__(self) -> None:
        self.serviceIndex: int = 1 if Math.random() > 0.5 else 0  # Index of player that has initial service
        self.pause: bool = True    # Start game in paused state
        self.keyCode: Optional[int] = None
        
        self.textFrame: Any = document.getElementById('text_frame')
        self.canvasFrame: Any = document.getElementById('canvas_frame')
        self.buttonsFrame: Any = document.getElementById('buttons_frame')
        
        self.canvas: Any = __new__(fabric.Canvas('canvas', {'backgroundColor': 'black', 'originX': 'center', 'originY': 'center'}))
        self.canvas.onWindowDraw = self.draw  # Install draw callback, will be called asynch
        self.canvas.lineWidth = 2
        self.canvas.clear()
    
        self.attributes: List[Attribute] = []  # All attributes will insert themselves here
        self.paddles: List[Paddle] = [Paddle(self, index) for index in range(2)]  # Pass game as parameter self
        self.ball: Ball = Ball(self)
        self.scoreboard: Scoreboard = Scoreboard(self)     

        window.setInterval(self.update, 10)  # Install update callback, time in ms
        window.setInterval(self.draw, 20)    # Install draw callback, time in ms
        window.addEventListener('keydown', self.keydown)
        window.addEventListener('keyup', self.keyup)
        
        self.buttons: List[Any] = []
        
        for key in ('A', 'Z', 'K', 'M', 'space', 'enter'):
            button: Any = document.getElementById(key)
            button.addEventListener('mousedown', (lambda aKey: lambda: self.mouseOrTouch(aKey, True))(key))  # Returns inner lambda
            button.addEventListener('touchstart', (lambda aKey: lambda: self.mouseOrTouch(aKey, True))(key))
            button.addEventListener('mouseup', (lambda aKey: lambda: self.mouseOrTouch(aKey, False))(key))
            button.addEventListener('touchend', (lambda aKey: lambda: self.mouseOrTouch(aKey, False))(key))
            button.style.cursor = 'pointer'
            button.style.userSelect = 'none'
            self.buttons.append(button)
            
        self.time: float = + __new__(Date)
        
        window.onresize = self.resize
        self.resize()
        
    def install(self) -> None:
        for attribute in self.attributes:
            attribute.install()
        
    def mouseOrTouch(self, key: str, down: bool) -> None:
        if down:
            if key == 'space':
                self.keyCode = space
            elif key == 'enter':
                self.keyCode = enter
            else:
                self.keyCode = ord(key)
        else:
            self.keyCode = None
 
    def update(self) -> None:
        oldTime: float = self.time
        self.time = + __new__(Date)
        self.deltaT: float = (self.time - oldTime) / 1000.0
        
        if self.pause:  # If in paused state
            if self.keyCode == space:  # If spacebar hit
                self.pause = False  # Start playing
            elif self.keyCode == enter:  # Else if enter hit
                self.scoreboard.reset()  # Reset score
        else:  # Else, so if in active state
            for attribute in self.attributes:  # Compute predicted values
                attribute.predict()
            
            for attribute in self.attributes:  # Correct values for bouncing and scoring
                attribute.interact()
            
            for attribute in self.attributes:  # Commit them to pyglet for display
                attribute.commit()
            
    def scored(self, playerIndex: int) -> None:
        # Player has scored
        self.scoreboard.increment(playerIndex)  # Increment player's points
        self.serviceIndex = 1 - playerIndex       # Grant service to the unlucky player
        
        for paddle in self.paddles:  # Put paddles in rest position
            paddle.reset()
            
        self.ball.reset()  # Put ball in rest position
        self.pause = True  # Wait for next round
        
    def commit(self) -> None:
        for attribute in self.attributes:
            attribute.commit()
        
    def draw(self) -> None:
        self.canvas.clear()
        for attribute in self.attributes:
            attribute.draw()
             
    def resize(self) -> None:
        self.pageWidth: float = window.innerWidth
        self.pageHeight: float = window.innerHeight
        
        self.textTop: float = 0

        if self.pageHeight > 1.2 * self.pageWidth:
            self.canvasWidth: float = self.pageWidth
            self.canvasTop: float = self.textTop + 300
        else:
            self.canvasWidth = 0.6 * self.pageWidth
            self.canvasTop = self.textTop + 200

        self.canvasLeft: float = 0.5 * (self.pageWidth - self.canvasWidth)
        self.canvasHeight: float = 0.6 * self.canvasWidth

        self.buttonsTop: float = self.canvasTop + self.canvasHeight + 50
        self.buttonsWidth: float = 500
            
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
        
    def scaleX(self, x: float) -> float:
        return x * (self.canvas.width / orthoWidth)
            
    def scaleY(self, y: float) -> float:
        return y * (self.canvas.height / orthoHeight)   
        
    def orthoX(self, x: float) -> float:
        return self.scaleX(x + orthoWidth // 2)
        
    def orthoY(self, y: float) -> float:
        return self.scaleY(orthoHeight - fieldHeight // 2 - y)
                
    def keydown(self, event: Any) -> None:
        self.keyCode = event.keyCode
        
    def keyup(self, event: Any) -> None:
        self.keyCode = None 
        
game: Game = Game()  # Create and run game