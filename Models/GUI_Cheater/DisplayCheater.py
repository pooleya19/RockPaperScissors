import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import pygame.camera
from pygame.locals import *
import time
import numpy as np
from PIL import Image


class DisplayCheater:
    def __init__(self, predictPicture):
        self.predictPicture = predictPicture

        pygame.init()
        self.running = True

        # ===== Parameters =====
        # Pygame
        self.width_screen = 600
        self.height_screen = 300
        self.maxRawPictureSize = 600

        self.color_background = (100,100,100)
        self.color_camera_background = (255,255,255,255)

        # Camera
        self.cameraIndex = 0
        self.cameraResolution_base = (640,480)
        self.cameraResolution_final = (300,300)
        self.useMask = False

        # ===== Init Pygame window =====
        self.screen = pygame.display.set_mode((self.width_screen, self.height_screen))
        pygame.display.set_caption("RockPaperScissors Data Collector")

        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.color_background)

        # ===== Init camera =====
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()
        if len(cameras) == 0:
            print("No cameras detected! Closing program...")
            self.quit()
            return
        else:
            print("Detected cameras:")
            for camera in cameras:
                print("\t", str(camera), sep='')

            print("Opening camera ", self.cameraIndex, ": ", cameras[self.cameraIndex], sep='')
            self.camera = pygame.camera.Camera(cameras[self.cameraIndex], self.cameraResolution_base)
            self.camera.start()
            self.image = None
          
        self.background_camera = pygame.Surface(self.cameraResolution_final)
        self.background_camera = self.background_camera.convert_alpha()
        self.background_camera.fill(self.color_camera_background)
        
        # ===== Create mask =====
        self.cameraMask = pygame.Surface(self.cameraResolution_final).convert_alpha()
        self.cameraMask.fill((0,0,0,0))
        pygame.draw.circle(self.cameraMask, (255,255,255,255), (self.cameraResolution_final[0]/2, self.cameraResolution_final[1]/2), self.cameraResolution_final[0]/2, 0)

        # ===== Game =====
        self.playing = False
        self.playingStartTime = 0
        self.loadTime = 1
        self.gameDelay = 1

    def update(self):
        self.handleEvents()
        self.updateCamera()
        self.draw()
        pygame.display.update()
        time.sleep(0.001)

    def handleEvents(self):
        # Check if program is alive
        if not self.running:
            return
        
        for event in pygame.event.get():
            if event.type == QUIT:
                print("Attempted Quit")
                self.quit()
            elif event.type == KEYDOWN:
                if self.getKey(K_ESCAPE):
                    print("Escape key pressed; Aborting.")
                    self.quit()
                elif self.getKey(K_SPACE):
                    self.startGame()
    
    def startGame(self):
        if self.playing:
            # Game already playing
            return

        self.playing = True
        self.playingStartTime = time.time()

    def convertSurfaceToNumpy(self, surface):
        npImage = np.frombuffer(self.image.get_buffer(), dtype=np.uint8).reshape((self.cameraResolution_final[0], self.cameraResolution_final[1],-1))
        return npImage
                    
    def makePredictionFunc(self):
        if self.image is None:
            return
        
        npImage = self.convertSurfaceToNumpy(self.image)
        npImageGray = np.uint8(0.2989*npImage[:,:,0] + 0.5870*npImage[:,:,1] + 0.1140*npImage[:,:,2])
        image_down = np.array(Image.fromarray(npImageGray, "L").resize((150,150)))
        Image.fromarray(image_down, "L").save("Temp.jpg")
        image_flat = image_down.reshape((1,-1))

        predictFunc = self.predictPicture
        prediction, certainties = predictFunc(image_flat)
        # print("PREDICTION: ", prediction)

        self.prediction = prediction
        self.certainties = certainties


    def updateCamera(self):
        if not self.camera.query_image():
            # Camera frame not ready
            return
        
        image_raw = self.camera.get_image()
        squareDiff = (self.cameraResolution_base[0] - self.cameraResolution_base[1])/2
        image_square = image_raw.subsurface(squareDiff, 0, self.cameraResolution_base[1], self.cameraResolution_base[1])
        image_square_copy = image_square.copy()
        image_final = pygame.transform.scale_by(image_square_copy, self.cameraResolution_final[1]/self.cameraResolution_base[1])
        
        if self.image is None or self.image.get_size() != self.cameraResolution_final:
            self.image = pygame.Surface(self.cameraResolution_final)
            self.image = self.image.convert_alpha()
        
        self.image.blit(self.background_camera, (0,0))
        if self.useMask:
            maskedImage = image_final.copy().convert_alpha()
            maskedImage.blit(self.cameraMask, (0,0), None, pygame.BLEND_RGBA_MULT) # Apply mask
            self.image.blit(maskedImage, (0,0))
        else:
            self.image.blit(image_final, (0,0))

    def draw(self):
        # Check if program is alive
        if not self.running:
            return
        
        # Draw background
        self.screen.blit(self.background, (0, 0))

        # Camera
        if self.image is not None:
            self.screen.blit(self.image, (0, 0))

        # Game
        currentTime = time.time()
        elapsedTime = currentTime - self.playingStartTime
        elapsedTimePercent = min(100, round(elapsedTime/self.loadTime*100,0))
        textSurface_loadingGame = getTextSurface("Loading Game: " + str(elapsedTimePercent) + "%", size=30)
        gameStartText = ""
        if elapsedTime >= self.loadTime:
            gameStartText += "Rock"
        if elapsedTime >= self.loadTime + self.gameDelay:
            gameStartText += ", Paper"
        if elapsedTime >= self.loadTime + 2*self.gameDelay:
            gameStartText += ", Scissors"
        if elapsedTime >= self.loadTime + 3*self.gameDelay:
            gameStartText += ", Shoot!"
            self.playing = False
        textSurface_gameStart = getTextSurface(gameStartText, size=25)
        

        self.screen.blit(textSurface_loadingGame, (300,0))
        self.screen.blit(textSurface_gameStart, (300, 100))
    
    def saveImage(self, imageName):
        imagePath = self.path_imageFolder + "/" + imageName + "_" + str(self.nextImageNumber) + ".jpg"
        self.nextImageNumber += 1

        print("Saving image ", imagePath, ".", sep='')
        pygame.image.save(self.image, imagePath)
        # image = Image.open(pygame.image.tobytes(self.image))
        # image.save(imagePath)

    def getKey(self, key):
        return pygame.key.get_pressed()[key]
    
    def quit(self):
        print("Stopping program...")
        self.running = False

import pygame.font
from pygame.locals import Color

def getTextSurface(text, size=50, color=None):
    font = pygame.font.SysFont("Calibri",size)
    if(color==None):
        textColor = Color(0,0,0)
    else:
        textColor = color
    antialias = False
    return font.render(text,antialias,textColor)