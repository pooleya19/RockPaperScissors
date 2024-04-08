import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import pygame.camera
from pygame.locals import *
import time
import numpy as np
from PIL import Image


class DataTester:
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

        # ===== Predictions =====
        self.certainties = [0,0,0]
        self.prediction = "None lmao"
        self.lastPredictionTime = 0
        self.predictionRate = 10 # Hz

    def update(self):
        currentTime = time.time()
        if currentTime - self.lastPredictionTime >= 1/self.predictionRate:
            self.lastPredictionTime = currentTime
            self.makePredictionFunc()

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
                    self.makePredictionFunc()

    def convertSurfaceToNumpy(self, surface):
        # imagePath = "Temp.jpg"
        # pygame.image.save(self.image, imagePath)
        # npImage = np.asarray(Image.open(imagePath))
        # npImage = pygame.surfarray.array3d(self.image)
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

        # Predictions
        textSurface_CertaintiesTitle = getTextSurface("Certainties:", size=50)
        textSurface_RockTitle = getTextSurface("Rock:", size=30)
        textSurface_PaperTitle = getTextSurface("Paper:", size=30)
        textSurface_ScissorsTitle = getTextSurface("Scissors:", size=30)
        textSurface_PredictionTitle = getTextSurface("Prediction:", size=50)

        textSurface_RockCertainty = getTextSurface(str(np.round(self.certainties[0],3)), size=30)
        textSurface_PaperCertainty = getTextSurface(str(np.round(self.certainties[1],3)), size=30)
        textSurface_ScissorsCertainty = getTextSurface(str(np.round(self.certainties[2],3)), size=30)
        textSurface_Prediction = getTextSurface(str(self.prediction), size=30)

        textX1 = 350
        textX2 = 470
        self.screen.blit(textSurface_CertaintiesTitle,      (textX1,  0))
        self.screen.blit(textSurface_RockTitle,             (textX1, 50))
        self.screen.blit(textSurface_PaperTitle,            (textX1,100))
        self.screen.blit(textSurface_ScissorsTitle,         (textX1,150))
        self.screen.blit(textSurface_PredictionTitle,       (textX1,200))

        self.screen.blit(textSurface_RockCertainty,         (textX2, 50))
        self.screen.blit(textSurface_PaperCertainty,        (textX2,100))
        self.screen.blit(textSurface_ScissorsCertainty,     (textX2,150))
        self.screen.blit(textSurface_Prediction,            (textX1,250))
    
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