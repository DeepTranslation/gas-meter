
#import gas_meter
import image_preparation
import contour_manipulation
import gas_meter
#import CNN
#import view
#import rounds

import pygame
import cProfile
import time
import timeit
import numpy as np
import cv2


class App:
    width = 1280
    height = 960
    tileWidth = 20
    tileHeight = 20
    SAND = (194, 178, 128)
    GRASS = (124, 252, 0)
    white = [255, 255, 255]
    rounds = 5
    loops = 100

    def __init__(self):
        #self.view = view.View()
        pygame.init()
        pygame.font.init()
        self._surface = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE)
        self._surface.fill(self.white)
        self.background = pygame.Surface(self._surface.get_size())
        self.background = self.background.convert()
        self.background.fill((250, 250, 250))
        pygame.display.set_caption('Meter Detection')
        
        pygame.draw.rect(self._surface, self.SAND,
                    [ self.width-self.tileWidth*1.5,
                      self.tileHeight*0.5,
                      self.tileWidth,
                      self.tileHeight])

        self.image_array = gas_meter.loadImages()
        #print(image_array.shape)
        self.numbers_list = gas_meter.getDigits(self.image_array)
        self.current_digit =0
        #print(numbers_list.shape)   
        digit = self.numbers_list[0,0] * 256
        #print(digit)
        #screen_image = cv2.cvtColor(digit, cv2.COLOR_RGB2BGR)
        #
        self.showDigit(digit)
        '''
        new_image =pygame.surfarray.make_surface(digit)
        flipped_image = pygame.transform.flip(new_image,False, True)
        rotated_image = pygame.transform.rotate(flipped_image, 270)
        resized_image = pygame.transform.scale(rotated_image, (digit.shape[1]*4, digit.shape[0]*4))
        self._surface.blit(resized_image,(100, 100)) '''
        pygame.display.flip()
        self.run()
    
    def showDigit(self, digit):
        new_image =pygame.surfarray.make_surface(digit)
        flipped_image = pygame.transform.flip(new_image,False, True)
        rotated_image = pygame.transform.rotate(flipped_image, 270)
        resized_image = pygame.transform.scale(rotated_image, (digit.shape[1]*4, digit.shape[0]*4))
        self._surface.blit(resized_image,(100, 100)) 

    def run(self):
        pr = cProfile.Profile()
        pr.enable() 
        '''
        for i in range(self.rounds):
            #self.world = world.World(self.width, self.height)

             
            round = rounds.Round(i, self.loops)
            round.on_execute()
        '''
        first_execution = True
        running = True
        counter = 0
        while running:
            key = 0
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    break

                #if i.type == pygame.KEYDOWN and not first_execution:
                 #   if i.key in range(pygame.K_KP0,pygame.K_KP9+1):
                        

            keys = pygame.key.get_pressed()
            if (keys[pygame.K_ESCAPE]):
                

                # for img_num in number of images:
                #   for digit_num in number of digits:
                #       show image
                #       read key from keyboard
                #       store img_num, digit arrray and target digit value in array


                running = False
                pygame.quit()
            '''
            if pygame.font:
                #font = pygame.font.SysFont('arial', 45)
                #font = pygame.font.get_default_font()
                #theFont = pygame.font.SysFont("Arial", 40, False, False)
                theFont = pygame.font.Font(None, 20)
                text = theFont.render("blablabla", 1, (10, 10, 10))
                textpos = text.get_rect(centerx=self.background.get_width()/2)

                self.background.blit(text, textpos)
            '''
            if (keys[pygame.K_l])and first_execution:
                while counter < 6:
                    
                    digit = self.numbers_list[0,counter+1] * 256
                    self.showDigit(digit)
                    pygame.display.flip()
                    counter += 1
                    time.sleep (1000.0 / 1000.0)
                '''
                
                pygame.display.update()'''
            #if running:
                #pygame.display.flip()
            #if i in xrange((keys[pygame.K_KP0])

        time.sleep (100.0 / 1000.0)
        pr.disable()
 
        #pr.print_stats(sort='time')

if __name__ == "__main__":
    App().run()
