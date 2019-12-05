
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


class App:
    width = 640
    height = 480
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
        self._surface = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE)
        self._surface.fill(self.white)
        pygame.display.set_caption('Meter Detection')
        
        pygame.draw.rect(self._surface, self.SAND,
                    [ self.width-self.tileWidth*1.5,
                      self.tileHeight*0.5,
                      self.tileWidth,
                      self.tileHeight])
        
        self.run()

    def run(self):
        pr = cProfile.Profile()
        pr.enable() 
        '''
        for i in range(self.rounds):
            #self.world = world.World(self.width, self.height)

             
            round = rounds.Round(i, self.loops)
            round.on_execute()
        '''

        running = True
        while running:
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    break
            keys = pygame.key.get_pressed()
            if (keys[pygame.K_ESCAPE]):
                
                running = False
                pygame.quit()
            if (keys[pygame.K_l]):
                image_array = gas_meter.loadImages()
                print(image_array.shape)

            if running:
                pygame.display.flip()

        time.sleep (100.0 / 1000.0);
        pr.disable()
 
        pr.print_stats(sort='time')

if __name__ == "__main__":
    App().run()
