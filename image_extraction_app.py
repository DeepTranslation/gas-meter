#import gas_meter
#import image_preparation
#import contour_manipulation

#import CNN
#import view
#import rounds
import cProfile

import time
#import timeit
import pickle
import numpy as np
#import cv2

import pygame
import image_extraction


class App:
    '''
    Program to train and use a neural network to find the actual gas meter section from
    within a larger image

    Create training values
        Load Images in sets of 20
            for each corner seperately
                Show images one by one and mark corner points of the gas meter section with a
                    mouse click
                store coordinates in list
                list dimensions: number of images x 4 corners x 2 point coordinates (x,y)

    Set up neural network

    Train NN

    Use NN'''

    width = 1280
    height = 960
    white = [255, 255, 255]
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 180)
    RED = (255, 0, 0)
    Corners = ["Upper Left Corner", "Upper Right Corner", "Lower Left Corner", \
    "Lower Right Corner", "END"]

    num_images_to_load = 2
    num_corners = 4

    def __init__(self):
        #self.view = view.View()
        pygame.init()
        pygame.font.init()
        self.font_obj = pygame.font.Font('freesansbold.ttf', 32)
        self.text_surface_obj = self.font_obj.render(self.Corners[0], True, self.GREEN, self.BLUE)
        self.text_rect_obj = self.text_surface_obj.get_rect()
        self.text_rect_obj.center = (200, 150)

        self._surface = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        self._surface.fill(self.white)
        #self.background = pygame.Surface(self._surface.get_size())
        #self.background = self.background.convert()
        #self.background.fill((250, 250, 250))
        pygame.display.set_caption('Image Sector Extraction')


        ### Load num_images_to_load images to find the gas meter corners in them for training the NN
        self.image_array, self.image_names = image_extraction.loadImages(self.num_images_to_load)

        self.number_images = len(self.image_array)

        ### Creating the array
        self.corner_array = np.zeros((self.number_images, self.num_corners, 2))
        print(self.corner_array.shape)

        print(self.number_images)
        image = self.image_array[0]
        #
        self.show_image(image)
        self._surface.blit(self.text_surface_obj, self.text_rect_obj)
        pygame.display.flip()
        self.run()

    def show_image(self, image):
        '''
        Prepare input image for blitting on pygame surface.
        '''
        new_image = pygame.surfarray.make_surface(image)
        flipped_image = pygame.transform.flip(new_image, False, True)
        rotated_image = pygame.transform.rotate(flipped_image, 270)
        resized_image = pygame.transform.scale(rotated_image, (self.width, self.height))
        #self._surface.blit(resized_image,(100, 100))
        self._surface.blit(resized_image, (0, 0))

    def run(self):
        '''
        Main Class execution.
        '''
        pr = cProfile.Profile()
        pr.enable() 
        running = True
        corner_counter = 0
        image_counter = 0
        self.corner_list = []
        self.image_list = []
        complete_images_list = []
        while running:
            
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    running = False
                    #print(corner_list)
                    pygame.quit()
                    break
                if i.type == pygame.MOUSEBUTTONDOWN and i.button == 1 and image_counter in range(0,self.num_images_to_load+1) and corner_counter in range(0,self.num_corners):
                    mouse_x,mouse_y= i.pos
                    print(mouse_x,mouse_y)
                    self.corner_list.append(i.pos)
                    self.image_list.append(self.image_names[image_counter])
                    image_counter += 1
                    if image_counter == self.num_images_to_load:
                        image_counter=0
                        corner_counter+=1
                        if corner_counter%2:
                            self.text_surface_obj = self.font_obj.render(self.Corners[corner_counter], True, self.GREEN, self.RED)
                        else: 
                            self.text_surface_obj = self.font_obj.render(self.Corners[corner_counter], True, self.GREEN, self.BLUE)


                        
                    image = self.image_array[image_counter] 
                    self.show_image(image)
                    self._surface.blit(self.text_surface_obj, self.text_rect_obj)
                    pygame.display.flip()
                    '''
                    Load Images in sets of 20
                        for each corner seperately
                        Show images one by one and mark corner points of the gas meter section with a mouse click
                        store coordinates in list
                    '''

                if i.type == pygame.KEYDOWN :
                    if i.key == pygame.K_d:
    # with key "d": delete previously entered digit value and reset image
                        if digit_counter >0:
                            digit_counter -= 1
                            del digits_list[-1]  
                        elif image_counter >0:
                            digit_counter = 6
                            image_counter -= 1
                            self.image_list = complete_images_list[-1]
                            digits_list = self.image_list[1]
                            del digits_list[-1]
                            del complete_images_list[-1]   


                        digit = self.numbers_list[image_counter,digit_counter] * 256  
                        self.showDigit(digit)
                        pygame.display.flip()
                        
                    if i.key == pygame.K_s:
    # save complete images list as pickled file
                        filename = 'cornerlist.pck'
                        outfile = open(filename,'wb')
                        pickle.dump(self.corner_list,outfile)
                        outfile.close()
                        filename = 'imagenamelist.pck'
                        outfile = open(filename,'wb')
                        pickle.dump(self.image_list,outfile)
                        outfile.close()
                    
                    if i.key == pygame.K_o:
    # save complete images list as pickled file ?????

                        #keys = pygame.key.get_pressed()

# closing program with EXCAPE
                    if i.key == pygame.K_ESCAPE:
                        print(self.corner_list)
                        running = False
                        
                        #pygame.quit()                   
                        #break
                    
                    
                    
    
        time.sleep (100.0 / 1000.0)
        pr.disable()
 
        #pr.print_stats(sort='time')

if __name__ == "__main__":
    App().run()
