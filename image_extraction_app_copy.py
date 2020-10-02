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
import parameters
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
    #white = [255, 255, 255]
    #GREEN = (0, 255, 0)
    #BLUE = (0, 0, 180)
    #RED = (255, 0, 0)
    Corners = ["Upper Left Corner", "Upper Right Corner", "Lower Left Corner", \
    "Lower Right Corner", "END - press ESC twice to exit"]
    num_images_to_load = 2
    num_corners = 4
    IMG_DIR = parameters.IMG_DIR # Enter Directory of all images
    COLOURS = parameters.COLOURS
    image_list = []

    def __init__(self):
        
        pygame.init()
        
        pygame.font.init()
        self.font_obj = pygame.font.Font('freesansbold.ttf', 50)
        #self.text_surface_obj = self.font_obj.render(self.Corners[0], True, self.GREEN, self.BLUE)
        self.text_surface_obj = self.font_obj.render(self.Corners[0], True, \
        self.COLOURS["GREEN"], self.COLOURS["BLUE"])
        self.text_rect_obj = self.text_surface_obj.get_rect()
        self.text_rect_obj.center = (300, 150)

        self._surface = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption('Image Sector Extraction')
        

        ### Load num_images_to_load images to find the gas meter corners in them for training the NN
        #self.image_array, self.image_names = image_extraction.load_images(self.num_images_to_load)
        
        self.number_images = self.num_images_to_load

        
        try:
            image_name_file = open("imagenamelist.pck", "rb")
            self.image_list = pickle.load(image_name_file)
            num_images_loaded = len(self.image_list)
            self.image_array, self.image_names = image_extraction.load_images(num_images_loaded, self.num_images_to_load)
        except IOError:
            image = self.image_array[0]
            self.image_array, self.image_names = image_extraction.load_images(0, self.num_images_to_load)
            print("File imagenamelist.pck not accessible")
        finally:
            image_name_file.close()


        ### Creating the array for storing corner coordinates
        self.corner_array = np.zeros((self.number_images, self.num_corners, 2))
        '''
        ### Load num_images_to_load images to find the gas meter corners in them for training the NN
        self.image_array, self.image_names = image_extraction.load_images(self.num_images_to_load)
        
        self.number_images = len(self.image_array)

        ### Creating the array for storing corner coordinates
        self.corner_array = np.zeros((self.number_images, self.num_corners, 2))
        '''
        image = self.image_array[0]
        self.scale = np.size(image, 1)/self.width
        self.show_image(image)
        self._surface.blit(self.text_surface_obj, self.text_rect_obj)
        pygame.display.flip()
        print(self.image_array.shape)
        self.run()

    def show_image(self, image):
        '''
        Prepare input image for blitting on pygame surface.
        '''
        new_image = pygame.surfarray.make_surface(image)
        flipped_image = pygame.transform.flip(new_image, False, True)
        rotated_image = pygame.transform.rotate(flipped_image, 270)
        resized_image = pygame.transform.scale(rotated_image, (self.width, self.height))
        self._surface.blit(resized_image, (0, 0))

    def save_data(self, data, filename):
        '''
        Pickles the array a list of data in a pickle file.
        
        input: list to be pickled and filename to pickle list in
        output: pickled file "filename.pck"

        '''
        
        pickle_file = filename + ".pck"
        data_file = open(pickle_file, "rb+")
        try:
            old_data = pickle.load(data_file)
            #data_list = old_list.append(data_list)
            if isinstance(old_data, list):
                data =  old_data.append(data)
            if isinstance(old_data, np.ndarray):
                data =  np.concatenate((old_data,data))
        except IOError:
            print("File does not exist yet")
        finally:
            pickle.dump(data, data_file)
            data_file.close()


    def run(self):
        '''
        Main Class execution.
        '''
        pr = cProfile.Profile()
        pr.enable()
        running = True
        showing_images = False
        corner_counter = 0
        image_counter = 0
        #self.corner_list = []
        #self.image_list = []
        #complete_images_list = []
        while running:

            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    running = False
                    #print(corner_list)
                    pygame.quit()
                    break
                if i.type == pygame.MOUSEBUTTONDOWN and i.button == 1 and \
                image_counter in range(0, self.num_images_to_load+1) and \
                corner_counter in range(0, self.num_corners):
                    mouse_x, mouse_y = i.pos
                    print(mouse_x, mouse_y)
                    self.corner_array[image_counter, corner_counter] = i.pos
                    #self.corner_list.append(i.pos)
                    self.image_list.append(self.image_names[image_counter])
                    image_counter += 1
                    if image_counter == self.num_images_to_load:
                        image_counter = 0
                        corner_counter += 1
                        if corner_counter%2:
                            self.text_surface_obj = \
                            self.font_obj.render(self.Corners[corner_counter], \
                            True, self.COLOURS["GREEN"], self.COLOURS["RED"])
                        else:
                            self.text_surface_obj = \
                            self.font_obj.render(self.Corners[corner_counter], \
                            True, self.COLOURS["GREEN"], self.COLOURS["BLUE"])
                    print(self.image_array.shape)
                    print(image_counter)
                    image = self.image_array[image_counter]
                    self.show_image(image)
                    self._surface.blit(self.text_surface_obj, self.text_rect_obj)
                    pygame.display.flip()

   
                if i.type == pygame.KEYDOWN:
                    if i.key == pygame.K_d:
    # with key "d": delete previously entered digit value and reset image
                       # if digit_counter > 0:
                       #     digit_counter -= 1
                       #     del digits_list[-1]
                       # elif image_counter > 0:
                       #     digit_counter = 6
                       #     image_counter -= 1
                       #     self.image_list = complete_images_list[-1]
                       #     digits_list = self.image_list[1]
                       #     del digits_list[-1]
                       #     del complete_images_list[-1]


                        #digit = self.numbers_list[image_counter, digit_counter] * 256
                        #self.showDigit(digit)
                        #pygame.display.flip()
                        pass

                    if i.key == pygame.K_s:
    # save complete images list as pickled file
                        #filename = 'cornerlist.pck'
                        #outfile = open(filename, 'wb')
                        #pickle.dump(np.round(self.corner_array*self.scale), outfile)
                        #outfile.close()
                        self.save_data(np.round(self.corner_array*self.scale),'cornerlist')

                        #filename = 'imagenamelist.pck'
                        #outfile = open(filename, 'wb')
                        #pickle.dump(self.image_list, outfile)
                        #outfile.close()
                        self.save_data(self.image_list,'imagenamelist')

                    if i.key == pygame.K_o:
    # open pickled file and show images with coordinates marked
                        if not showing_images:
                            showing_images = True
                            self.corner_array = pickle.load(open("cornerlist.pck", "rb"))
                            self.corner_array=np.round(self.corner_array/self.scale)
                            self.image_list = pickle.load(open("imagenamelist.pck", "rb"))
                            self.number_images = self.corner_array.shape[0]
                            #self.number_images = len(self.image_list)
                            self.image_array, self.image_names = image_extraction.load_images(0,self.num_images_to_load)
                           
                            
                            self.image_counter = 0
                            image = self.image_array[self.image_counter]
                            cornered_image = image.astype(int)
                            
                            cornered_image[self.corner_array[self.image_counter, :, 1].astype(int), self.corner_array[self.image_counter, :, 0].astype(int)]=self.COLOURS["GREEN"]
                            
                            self.show_image(cornered_image)
                            
                            pygame.display.flip()

                    if i.key == pygame.K_DOWN:
    # show next image with coordinates marked
                        if showing_images:
                            
                            if self.image_counter in range(0, self.number_images-1):
                                self.image_counter += 1
                                image = self.image_array[self.image_counter]
                                cornered_image = image.astype(int)
                                cornered_image[self.corner_array[self.image_counter, : , 1].astype(int), self.corner_array[self.image_counter, : , 0].astype(int)]=self.COLOURS["GREEN"]
                               

                                self.show_image(cornered_image)
                                
                                pygame.display.flip()

                        
                        

# closing program with EXCAPE
# requires pressing ESC twice (for whatever reason???)
                    if i.key == pygame.K_ESCAPE:
                        #print(self.corner_list)
                        print(self.corner_array)
                        print(np.round(self.corner_array*self.scale))
                        running = False

        time.sleep(100.0 / 1000.0)
        pr.disable()

        #pr.print_stats(sort='time')

if __name__ == "__main__":
    App().run()
