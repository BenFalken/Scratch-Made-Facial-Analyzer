#### PROGRAM MECHANICS ####

# Step One: Create Image Using Pygame Window (In this case, a face)
## Press any button to exit and finish drawing
## Saves Image
# Step Two: Look for Component Images (ie: Eyes, Nose)
## Selenium driver looks on internet for Component Images
## Component Images saved
# Step Three: Find Python Image Proportions
## Take random slices of Pygame Face Image
## Pair with approproate Component Images and run through neural network
## If a match between Component Image and slice of Drawing is confirmed, record the coordinates of the element
## Repeat until all elements found - if not, exit
# Step Four
##Take pool of images similar (in name) to the Created Pygame Image
## Put through above criteria for the Pygame Face Image
## Average out the resultant lists and compare with Pygame Face Image list
## Print out suggestions as to where to fix positioning of Components (Proportioning) (Note that the list references the distance between the eyes as the first element)

#### START PROGRAM ####

import os, time, sys, json, requests, certifi, keras, ast
import matplotlib.pyplot as plt
import numpy as np
import pygame as py
from math import sqrt
from random import randint as rand
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Input, Dropout
from keras.layers.recurrent import GRU
from keras import backend as K
from scipy import spatial
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image

# Imports #

os.environ["PATH"] += os.pathsep + os.getcwd()
download_path = "dataset/"

White = (255, 255, 255)
Black = (0, 0, 0)

# Declare Starter variables #

LastPos=(0,0)

class ProgramDefaults:
    def __init__(self, ImageSize, NetworkSamples, ScreenSize, SearchFeatures, MinPixels, ConvertType):
        self.ImageSize = ImageSize
        self.NetworkSamples = NetworkSamples
        self.ScreenSize = ScreenSize
        self.SearchFeatures = SearchFeatures
        self.MinPixels = MinPixels
        self.ConvertType = ['1', 'L'][ConvertType]
    def Empty(self):
        return [None for Index in range(self.SearchFeatures)]
    def RatioCompatibility(self, InputFile):
        print('Ratio:' + (self.ImageSize/InputFile.size[0]))
        if abs((self.ImageSize/InputFile.size[0])-1) < 0.5 and abs((self.ImageSize/InputFile.size[1])-1) < 0.5:
            return True
    def Quarterize(self, InputFile):
        return InputFile.crop((InputFile.size[0]/4, 
        InputFile.size[1]/4, 
        3*InputFile.size[0]/4, 
        3*InputFile.size[0]/4))
    def ReduceList(self, FeatureList, RequiredSize):
        ErrorRad = self.ImageSize/10
        while len(FeatureList) > RequiredSize:
            for Item in FeatureList:
                for Comparison in [FeatureList[Index] for Index in range(len(FeatureList)) if FeatureList[Index] != Item]:
                    DistanceDifference = sqrt(((Item[0]-Comparison[0])**2)+((Item[1]-Comparison[1])**2)) 
                    if DistanceDifference < ErrorRad:
                        FeatureList.append(((Item[0]+Comparison[0])/2, (Item[1]+Comparison[1])/2))
                        del FeatureList[FeatureList.index(Item)]
                        del FeatureList[FeatureList.index(Comparison)]
                        print('Revised list: ' + str(FeatureList))
            ErrorRad += 2
        if RequiredSize == 2:
            try:
                return FeatureList[0], FeatureList[1]
            except:
                return FeatureList[0]
        return FeatureList[0]
    def Expand(self, Quantity):
        return ((Quantity**2)+Quantity)/2


StandardDefault = ProgramDefaults(ImageSize=100, NetworkSamples=100, ScreenSize=500, 
                                SearchFeatures=3, MinPixels=(100**2)/3, ConvertType=1)

# Make Specific Default Modes #

def DrawFace():
    MouseDown, Running = False, True
    DrawNewPic = input('Draw New? (Yes or No): ')
    if DrawNewPic == 'Yes':
        Screen = py.display.set_mode((StandardDefault.ScreenSize, StandardDefault.ScreenSize))
        Screen.fill(White)
        Clock = py.time.Clock()
        py.init()
        while Running:
            for event in py.event.get():
                pos = py.mouse.get_pos()
                if event.type == py.MOUSEBUTTONDOWN:
                    MouseDown = True
                    py.draw.ellipse(Screen, Black, (pos[0], pos[1], 5, 5))
                    LastPos = pos
                elif event.type == py.MOUSEBUTTONUP:
                    MouseDown = False
                if event.type == py.MOUSEMOTION and MouseDown == True:
                    for Index in range(1, abs(LastPos[0]-pos[0])):
                        py.draw.ellipse(Screen, Black, (LastPos[0]+(Index/abs(Index)), 
                        LastPos[1]+((LastPos[1]-pos[1])/(LastPos[0]-pos[0])), 5, 5))
                    py.draw.ellipse(Screen, Black, (pos[0], pos[1], 5, 5))
                if event.type == py.KEYDOWN:
                    Img = py.image.save(Screen, 'FacePic.jpg')
                    py.display.quit()
                    py.quit()
                    return Img
            Clock.tick(120)
            py.display.update()
    return None

# Return Pygame Display Face Image #

def Network_train(Possible_Features):
    global StandardDefault

    DataLen = StandardDefault.SearchFeatures*StandardDefault.NetworkSamples

    FeatureInputs=np.zeros((DataLen, StandardDefault.ImageSize**2))
    FeatureOutputs=np.zeros((DataLen, 
        StandardDefault.SearchFeatures))
    ListItem = 0

    print('Network is Training...')

    for Index in range(StandardDefault.SearchFeatures):
        for Item in Possible_Features[Index]:
            try:
                FeaturePhoto = Image.open(open(Item))
                FeaturePhoto = StandardDefault.Quarterize(FeaturePhoto)
                FeaturePhoto = Compile(FeaturePhoto, StandardDefault.ImageSize, 
                    StandardDefault.ImageSize)
                for Element in range(StandardDefault.ImageSize**2):
                    FeatureInputs[ListItem][Element] = FeaturePhoto[Element]
                FeatureOutputs[ListItem][Index] = 1
                ListItem+=1
            except:
                #print("FAILED")
                continue

    FeatureOutputs=np.array([FeatureOutputs[Item] for Item in range(DataLen) if np.sum(FeatureInputs[Item]) > 255*StandardDefault.MinPixels]) # NEW FUNCTION
    FeatureInputs=np.array([FeatureInputs[Item] for Item in range(DataLen) if np.sum(FeatureInputs[Item]) > 255*StandardDefault.MinPixels])

    InputLayer = Input(shape=(StandardDefault.ImageSize**2,), name='InputLayer')
    ReshapeLayer = Reshape((StandardDefault.ImageSize, StandardDefault.ImageSize, 1), name='ReshapeLayer')(InputLayer)

    Conv2DLayer = Conv2D(kernel_size = (5,5), filters = 400, activation='relu', name='ConvolutionalLayer')(ReshapeLayer)
    MaxPooling2DLayer = MaxPooling2D(pool_size = (2,2), strides=(2,2), name='MaxPoolingLayer')(Conv2DLayer)

    DropoutLayer = Dropout(0.25, name='DropoutLayer')(MaxPooling2DLayer)
    FlattenLayer = Flatten(name='FlattenLayer')(DropoutLayer)

    DenseLayer1 = Dense(128, name='FirstDenseLayer')(FlattenLayer)
    DenseLayer2 = Dense(64, name='SecondDenseLayer')(DenseLayer1)
    FinalLayer = Dense(StandardDefault.SearchFeatures, activation='softmax', name='FinalDenseLayer')(DenseLayer2)

    TrainedModel = Model(InputLayer, FinalLayer)
    TrainedModel.summary()

    TrainedModel.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    TrainedModel.fit(FeatureInputs, FeatureOutputs, batch_size=None, epochs=5, verbose=2)
    return TrainedModel

# Train Neural Network #

def Network_test(File, Crop_Coordinates, TrainedModel): # MAKE GRU LAYER
    global StandardDefault

    Averaged_x, Averaged_y = Crop_Coordinates[0], Crop_Coordinates[1]
    File = File.crop((float(Averaged_x-(StandardDefault.ImageSize/2)), float(Averaged_y-(StandardDefault.ImageSize/2)), 
    float(Averaged_x+(StandardDefault.ImageSize/2)), float(Averaged_y+(StandardDefault.ImageSize/2)))) 
    #File.show()

    PlaceHolder = np.zeros((1,File.size[0]*File.size[1]))
    PlaceHolder[0] = Compile(File, StandardDefault.ImageSize, StandardDefault.ImageSize)
    print(np.sum(PlaceHolder[0]))/(StandardDefault.ImageSize**2)

    [Prediction]=[Item for Item in range(3) if int(TrainedModel.predict(PlaceHolder)[0][Item])==1]

    print(str(TrainedModel.predict(PlaceHolder)[0][Item]))
    print('Prediction: ' + str(Prediction))

    return (Prediction)

# Test Neural Network #

def CollectImages(searchtext, num_requested):
    print('')
    print('Looking for: ' + searchtext)

    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    extensions = {"jpg", "jpeg", "png"}
    img_count = 0
    downloaded_img_count = 0

    if os.path.isfile("File_"+str(StandardDefault.NetworkSamples/10)+searchtext+".jpg") != True:
        number_of_scrolls = num_requested / 5 + 1 
        Imges_saved=[]

        if not os.path.exists(download_path + searchtext.replace(" ", "_")):
            os.makedirs(download_path + searchtext.replace(" ", "_"))

        url = "https://www.google.co.in/search?q="+searchtext+"&source=lnms&tbm=isch"
        driver = webdriver.Safari()
        driver.get(url)

        for _ in xrange(number_of_scrolls):
            for _ in xrange(10):
                driver.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(0.2)
            time.sleep(0.5)
            try:
                driver.find_element_by_xpath("//input[@value='Show more results']").click()
            except Exception as e:
                print('Less images found:', e)
                break

        imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
        print('Total images:' + str(len(imges)))
        for img in imges:
            if downloaded_img_count >= num_requested:
                break
            img_count += 1
            img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
            img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
            print("Downloading Image: " + str(img_count)) #, ": ", img_url
            print('')
            if img_type not in extensions:
                img_type = "jpg"
            img_file = requests.get(img_url, stream=True)
            if img_file.status_code == 200 and searchtext == "Front_View_Face_Drawing":
                with open("File_"+str(img_count)+".jpg", 'wb') as File:
                    print('Reading File...')
                    for chunk in img_file:
                        File.write(chunk)
                with open("File_"+str(img_count)+".jpg", 'rb') as File:
                    print('Writing File...')
                    File = Image.open(File)
                    if File.size[0] < File.size[1]:
                        Imges_saved.append(Read(File, ScreenSize=StandardDefault.ScreenSize, PositionList=StandardDefault.Empty()))
                        File.close()
            elif img_file.status_code == 200 and searchtext != 'Front_View_Face_Drawing':
                with open("File_"+str(img_count)+searchtext+".jpg", 'wb') as File:
                    for chunk in img_file:
                        File.write(chunk)
                    File.close()
                Imges_saved.append("File_"+str(img_count)+searchtext+".jpg")
            downloaded_img_count += 1

        print('Total downloaded: ' + str(downloaded_img_count) + '/' + str(img_count))

        driver.quit()
    else:
        Imges_saved = ["File_" + str(img_count+1) + searchtext + ".jpg" for img_count in range(num_requested)]
    return Imges_saved

# Find Images Online with Selenium #

def Compile(File, Width, Height):
    File = File.convert('RGB')
    File = File.convert(StandardDefault.ConvertType)
    File = File.resize((Width, Height), Image.ANTIALIAS)
    return np.array(list(File.getdata()))

# Transform Images into Numpy Arrays by Size #

def Read(File, ScreenSize, PositionList):
    File.show()
    Content = input('Usable Content? (True of False): ')
    if str(Content) == 'True':
        CollectData = []
        Data = Compile(File, File.size[0], File.size[1])
        print(Data)
        DataLen = len(Data)
        Data_x = [Index%File.size[0] for Index in range(DataLen) if Data[Index] < 100]
        Data_y = [(Index-Index%File.size[0])/File.size[0] for Index in range(DataLen) if Data[Index] < 100]

        for _ in xrange(StandardDefault.ImageSize/2):
            Data_x_len, Data_y_len = len(Data_x), len(Data_y)
            
            Tree_List=spatial.KDTree(zip(Data_x, Data_y))
            Coordinate = rand(rand(0, int(round(Data_x_len/4))), rand(Data_x_len-int(round(Data_x_len/4)), Data_x_len))
            Tree_point=spatial.KDTree([[Data_x[Coordinate], Data_y[Coordinate]]])
            Tree_Neighbors = spatial.KDTree.query_ball_tree(Tree_point, Tree_List, r=StandardDefault.ImageSize/2, p=2.0)
            NeighborBoundary = spatial.KDTree.query_ball_tree(Tree_point, Tree_List, r=2*StandardDefault.ImageSize/3, p=2.0)
            NeighborBoundary = [Data[Item] for Item in NeighborBoundary if Item not in Tree_Neighbors[0]]
            NeighborBoundary = sum(NeighborBoundary[0])/len(NeighborBoundary[0])
            
            if len(Tree_Neighbors[0]) > 250 and len(Tree_Neighbors[0]) < 3.1416*((StandardDefault.ImageSize/2)**2)/4 and NeighborBoundary > 175:
                Averaged_x = sum([Data_x[Tree_Neighbors[0][Index]] for Index in range(len(Tree_Neighbors[0]))])/len(Tree_Neighbors[0])
                Averaged_y = sum([Data_y[Tree_Neighbors[0][Index]] for Index in range(len(Tree_Neighbors[0]))])/len(Tree_Neighbors[0])
                CollectData.append([len(Tree_Neighbors[0]), (Averaged_x, Averaged_y)])
                TestZipData = zip(Data_x, Data_y)
                TestZipData = [TestZipData[Coordinate] for Coordinate in range(len(TestZipData)) 
                if abs(TestZipData[Coordinate][0]-Averaged_x) > StandardDefault.ImageSize/10 and 
                abs(TestZipData[Coordinate][1]-Averaged_y) > StandardDefault.ImageSize/10]
                Data_x, Data_y = zip(*TestZipData)
            else:
                continue
        CollectData = sorted(CollectData, reverse=True)[0:10]
        print(CollectData)
        for Item in CollectData:
            Prediction = Network_test(File, Crop_Coordinates=Item[1], TrainedModel=load_model('TrainedModel')) #pickle.load(open('TrainedModel.sav', 'rb'))
            if PositionList[Prediction] == None:
                PositionList[Prediction] = []
            PositionList[Prediction].append(Item[1])
    PositionList = [PositionList[Index] for Index in range(StandardDefault.SearchFeatures) if PositionList[Index] != None]
    print(PositionList)

    if len(PositionList) > 0:
        for Item in PositionList:
            PositionList[PositionList.index(Item)] = StandardDefault.ReduceList(Item, 2 - int(round(PositionList.index(Item)/2)))
        print('Finished Position List: ' + str(PositionList))
        try:
            PositionList = spatial.distance.pdist(PositionList[0])
        except:
            PositionList = spatial.distance.pdist(PositionList)
        while len(PositionList) < StandardDefault.Expand(StandardDefault.SearchFeatures): 
            PositionList = np.append(PositionList, None)
        print('Spatial Position List: ' + str(PositionList))

        return PositionList
    print('Insufficient elements. Cannot use this Image!')
    return [None for Index in range(StandardDefault.Expand(StandardDefault.SearchFeatures))]

# Takes and Dissects Facial Images into lists #

if __name__ == "__main__":
    FeatureList = [CollectImages('Eye_Clipart', StandardDefault.NetworkSamples), 
                    CollectImages('SideNoseBlackandWhiteClipart', StandardDefault.NetworkSamples), 
                    CollectImages('Closed_Mouth_Clipart', StandardDefault.NetworkSamples)]
    if os.path.isfile('TrainedModel') != True:
        TrainedModel = Network_train(FeatureList).save('TrainedModel')
    FaceImg=DrawFace()
    DrawnNeighborList = Read(Image.open('FacePic.jpg'), StandardDefault.ScreenSize, PositionList=StandardDefault.Empty()) # Customize what to call your drawing
    if len(DrawnNeighborList) != 0:

        print('Looking for Faces...')
        FeatureNeighborList=[]

        for Index in range(10): # Note: Adjust this and the CollectImages() argument accordingly
            FeatureNeighborList.append(Read(Image.open(open(CollectImages('Front_View_Face_Drawing', 10)[Index])), 
            StandardDefault.ScreenSize, PositionList=StandardDefault.Empty()))

        Iteration = StandardDefault.Expand(StandardDefault.SearchFeatures)

        FeatureNeighborList = [FeatureNeighborList[Iteration*Index:(Iteration*Index)+Iteration] for Index in range(len(FeatureNeighborList)/Iteration)][0]
        FeatureNeighborList = [[Item/Set[0] if Set[0] and Item != None else Item for Item in Set] for Set in FeatureNeighborList]
        AvgFeatureList = [sum([FeatureNeighborList[Index][Item] for Index in range(len(FeatureNeighborList)) if FeatureNeighborList[Index][Item] != None])/len(FeatureNeighborList) for Item in range(StandardDefault.Expand(StandardDefault.SearchFeatures))]
        UnitDrawnNeighborList = [Item/DrawnNeighborList[0] if DrawnNeighborList[0] and Item != None else Item for Item in DrawnNeighborList]
        
        CorrectionList = [float(DrawnNeighborList[0]*(AvgFeatureList[0]-UnitDrawnNeighborList[0])) if AvgFeatureList[Index] and DrawnNeighborList[Index] != None else 0 for Index in range(StandardDefault.Expand(StandardDefault.SearchFeatures))]
        print('Here are some corrections you may love! List suggests changes in this order: \
            Distance Between Eyes, Distance Between First Eye and Nose, Distance Between First Eye and Mouth, Distance Between \
            Second Eye and Nose, Distance Between Second Eye and Mouth' + str(CorrectionList))
    else:
        print('Program Shutting Off')
        time.sleep(0.5)
    print('')
    print('___ Program has finished. ___')
    print('')

#### END PROGRAM ####
